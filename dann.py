"""
Domain-Adversarial Neural Network (DANN) training for subject-invariant
cognitive-load classification.

Motivation: this project's documented bottleneck is *subject variability* —
LOSO generalization sits far below the within-subject ceiling because the
model latches onto person-specific physiological baselines that don't transfer
to unseen people. DANN attacks that directly: a gradient-reversal
subject-discriminator is trained on the fused embedding so the representation
is pushed to predict cognitive LOAD while being uninformative about WHO the
subject is (Ganin & Lempitsky, 2015).

This module is self-contained — it does NOT touch the shared trainer.py loop
used by serving/eval. It provides:
    - train_dann(...)  : dual-loss training loop with a ramped GRL lambda
    - subject_probe(...): the subject-invariance diagnostic (can a fresh
      classifier still recover subject identity from the learned features?)

The heavy machinery (GRL, DomainDiscriminator, HybridCNNLSTM.forward_dann) lives
in models/architectures.py; this module orchestrates the training.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from config import Config
from trainer import build_criterion, build_optimizer


# ── Dataset carrying domain (subject) labels ──────────────────

class DANNDataset(Dataset):
    def __init__(self, eeg, fnirs, labels, domains):
        self.eeg = torch.as_tensor(eeg, dtype=torch.float32)
        self.fnirs = torch.as_tensor(fnirs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.int64)
        self.domains = torch.as_tensor(domains, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.eeg[i], self.fnirs[i], self.labels[i], self.domains[i]


def make_dann_loader(eeg, fnirs, labels, domains, cfg, shuffle=False, drop_last=False):
    return DataLoader(
        DANNDataset(eeg, fnirs, labels, domains),
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


# ── Lambda schedule (Ganin) ───────────────────────────────────

def lambda_schedule(progress, gamma=10.0, lambda_max=1.0):
    """p in [0,1] -> lambda in [0, lambda_max] via 2/(1+exp(-gamma*p)) - 1."""
    p = min(max(progress, 0.0), 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


# ── Task-head validation metric ───────────────────────────────

@torch.no_grad()
def _val_f1(model, val_loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for eeg, fnirs, labels, _ in val_loader:
        eeg, fnirs = eeg.to(device), fnirs.to(device)
        logits = model(eeg, fnirs)
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(labels.numpy())
    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)
    return f1_score(np.asarray(all_labels), preds, average="macro")


# ── DANN training loop ────────────────────────────────────────

def train_dann(model, train_loader, val_loader, meta, cfg, device,
               n_domains, verbose=False):
    """Train `model` (a HybridCNNLSTM) with an adversarial subject head.

    Loaders must yield (eeg, fnirs, label, domain). The domain head is
    attached here. Returns the model restored to its best val-macro-F1 state.
    """
    cfg = cfg or Config()
    model = model.to(device)
    model.attach_domain_head(
        n_domains,
        hidden=int(getattr(cfg, "DANN_HIDDEN", 64)),
        dropout=float(getattr(cfg, "DANN_DOMAIN_DROPOUT", 0.5)),
    )
    model = model.to(device)

    task_criterion = build_criterion(cfg, meta.get("class_weights"), device)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.LR_PATIENCE, factor=0.5
    )

    gamma = float(getattr(cfg, "DANN_GAMMA", 10.0))
    lambda_max = float(getattr(cfg, "DANN_LAMBDA_MAX", 1.0))
    total_epochs = int(getattr(cfg, "DANN_EPOCHS", cfg.EPOCHS))
    patience = cfg.EARLY_STOP

    best_f1, best_state, counter = -np.inf, None, 0
    n_batches = max(1, len(train_loader))

    for epoch in range(total_epochs):
        model.train()
        dom_correct, dom_total = 0, 0
        for bi, (eeg, fnirs, labels, domains) in enumerate(train_loader):
            eeg = eeg.to(device); fnirs = fnirs.to(device)
            labels = labels.to(device); domains = domains.to(device)

            progress = (epoch + bi / n_batches) / max(1, total_epochs)
            lam = lambda_schedule(progress, gamma, lambda_max)

            optimizer.zero_grad()
            task_logits, domain_logits = model.forward_dann(eeg, fnirs, lam)
            task_loss = task_criterion(task_logits, labels)
            domain_loss = domain_criterion(domain_logits, domains)
            loss = task_loss + domain_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            dom_correct += (domain_logits.argmax(1) == domains).sum().item()
            dom_total += domains.size(0)

        val_f1 = _val_f1(model, val_loader, device)
        scheduler.step(val_f1)

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if verbose and (epoch % 10 == 0 or counter >= patience):
            print(f"    dann ep {epoch:3d} lam={lam:.3f} "
                  f"val_f1={val_f1:.4f} best={best_f1:.4f} "
                  f"dom_acc={dom_correct/max(1,dom_total):.3f} es={counter}/{patience}")
        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_f1": float(best_f1)}


# ── Subject-invariance diagnostic ─────────────────────────────

@torch.no_grad()
def _embeddings(model, eeg, fnirs, device, batch=256):
    model.eval()
    eeg_t = torch.as_tensor(eeg, dtype=torch.float32)
    fnirs_t = torch.as_tensor(fnirs, dtype=torch.float32)
    out = []
    for i in range(0, len(eeg_t), batch):
        e = eeg_t[i:i + batch].to(device)
        f = fnirs_t[i:i + batch].to(device)
        out.append(model.get_embeddings(e, f).cpu().numpy())
    return np.vstack(out)


def subject_probe(model, eeg, fnirs, subjects, device, seed=42, test_size=0.3):
    """Can a fresh classifier recover subject identity from the learned
    representation? Lower accuracy == more subject-invariant.

    Trains a logistic regression on the model's fused embeddings to predict
    subject id, on a stratified split of the SAME subjects. Returns test
    accuracy, balanced accuracy, and the chance level (1/n_subjects).
    """
    subjects = np.asarray(subjects)
    uniq, y = np.unique(subjects, return_inverse=True)
    n_sub = len(uniq)
    if n_sub < 2:
        return {"probe_acc": float("nan"), "probe_balanced_acc": float("nan"),
                "chance": float("nan"), "n_subjects": int(n_sub)}

    X = _embeddings(model, eeg, fnirs, device)
    # Need >=2 samples per class for a stratified split; drop singletons.
    counts = np.bincount(y)
    keep = np.isin(y, np.where(counts >= 2)[0])
    Xk, yk = X[keep], y[keep]
    n_sub_eff = len(np.unique(yk))
    if n_sub_eff < 2:
        return {"probe_acc": float("nan"), "probe_balanced_acc": float("nan"),
                "chance": 1.0 / n_sub, "n_subjects": int(n_sub)}

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(sss.split(Xk, yk))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xk[tr], yk[tr])
    pred = clf.predict(Xk[te])
    return {
        "probe_acc": float(accuracy_score(yk[te], pred)),
        "probe_balanced_acc": float(balanced_accuracy_score(yk[te], pred)),
        "chance": float(1.0 / n_sub_eff),
        "n_subjects": int(n_sub_eff),
    }
