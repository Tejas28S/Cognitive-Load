"""
train.py — training loop for one fold.

Model selection inside a fold uses the VALIDATION split only (subjects that
are neither the held-out subject nor in the lockbox).  The held-out subject is
touched exactly once, after `restore_best`, to produce that fold's
predictions.  There is no path in this file that can read test labels.

Includes:
  * early stopping on validation macro F1 with best-weight restore
  * ReduceLROnPlateau / CosineAnnealingLR / none
  * weight decay, dropout, gaussian feature noise, modality dropout
  * optional DANN with the standard Ganin lambda ramp
  * train-mode-off metrics on the training split each epoch, so the
    generalization gap is measured without dropout/noise distorting it
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from .archs import build_model, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name="cpu"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def make_loader(eeg, fnirs, y, domains, batch_size, shuffle, drop_last=False):
    tensors = [
        torch.as_tensor(eeg, dtype=torch.float32),
        torch.as_tensor(fnirs, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    ]
    if domains is not None:
        tensors.append(torch.as_tensor(domains, dtype=torch.long))
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size,
        shuffle=shuffle, drop_last=drop_last, num_workers=0,
    )


def class_weights_from(y, n_classes, mode="balanced"):
    if mode in (None, "none"):
        return None
    counts = np.bincount(np.asarray(y).astype(int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, n_classes=2):
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        eeg, fnirs, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits = model(eeg, fnirs)
        if criterion is not None:
            loss_sum += float(criterion(logits, y).detach()) * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs = np.vstack(probs)
    labels = np.concatenate(labels)
    preds = probs.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_classes)), average="macro", zero_division=0
    )
    return {
        "loss": loss_sum / max(n, 1) if criterion is not None else None,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "probs": probs,
        "labels": labels,
        "preds": preds,
    }


def _dann_lambda(spec, progress):
    """Ganin & Lempitsky ramp: 0 -> lambda_max as training progresses."""
    lam_max = float(spec.get("dann_lambda", 0.0))
    if not spec.get("dann_ramp", True):
        return lam_max
    return lam_max * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def train_fold(spec, schema, n_classes, split, domains=None, seed=42,
               device="cpu", log=print):
    """Train one fold. `split` holds (eeg, fnirs, y) for train/val/test."""
    set_seed(seed)
    device = resolve_device(device)

    eeg_tr, fn_tr, y_tr = split["train"]
    eeg_va, fn_va, y_va = split["val"]

    dom_tr = domains["train"] if domains else None
    n_domains = int(domains["n_domains"]) if domains else 0

    model = build_model(spec, schema, n_classes).to(device)
    use_dann = bool(spec.get("use_dann")) and n_domains > 1
    if use_dann:
        model.attach_domain_head(
            n_domains, hidden=int(spec.get("dann_hidden", 64)),
            dropout=float(spec.get("dann_dropout", 0.5)),
        ).to(device)

    bs = int(spec.get("batch_size", 64))
    tr_loader = make_loader(eeg_tr, fn_tr, y_tr, dom_tr, bs, True, drop_last=True)
    tr_eval_loader = make_loader(eeg_tr, fn_tr, y_tr, None, 512, False)
    va_loader = make_loader(eeg_va, fn_va, y_va, None, 512, False)

    cw = class_weights_from(y_tr, n_classes, spec.get("class_weights", "balanced"))
    criterion = nn.CrossEntropyLoss(
        weight=cw.to(device) if cw is not None else None,
        label_smoothing=float(spec.get("label_smoothing", 0.0)),
    )
    domain_criterion = nn.CrossEntropyLoss()

    opt_name = spec.get("optimizer", "adamw").lower()
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt_name]
    optimizer = opt_cls(
        model.parameters(),
        lr=float(spec.get("lr", 3e-4)),
        weight_decay=float(spec.get("weight_decay", 0.01)),
    )

    max_epochs = int(spec.get("max_epochs", 60))
    sched_name = str(spec.get("scheduler", "plateau")).lower()
    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            patience=int(spec.get("lr_patience", 5)),
            factor=float(spec.get("lr_factor", 0.5)),
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    patience = int(spec.get("early_stop_patience", 12))
    min_delta = float(spec.get("early_stop_min_delta", 1e-4))
    noise = float(spec.get("augment_noise", 0.0))

    history = {k: [] for k in (
        "epoch", "lr", "train_loss", "train_accuracy", "train_precision_macro",
        "train_recall_macro", "train_f1_macro", "val_loss", "val_accuracy",
        "val_precision_macro", "val_recall_macro", "val_f1_macro",
        "domain_loss", "dann_lambda",
    )}

    best_score, best_state, best_epoch, bad = -np.inf, None, 0, 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        progress = (epoch - 1) / max(max_epochs - 1, 1)
        lam = _dann_lambda(spec, progress) if use_dann else 0.0
        dom_loss_sum, dom_n = 0.0, 0

        for batch in tr_loader:
            eeg, fnirs, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            dom = batch[3].to(device) if len(batch) > 3 else None
            optimizer.zero_grad()

            if noise > 0:
                eeg = eeg + torch.randn_like(eeg) * noise
                fnirs = fnirs + torch.randn_like(fnirs) * noise

            if use_dann and dom is not None:
                logits, dom_logits = model.forward_dann(eeg, fnirs, lambda_=lam)
                d_loss = domain_criterion(dom_logits, dom)
                loss = criterion(logits, y) + d_loss
                dom_loss_sum += float(d_loss.detach()) * y.size(0)
                dom_n += y.size(0)
            else:
                loss = criterion(model(eeg, fnirs), y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        tr = evaluate(model, tr_eval_loader, device, criterion, n_classes)
        va = evaluate(model, va_loader, device, criterion, n_classes)

        history["epoch"].append(epoch)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["domain_loss"].append(dom_loss_sum / dom_n if dom_n else None)
        history["dann_lambda"].append(float(lam))
        for k in ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"):
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])

        score = va["f1_macro"]  # checkpoint metric: validation macro F1
        if scheduler is not None:
            scheduler.step(score) if sched_name == "plateau" else scheduler.step()

        if score > best_score + min_delta:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                log(f"      early stop @ epoch {epoch} (best epoch {best_epoch}, "
                    f"val macro F1 {best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_score),
        "epochs_run": len(history["epoch"]),
        "train_seconds": float(time.time() - t0),
        "n_params": count_params(model),
        "device": str(device),
    }


@torch.no_grad()
def predict(model, eeg, fnirs, device="cpu", batch_size=512):
    device = resolve_device(device) if isinstance(device, str) else device
    model.eval().to(device)
    out = []
    for i in range(0, len(eeg), batch_size):
        e = torch.as_tensor(eeg[i:i + batch_size], dtype=torch.float32, device=device)
        f = torch.as_tensor(fnirs[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(torch.softmax(model(e, f), dim=-1).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def modality_gates(model, eeg, fnirs, device="cpu", batch_size=512):
    device = resolve_device(device) if isinstance(device, str) else device
    model.eval().to(device)
    eg, fg = [], []
    for i in range(0, len(eeg), batch_size):
        e = torch.as_tensor(eeg[i:i + batch_size], dtype=torch.float32, device=device)
        f = torch.as_tensor(fnirs[i:i + batch_size], dtype=torch.float32, device=device)
        g = model.gates(e, f)
        if g is None:
            return None
        eg.append(g[0].cpu().numpy())
        fg.append(g[1].cpu().numpy())
    return np.concatenate(eg), np.concatenate(fg)


@torch.no_grad()
def embeddings(model, eeg, fnirs, device="cpu", batch_size=512):
    device = resolve_device(device) if isinstance(device, str) else device
    model.eval().to(device)
    out = []
    for i in range(0, len(eeg), batch_size):
        e = torch.as_tensor(eeg[i:i + batch_size], dtype=torch.float32, device=device)
        f = torch.as_tensor(fnirs[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(model.embeddings(e, f).cpu().numpy())
    return np.vstack(out)
