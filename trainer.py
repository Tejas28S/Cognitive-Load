# ============================================================
#  trainer.py  –  Training engine with tqdm progress bars
# ============================================================

import os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import Config


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_idx = targets.view(-1, 1)
        log_pt = log_probs.gather(1, target_idx).squeeze(1)
        pt = probs.gather(1, target_idx).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, targets)
        return loss.mean()


def build_criterion(cfg, class_weights, device):
    weights = class_weights.to(device) if class_weights is not None else None
    loss_name = getattr(cfg, "LOSS_FN", "weighted_ce").lower()
    label_smoothing = getattr(cfg, "LABEL_SMOOTHING", 0.0)

    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(alpha=weights, gamma=float(getattr(cfg, "FOCAL_GAMMA", 2.0)))
    raise ValueError("Config.LOSS_FN must be 'weighted_ce' or 'focal'")


def build_optimizer(model, cfg):
    optimizer_name = getattr(cfg, "OPTIMIZER", "adam").lower()
    kwargs = {
        "lr": cfg.LEARNING_RATE,
        "weight_decay": cfg.WEIGHT_DECAY,
    }

    if optimizer_name == "adam":
        return Adam(model.parameters(), **kwargs)
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), **kwargs)
    if optimizer_name == "lion":
        from lion_pytorch import Lion
        return Lion(model.parameters(), **kwargs)
    raise ValueError("Config.OPTIMIZER must be 'adam', 'adamw', or 'lion'")


# ── Early stopping ────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, path="outputs/best_model.pt",
                 mode="min", metric_name="val_loss"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.mode       = mode
        self.metric_name = metric_name
        self.best_score = -np.inf if mode == "max" else np.inf
        self.counter    = 0
        self.best_state = None

    def _is_improvement(self, score):
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def __call__(self, score, model):
        if self._is_improvement(score):
            self.best_score = score
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, self.path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ── One epoch ─────────────────────────────────────────────────

def mixup_data(eeg, fnirs, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = eeg.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_eeg = lam * eeg + (1 - lam) * eeg[index, :]
    mixed_fnirs = lam * fnirs + (1 - lam) * fnirs[index, :]
    y_a, y_b = y, y[index]
    return mixed_eeg, mixed_fnirs, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def run_epoch(model, loader, criterion, optimizer, device,
              training=True, is_unimodal=False, modality="eeg",
              epoch_bar=None, cfg=None):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    mixup_alpha = getattr(cfg, "MIXUP_ALPHA", 0.0) if training and cfg else 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            eeg, fnirs, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            domains = batch[3].to(device) if len(batch) > 3 else None

            if training:
                optimizer.zero_grad()

            if training and getattr(cfg, "AUGMENT_NOISE", 0.0) > 0:
                eeg = eeg + torch.randn_like(eeg) * cfg.AUGMENT_NOISE
                fnirs = fnirs + torch.randn_like(fnirs) * cfg.AUGMENT_NOISE

            apply_mixup = training and mixup_alpha > 0 and not is_unimodal
            if apply_mixup:
                eeg, fnirs, targets_a, targets_b, lam = mixup_data(eeg, fnirs, labels, mixup_alpha, device)

            use_dann = training and getattr(cfg, "USE_DANN", False) and domains is not None
            if use_dann:
                task_logits, domain_logits = model.forward_dann(eeg, fnirs, lambda_=getattr(cfg, "DANN_LAMBDA", 0.1))
                loss_task = criterion(task_logits, labels)
                loss_domain = nn.CrossEntropyLoss()(domain_logits, domains)
                loss = loss_task + loss_domain
                logits = task_logits
            else:
                if is_unimodal:
                    inp    = eeg if modality == "eeg" else fnirs
                    logits = model(inp)
                else:
                    logits = model(eeg, fnirs)

                if apply_mixup:
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                else:
                    loss = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            
            if apply_mixup:
                correct += (lam * (preds == targets_a).sum().item() + (1 - lam) * (preds == targets_b).sum().item())
            else:
                correct += (preds == labels).sum().item()
            
            total += labels.size(0)

            if epoch_bar:
                epoch_bar.set_postfix({
                    "loss": f"{total_loss/total:.4f}",
                    "acc" : f"{correct/total:.2%}"
                }, refresh=False)
                epoch_bar.update(1)

    return total_loss / total, correct / total


# ── Single train run ──────────────────────────────────────────

def train_model(model, train_loader, val_loader, meta,
                cfg: Config = None, verbose=True,
                is_unimodal=False, modality="eeg"):
    cfg    = cfg or Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    if getattr(cfg, "USE_DANN", False) and "n_domains" in meta:
        model.attach_domain_head(n_domains=meta["n_domains"]).to(device)

    cw        = meta.get("class_weights")
    criterion = build_criterion(cfg, cw, device)
    optimizer = build_optimizer(model, cfg)
    
    lr_scheduler_type = getattr(cfg, "LR_SCHEDULER", "plateau").lower()
    metric = getattr(cfg, "CHECKPOINT_METRIC", "val_f1_macro")
    mode = "min" if "loss" in metric else "max"
    
    if lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode=mode,
                                      patience=cfg.LR_PATIENCE, factor=0.5)
    es = EarlyStopping(
        patience=cfg.EARLY_STOP,
        path=cfg.MODEL_PATH,
        mode=mode,
        metric_name=metric,
    )

    history = {"train_loss": [], "val_loss": [],
               "train_acc" : [], "val_acc" : [],
               "val_f1_macro": [], "val_auc": []}

    total_params = sum(p.numel() for p in model.parameters())
    gpu_name     = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  Device  : {device}  ({gpu_name})")
        print(f"  Model   : {model.__class__.__name__}  ({total_params:,} params)")
        print(f"  Loss    : {getattr(cfg, 'LOSS_FN', 'weighted_ce')}")
        print(f"  Optim   : {getattr(cfg, 'OPTIMIZER', 'adam')}  wd={cfg.WEIGHT_DECAY:g}")
        print(f"  Batches : {len(train_loader)} train / {len(val_loader)} val")
        print(f"{'─'*65}")

    # ── Epoch-level progress bar ──────────────────────────────
    epoch_pbar = tqdm(
        range(1, cfg.EPOCHS + 1),
        desc   = f"  Training",
        unit   = "epoch",
        ncols  = 80,
        colour = "cyan",
    )

    for epoch in epoch_pbar:
        # ── Batch-level bar (train) ───────────────────────────
        batch_bar = tqdm(
            total   = len(train_loader),
            desc    = f"    Ep {epoch:>3}/{cfg.EPOCHS}",
            leave   = False,
            ncols   = 75,
            unit    = "batch",
            colour  = "green",
        )

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, training=True,
                                    is_unimodal=is_unimodal, modality=modality,
                                    epoch_bar=batch_bar, cfg=cfg)
        batch_bar.close()

        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    optimizer, device, training=False,
                                    is_unimodal=is_unimodal, modality=modality)
        va_metrics = _validation_metrics(
            model, val_loader, device,
            is_unimodal=is_unimodal, modality=modality,
        )
        checkpoint_score = va_metrics["f1_macro"] if "f1_macro" in metric else va_loss
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(checkpoint_score)
        else:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1_macro"].append(checkpoint_score)
        history["val_auc"].append(va_metrics["auc"])

        lr_now = optimizer.param_groups[0]['lr']

        # Update epoch bar suffix with live stats
        epoch_pbar.set_postfix({
            "tr_loss" : f"{tr_loss:.4f}",
            "tr_acc"  : f"{tr_acc:.2%}",
            "va_loss" : f"{va_loss:.4f}",
            "va_acc"  : f"{va_acc:.2%}",
            "va_f1"   : f"{checkpoint_score:.4f}",
            "lr"      : f"{lr_now:.0e}",
            "es"      : f"{es.counter}/{cfg.EARLY_STOP}",
        })

        if es(checkpoint_score, model):
            epoch_pbar.write(
                f"\n  ⏹  Early stop @ epoch {epoch} | "
                f"best val score ({metric}): {es.best_score:.4f} | "
                f"best val acc: {max(history['val_acc']):.2%}"
            )
            break

    epoch_pbar.close()
    es.restore(model)

    if verbose:
        print(f"\n  ✅ Training complete")
        print(f"     Best val accuracy : {max(history['val_acc']):.2%}")
        print(f"     Best val macro F1 : {max(history['val_f1_macro']):.4f}")
        print(f"     Best val loss     : {min(history['val_loss']):.4f}")
        print(f"{'─'*65}")

    return model, history


@torch.no_grad()
def _validation_metrics(model, loader, device, is_unimodal=False, modality="eeg"):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        eeg, fnirs, labels = batch[0].to(device), batch[1].to(device), batch[2]
        if is_unimodal:
            inp = eeg if modality == "eeg" else fnirs
            logits = model(inp)
        else:
            logits = model(eeg, fnirs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.asarray(all_labels)
    preds = probs.argmax(axis=1)
    metrics = {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        metrics["auc"] = float("nan")
    return metrics


# ── Cross-validation run ──────────────────────────────────────

def cross_validate(model_builder_fn, data_loader_obj,
                   cfg: Config = None, verbose=True):
    cfg = cfg or Config()
    fold_results = []

    print(f"\n{'═'*65}")
    print(f"  {cfg.CV_FOLDS}-Fold Stratified Cross-Validation")
    print(f"{'═'*65}")

    for fold, tr_loader, va_loader, fold_meta in data_loader_obj.cv_folds():
        print(f"\n  ▶  Fold {fold}/{cfg.CV_FOLDS}")
        model = model_builder_fn(fold_meta)
        model, history = train_model(model, tr_loader, va_loader,
                                     fold_meta, cfg, verbose=False)

        best_val_acc  = max(history["val_acc"])
        best_val_loss = min(history["val_loss"])
        best_val_f1 = max(history["val_f1_macro"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = _validation_metrics(model.to(device), va_loader, device)
        fold_results.append({
            "fold": fold,
            "val_acc": metrics["acc"],
            "val_f1_macro": metrics["f1_macro"],
            "val_auc": metrics["auc"],
            "best_val_acc": best_val_acc,
            "best_val_f1_macro": best_val_f1,
            "val_loss": best_val_loss,
        })
        print(
            f"     Fold {fold} → Acc: {metrics['acc']:.2%}  "
            f"Macro F1: {metrics['f1_macro']:.4f}  AUC: {metrics['auc']:.4f}"
        )

    accs = np.array([r["val_acc"] for r in fold_results])
    f1s = np.array([r["val_f1_macro"] for r in fold_results])
    aucs = np.array([r["val_auc"] for r in fold_results], dtype=float)
    print(f"\n{'─'*65}")
    print(f"  CV Accuracy : {np.mean(accs):.2%} +/- {np.std(accs):.2%}")
    print(f"  CV Macro F1 : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  CV AUC      : {np.nanmean(aucs):.4f} +/- {np.nanstd(aucs):.4f}")
    print(f"{'═'*65}")
    return fold_results
