# ============================================================
#  evaluate.py  –  Metrics, confusion matrix, plots
# ============================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score, recall_score
)
from config import Config

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})


# ── Inference ─────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device, is_unimodal=False, modality="eeg"):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for eeg, fnirs, labels in loader:
        eeg, fnirs = eeg.to(device), fnirs.to(device)
        inp    = eeg if (is_unimodal and modality == "eeg") else \
                 (fnirs if is_unimodal else None)
        logits = model(inp) if is_unimodal else model(eeg, fnirs)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.vstack(all_probs), np.array(all_labels)


# ── Full evaluation report ────────────────────────────────────

def evaluate_model(model, test_loader, cfg: Config = None,
                   model_name="Hybrid CNN-LSTM",
                   is_unimodal=False, modality="eeg"):
    cfg    = cfg or Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    preds, probs, labels = predict(model, test_loader, device,
                                   is_unimodal, modality)

    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro")
    f1_wt  = f1_score(labels, preds, average="weighted")
    per_class_recall = recall_score(
        labels,
        preds,
        labels=list(range(cfg.NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    print(f"\n{'═'*55}")
    print(f"  {model_name}  –  Test Results")
    print(f"{'═'*55}")
    print(f"  Accuracy        : {acc:.4f}  ({acc:.2%})")
    print(f"  Macro F1        : {f1_mac:.4f}")
    print(f"  Weighted F1     : {f1_wt:.4f}")
    print(f"  ROC-AUC (macro) : {auc:.4f}")
    print(
        "  Per-class recall: "
        + ", ".join(
            f"{name}={recall:.2%}"
            for name, recall in zip(cfg.CLASS_NAMES, per_class_recall)
        )
    )
    print(f"\n  Per-class report:")
    print(classification_report(labels, preds,
                                 target_names=cfg.CLASS_NAMES, digits=4))

    return {"acc": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt, "auc": auc,
            "per_class_recall": per_class_recall,
            "preds": preds, "probs": probs, "labels": labels}


# ── Plot helpers ──────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, cfg, title, ax=None):
    cm   = confusion_matrix(labels, preds)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=cfg.CLASS_NAMES,
                yticklabels=cfg.CLASS_NAMES, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")


def plot_training_history(history, title="Training History", ax=None):
    fig_local = None
    if ax is None:
        fig_local, ax = plt.subplots(1, 2, figsize=(12, 4))
    axes = ax if hasattr(ax, '__len__') else [ax]

    ax[0].plot(history["train_loss"], label="Train", linewidth=2)
    ax[0].plot(history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    ax[0].set_title(f"{title} – Loss"); ax[0].set_xlabel("Epoch")
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot([a*100 for a in history["train_acc"]], label="Train", linewidth=2)
    ax[1].plot([a*100 for a in history["val_acc"]],   label="Val",   linewidth=2, linestyle="--")
    ax[1].set_title(f"{title} – Accuracy"); ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)"); ax[1].legend(); ax[1].grid(True)

    return fig_local


def plot_comparison_bar(results_dict, metric="acc", cfg=None):
    """Bar chart comparing multiple model results."""
    cfg    = cfg or Config()
    names  = list(results_dict.keys())
    values = [results_dict[n][metric] * 100 for n in names]
    colors = sns.color_palette("muted", len(names))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_title("Model Comparison – " + metric.upper())
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return fig


# ── Master dashboard ──────────────────────────────────────────

def save_evaluation_dashboard(history, results_dict, cfg=None, out_dir="outputs/"):
    """
    Saves a single PNG with:
      - Training curves (hybrid model)
      - Confusion matrices for all models
      - Bar chart comparison
    """
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)

    n_models = len(results_dict)
    fig = plt.figure(figsize=(6 * n_models, 14))
    gs  = gridspec.GridSpec(3, n_models, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: training curves (span all columns for hybrid)
    ax_loss = fig.add_subplot(gs[0, :n_models//2 or 1])
    ax_acc  = fig.add_subplot(gs[0, n_models//2 or 1:] if n_models > 1 else gs[0, :])
    hybrid_hist = history.get("hybrid_cnn_lstm") or list(history.values())[0]
    ax_loss.plot(hybrid_hist["train_loss"], label="Train", lw=2)
    ax_loss.plot(hybrid_hist["val_loss"],   label="Val",   lw=2, ls="--")
    ax_loss.set_title("Hybrid Model – Loss"); ax_loss.legend(); ax_loss.grid(True)

    ax_acc.plot([a*100 for a in hybrid_hist["train_acc"]], label="Train", lw=2)
    ax_acc.plot([a*100 for a in hybrid_hist["val_acc"]],   label="Val",   lw=2, ls="--")
    ax_acc.set_title("Hybrid Model – Accuracy"); ax_acc.set_ylabel("Acc (%)")
    ax_acc.legend(); ax_acc.grid(True)

    # Row 1: confusion matrices
    for i, (name, res) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[1, i])
        plot_confusion_matrix(res["labels"], res["preds"], cfg, name, ax=ax)

    # Row 2: bar chart (span all)
    ax_bar = fig.add_subplot(gs[2, :])
    names  = list(results_dict.keys())
    accs   = [results_dict[n]["acc"] * 100 for n in names]
    f1s    = [results_dict[n]["f1_macro"] * 100 for n in names]
    x      = np.arange(len(names)); w = 0.35
    bars1  = ax_bar.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0")
    bars2  = ax_bar.bar(x + w/2, f1s,  w, label="Macro F1",  color="#DD8452")
    ax_bar.bar_label(bars1, fmt="%.1f%%", padding=2, fontsize=9)
    ax_bar.bar_label(bars2, fmt="%.1f%%", padding=2, fontsize=9)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(names)
    ax_bar.set_ylim(0, 115); ax_bar.set_ylabel("Score (%)")
    ax_bar.set_title("Model Comparison"); ax_bar.legend(); ax_bar.grid(axis="y", alpha=0.4)

    path = os.path.join(out_dir, "evaluation_dashboard.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📊 Dashboard saved → {path}")
    return path
