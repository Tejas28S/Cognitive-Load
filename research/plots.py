"""
plots.py — figures.

Every figure is written inside the run directory that produced it, so two runs
can never overwrite each other's plots.  All figures are drawn from the same
arrays the JSON metrics come from; nothing here recomputes or smooths numbers.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

EEG_C, FN_C = "#4C78A8", "#F58518"
OK_C, BAD_C = "#54A24B", "#E45756"


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _clean(seq):
    return np.array([np.nan if v is None else v for v in seq], dtype=float)


def loss_curves(mh, out_dir, prefix=""):
    ep = mh["epoch"]
    paths = []
    for name, keys, title, ylab in (
        ("training_loss", ["train_loss"], "Training loss (mean over folds)", "loss"),
        ("validation_loss", ["val_loss"], "Validation loss (mean over folds)", "loss"),
    ):
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for k in keys:
            ax.plot(ep, _clean(mh[k]), color=EEG_C, lw=1.8)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.set_title(title)
        paths.append(_save(fig, os.path.join(out_dir, f"{prefix}{name}.png")))

    # combined, which is the one that actually shows the gap
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ep, _clean(mh["train_loss"]), color=EEG_C, lw=1.8, label="train")
    ax.plot(ep, _clean(mh["val_loss"]), color=FN_C, lw=1.8, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Train vs validation loss (mean over folds)")
    ax.legend(frameon=False)
    paths.append(_save(fig, os.path.join(out_dir, f"{prefix}loss_curves.png")))
    return paths


def metric_curves(mh, out_dir, prefix=""):
    ep = mh["epoch"]
    metrics = [("accuracy", "accuracy"), ("f1_macro", "macro F1"),
               ("precision_macro", "macro precision"), ("recall_macro", "macro recall")]
    paths = []
    for split, title in (("train", "Training"), ("val", "Validation")):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for key, label in metrics:
            ax.plot(ep, _clean(mh[f"{split}_{key}"]), lw=1.6, label=label)
        ax.set_xlabel("epoch"); ax.set_ylabel("score")
        ax.set_title(f"{title} metrics (mean over folds)")
        ax.legend(frameon=False, fontsize=8)
        paths.append(_save(fig, os.path.join(out_dir, f"{prefix}{split}ing_metrics.png"
                                             if split == "train"
                                             else f"{prefix}validation_metrics.png")))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ep, _clean(mh["train_f1_macro"]), color=EEG_C, lw=1.8, label="train macro F1")
    ax.plot(ep, _clean(mh["val_f1_macro"]), color=FN_C, lw=1.8, label="validation macro F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("macro F1")
    ax.set_title("Generalization gap during training")
    ax.legend(frameon=False)
    paths.append(_save(fig, os.path.join(out_dir, f"{prefix}generalization_gap.png")))
    return paths


def confusion_matrix_fig(cm, class_names, out_path, title="Confusion matrix"):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6 + 0.4 * len(class_names),
                                    3.1 + 0.4 * len(class_names)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.1%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    return _save(fig, out_path)


def roc_fig(curves, out_path, title="ROC curve (held-out subjects)"):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for c in curves["roc"]:
        pts = c["points"]
        auc = c["auc"]
        ax.plot([p["fpr"] for p in pts], [p["tpr"] for p in pts], lw=1.8,
                label=f"{c['class_name']} (AUC {auc:.3f})" if auc else c["class_name"])
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999", label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, out_path)


def pr_fig(curves, out_path, title="Precision-Recall (held-out subjects)"):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for c in curves["pr"]:
        pts = c["points"]
        ap = c["average_precision"]
        ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], lw=1.8,
                label=f"{c['class_name']} (AP {ap:.3f})" if ap else c["class_name"])
        ax.axhline(c["baseline"], ls="--", lw=1, color="#999")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, out_path)


def per_subject_fig(per_subject, out_path, metric="f1_macro", chance=None):
    rows = sorted(per_subject, key=lambda r: r[metric] if r[metric] is not None else -1)
    names = [f"S{r['subject_id']}" for r in rows]
    vals = _clean([r[metric] for r in rows])
    mean = np.nanmean(vals)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.32 * len(rows)), 3.4))
    ax.bar(names, vals, color=[OK_C if v >= mean else BAD_C for v in vals])
    ax.axhline(mean, color="#333", lw=1.2, ls="-", label=f"mean {mean:.3f}")
    if chance is not None:
        ax.axhline(chance, color="#999", lw=1.2, ls="--", label=f"chance {chance:.2f}")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Per-subject held-out performance (LOSO)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_path)


def experiment_comparison_fig(records, out_path, metric="f1_macro"):
    rows = sorted(records, key=lambda r: r["fold_aggregate"][metric]["mean"] or 0)
    names = [r["experiment_id"] for r in rows]
    means = _clean([r["fold_aggregate"][metric]["mean"] for r in rows])
    stds = _clean([r["fold_aggregate"][metric]["std"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.34 * len(rows))))
    ax.barh(names, means, xerr=stds, color=EEG_C, alpha=0.85,
            error_kw={"ecolor": "#555", "lw": 1})
    ax.set_xlabel(f"{metric.replace('_', ' ')} (mean +/- std over folds)")
    ax.set_title("Experiment comparison — development LOSO")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, out_path)


def tsne_fig(emb2d, labels, label_names, out_path, title, palette=None):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    uniq = sorted(set(labels.tolist()))
    cmap = palette or plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 2)))
    for i, u in enumerate(uniq):
        m = labels == u
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=4, alpha=0.55,
                   color=cmap[i % len(cmap)],
                   label=str(label_names[u]) if isinstance(label_names, dict)
                   else str(label_names[i]) if label_names else str(u))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title)
    if len(uniq) <= 8:
        ax.legend(frameon=False, fontsize=8, markerscale=2.5)
    return _save(fig, out_path)


def shap_fig(ranking, out_path, top_n=25, title="SHAP feature importance"):
    rows = ranking[:top_n][::-1]
    names = [r["feature"] for r in rows]
    vals = [r["mean_abs_shap"] for r in rows]
    colors = [EEG_C if r["modality"] == "EEG" else FN_C for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.24 * len(rows))))
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=EEG_C),
               plt.Rectangle((0, 0), 1, 1, color=FN_C)]
    ax.legend(handles, ["EEG", "fNIRS"], frameon=False, fontsize=8)
    return _save(fig, out_path)


def shap_category_fig(categories, out_path, title="SHAP importance by feature family"):
    rows = sorted(categories, key=lambda r: r["mean_abs_shap"])
    fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.3 * len(rows))))
    ax.barh([r["category"] for r in rows], [r["mean_abs_shap"] for r in rows],
            color=[EEG_C if r["modality"] == "EEG" else FN_C for r in rows])
    ax.set_xlabel("mean |SHAP value| (summed over features in family)")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, out_path)


def modality_gate_fig(gates, out_path):
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.bar(["EEG", "fNIRS"], [gates["eeg_mean"], gates["fnirs_mean"]],
           color=[EEG_C, FN_C])
    ax.set_ylabel("mean fusion gate")
    ax.set_title("Learned modality attention")
    for i, v in enumerate([gates["eeg_mean"], gates["fnirs_mean"]]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return _save(fig, out_path)
