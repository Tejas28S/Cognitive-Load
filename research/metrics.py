"""
metrics.py — every metric is computed here, from saved predictions only.

Nothing in this module knows what a "good" number looks like.  It takes
(y_true, y_pred, y_prob) and returns the full record.  `metrics.json` and the
frontend both read the output of this file, so a metric shown in the UI can
always be recomputed from predictions.csv.

Metric validity is decided by the data, not hardcoded per model: Top-K is only
emitted when K < n_classes, ROC-AUC OVO only when n_classes > 2, and anything
undefined for the given labels is reported as null with a stated reason rather
than silently dropped or faked.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Plain-English explanations shipped to the frontend so the UI never has to
# hardcode metric copy either.
METRIC_EXPLANATIONS = {
    "accuracy": "Percentage of predictions that are correct.",
    "balanced_accuracy": "Accuracy averaged over classes, so a large class cannot dominate the score.",
    "precision_macro": "Of the samples predicted as a class, how many actually belong to that class — averaged equally over classes.",
    "recall_macro": "Of the samples belonging to a class, how many the model correctly identifies — averaged equally over classes.",
    "f1_macro": "A balance between precision and recall, averaged equally over classes.",
    "precision_weighted": "Precision averaged over classes, weighting each class by how many samples it has.",
    "recall_weighted": "Recall averaged over classes, weighting each class by how many samples it has.",
    "f1_weighted": "F1 averaged over classes, weighting each class by how many samples it has.",
    "precision_micro": "Precision computed over all predictions pooled together.",
    "recall_micro": "Recall computed over all predictions pooled together.",
    "f1_micro": "F1 computed over all predictions pooled together; equals accuracy in single-label classification.",
    "cohen_kappa": "Measures agreement between predictions and true labels while accounting for agreement expected by chance.",
    "mcc": "A correlation-based classification measure that remains useful when class distributions are uneven.",
    "roc_auc_macro_ovr": "Measures how well the model separates classes across different decision thresholds, one class against the rest.",
    "roc_auc_macro_ovo": "Measures separation across thresholds averaged over every pair of classes. Only defined for more than two classes.",
    "average_precision_macro": "Area under the precision-recall curve, averaged over classes; sensitive to performance on the rarer class.",
    "log_loss": "Penalises confident wrong predictions; lower is better. Measures probability quality, not just the argmax.",
    "top_k_accuracy": "How often the true class is among the model's K highest-scoring classes. Not applicable for binary classification.",
    "brier_score": "Mean squared error of the predicted probabilities; lower is better.",
}


def _safe(fn, *args, **kw):
    """Run a metric, returning (value, error_reason)."""
    try:
        v = float(fn(*args, **kw))
        if not np.isfinite(v):
            return None, "metric evaluated to a non-finite value"
        return v, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def compute_metrics(y_true, y_pred, y_prob, n_classes, class_names=None):
    """Full metric record for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    # float32 softmax rows can miss 1.0 by ~1e-7, which log_loss warns about.
    row_sum = y_prob.sum(axis=1, keepdims=True)
    y_prob = np.divide(y_prob, row_sum, out=np.full_like(y_prob, 1.0 / n_classes),
                       where=row_sum > 0)
    labels = list(range(n_classes))
    class_names = class_names or [str(i) for i in labels]
    binary = n_classes == 2

    out = {"n_samples": int(len(y_true)), "n_classes": int(n_classes),
           "class_names": list(class_names), "unavailable": {}}

    def put(key, value, reason=None):
        out[key] = value
        if value is None and reason:
            out["unavailable"][key] = reason

    put("accuracy", *_safe(accuracy_score, y_true, y_pred))
    put("balanced_accuracy", *_safe(balanced_accuracy_score, y_true, y_pred))

    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)

    put("cohen_kappa", *_safe(cohen_kappa_score, y_true, y_pred, labels=labels))
    put("mcc", *_safe(matthews_corrcoef, y_true, y_pred))

    # ── probability-based metrics ──────────────────────────────────────
    present = np.unique(y_true)
    if len(present) < 2:
        reason = f"only one class present in this evaluation set ({present.tolist()})"
        for k in ("roc_auc_macro_ovr", "roc_auc_macro_ovo", "average_precision_macro"):
            put(k, None, reason)
    else:
        score = y_prob[:, 1] if binary else y_prob
        kw = {} if binary else {"multi_class": "ovr", "average": "macro", "labels": labels}
        put("roc_auc_macro_ovr", *_safe(roc_auc_score, y_true, score, **kw))
        if binary:
            put("roc_auc_macro_ovo", None,
                "one-vs-one is undefined for binary classification (identical to OVR)")
        else:
            put("roc_auc_macro_ovo", *_safe(
                roc_auc_score, y_true, y_prob,
                multi_class="ovo", average="macro", labels=labels))
        aps = []
        for c in labels:
            if (y_true == c).sum() == 0:
                continue
            from sklearn.metrics import average_precision_score
            v, _ = _safe(average_precision_score, (y_true == c).astype(int), y_prob[:, c])
            if v is not None:
                aps.append(v)
        put("average_precision_macro", float(np.mean(aps)) if aps else None,
            None if aps else "no class had positive samples")

    put("log_loss", *_safe(log_loss, y_true, y_prob, labels=labels))
    onehot = np.eye(n_classes)[y_true]
    out["brier_score"] = float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))

    # ── top-K: only meaningful when K < n_classes ──────────────────────
    out["top_k_accuracy"] = {}
    for k in (3, 5):
        if k < n_classes:
            from sklearn.metrics import top_k_accuracy_score
            v, err = _safe(top_k_accuracy_score, y_true, y_prob, k=k, labels=labels)
            out["top_k_accuracy"][f"top_{k}"] = v
        else:
            out["unavailable"][f"top_{k}_accuracy"] = (
                f"Not applicable: K={k} is not smaller than the number of classes "
                f"({n_classes})."
            )

    # ── per-class ──────────────────────────────────────────────────────
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn_ = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn_ - fp)
        per_class.append({
            "class_index": i, "class_name": name,
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(s[i]),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
        })
    out["per_class"] = per_class
    out["confusion_matrix"] = cm.astype(int).tolist()
    out["confusion_matrix_normalized"] = (
        cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ).tolist()
    out["support"] = {class_names[i]: int(s[i]) for i in labels}
    return out


def curve_points(y_true, y_prob, n_classes, class_names=None, max_points=300):
    """ROC and PR curves, one-vs-rest, thinned for transport to the browser."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    class_names = class_names or [str(i) for i in range(n_classes)]
    roc, pr = [], []

    def thin(*arrays):
        n = len(arrays[0])
        if n <= max_points:
            idx = np.arange(n)
        else:
            idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        return [np.asarray(a)[idx] for a in arrays]

    targets = [1] if n_classes == 2 else list(range(n_classes))
    for c in targets:
        yb = (y_true == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb, y_prob[:, c])
        fpr, tpr = thin(fpr, tpr)
        auc, _ = _safe(roc_auc_score, yb, y_prob[:, c])
        roc.append({
            "class_index": int(c), "class_name": class_names[c], "auc": auc,
            "points": [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)],
        })
        prec, rec, _ = precision_recall_curve(yb, y_prob[:, c])
        prec, rec = thin(prec, rec)
        from sklearn.metrics import average_precision_score
        ap, _ = _safe(average_precision_score, yb, y_prob[:, c])
        pr.append({
            "class_index": int(c), "class_name": class_names[c],
            "average_precision": ap, "baseline": float(yb.mean()),
            "points": [{"recall": float(a), "precision": float(b)}
                       for a, b in zip(rec, prec)],
        })
    return {"roc": roc, "pr": pr}


def per_subject_metrics(df, n_classes, class_names, prob_cols):
    """Metrics for each subject's own held-out predictions."""
    rows = []
    for subject, g in df.groupby("subject_id", sort=True):
        probs = g[prob_cols].to_numpy(float)
        m = compute_metrics(
            g["y_true"].to_numpy(int), g["y_pred"].to_numpy(int),
            probs, n_classes, class_names,
        )
        rows.append({
            "subject_id": int(subject),
            "fold": int(g["fold"].iloc[0]),
            "n_windows": int(len(g)),
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "balanced_accuracy": m["balanced_accuracy"],
            "cohen_kappa": m["cohen_kappa"],
            "mcc": m["mcc"],
            "roc_auc": m["roc_auc_macro_ovr"],
            "log_loss": m["log_loss"],
            "mean_confidence": float(probs.max(axis=1).mean()),
            "class_distribution": {
                class_names[c]: int((g["y_true"].to_numpy(int) == c).sum())
                for c in range(n_classes)
            },
        })
    return rows


def aggregate_folds(per_subject, keys=None):
    """Mean / std / min / max across LOSO folds — the cross-subject summary."""
    keys = keys or [
        "accuracy", "f1_macro", "precision_macro", "recall_macro",
        "balanced_accuracy", "cohen_kappa", "mcc", "roc_auc",
    ]
    out = {"n_folds": len(per_subject)}
    for k in keys:
        vals = np.array(
            [r[k] for r in per_subject if r.get(k) is not None], dtype=float
        )
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[k] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    return out


def metrics_to_rows(metrics):
    """Flatten the metric record for metrics.csv."""
    rows = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            rows.append({"metric": k, "value": v,
                         "explanation": METRIC_EXPLANATIONS.get(k, "")})
        elif v is None and k in METRIC_EXPLANATIONS:
            rows.append({"metric": k, "value": "",
                         "explanation": metrics.get("unavailable", {}).get(k, "")})
    for k, v in (metrics.get("top_k_accuracy") or {}).items():
        rows.append({"metric": f"top_k_accuracy.{k}", "value": v,
                     "explanation": METRIC_EXPLANATIONS["top_k_accuracy"]})
    for pc in metrics.get("per_class", []):
        for m in ("precision", "recall", "f1", "support"):
            rows.append({
                "metric": f"per_class.{pc['class_name']}.{m}",
                "value": pc[m],
                "explanation": METRIC_EXPLANATIONS.get(f"{m}_macro", ""),
            })
    return rows
