"""
explain.py — SHAP, t-SNE and the subject-invariance probe.

All three are computed from the ACTUAL trained model of the run that calls
them.  Nothing is cached across runs and nothing is read from the legacy
outputs/ artifacts, so a value shown on the Evaluation page always belongs to
the run selected in the UI.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn


class _ConcatWrapper(nn.Module):
    """Present the two-input model as a single-input one for SHAP."""

    def __init__(self, model, n_eeg):
        super().__init__()
        self.model = model
        self.n_eeg = n_eeg

    def forward(self, x):
        return self.model(x[:, : self.n_eeg], x[:, self.n_eeg:])


def feature_names(schema):
    return list(schema.eeg_cols) + list(schema.fnirs_cols)


def feature_modality(name):
    if name.startswith("eeg_"):
        return "EEG"
    return "fNIRS"


def feature_category(name):
    """Group a column into a readable feature family."""
    m = re.match(r"^eeg_ch\d+_(.+)$", name)
    if m:
        return f"EEG {m.group(1).replace('_', ' ')}"
    m = re.match(r"^(hbo|hbr)_ch\d+_(.+)$", name)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    return name


def feature_channel(name):
    m = re.match(r"^(?:eeg|hbo|hbr)_ch(\d+)_", name)
    return int(m.group(1)) if m else None


def compute_shap(model, eeg_bg, fn_bg, eeg_ex, fn_ex, schema, class_names,
                 device="cpu", max_background=80, max_explain=160, log=print):
    """Mean |SHAP| per feature, aggregated to families, channels and modality.

    Uses GradientExplainer: DeepExplainer fails its additivity check on this
    architecture (BatchNorm + gated fusion), which the project already
    documented.
    """
    import shap

    dev = torch.device(device)
    model = model.to(dev).eval()
    n_eeg = len(schema.eeg_cols)
    wrapped = _ConcatWrapper(model, n_eeg).to(dev).eval()

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(eeg_bg), min(max_background, len(eeg_bg)), replace=False)
    ex_idx = rng.choice(len(eeg_ex), min(max_explain, len(eeg_ex)), replace=False)

    background = torch.tensor(
        np.hstack([eeg_bg[bg_idx], fn_bg[bg_idx]]), dtype=torch.float32, device=dev
    )
    to_explain = torch.tensor(
        np.hstack([eeg_ex[ex_idx], fn_ex[ex_idx]]), dtype=torch.float32, device=dev
    )

    explainer = shap.GradientExplainer(wrapped, background)
    values = explainer.shap_values(to_explain)

    # Normalise shap's return shape to (n_classes, n_samples, n_features).
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=0)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:                 # (n_samples, n_features, n_classes)
            arr = np.transpose(arr, (2, 0, 1))
        else:                             # (n_samples, n_features)
            arr = arr[None, ...]

    mean_abs = np.abs(arr).mean(axis=1)           # (n_classes, n_features)
    overall = mean_abs.mean(axis=0)               # (n_features,)
    names = feature_names(schema)

    ranking = [
        {
            "rank": i + 1,
            "feature": names[j],
            "modality": feature_modality(names[j]),
            "category": feature_category(names[j]),
            "channel": feature_channel(names[j]),
            "mean_abs_shap": float(overall[j]),
            "per_class": {
                class_names[c]: float(mean_abs[c, j]) for c in range(mean_abs.shape[0])
            },
        }
        for i, j in enumerate(np.argsort(-overall))
    ]

    def _group(keyfn):
        acc = {}
        for j, nm in enumerate(names):
            k = keyfn(nm)
            if k is None:
                continue
            acc.setdefault(k, {"sum": 0.0, "n": 0, "modality": feature_modality(nm)})
            acc[k]["sum"] += float(overall[j])
            acc[k]["n"] += 1
        return acc

    cats = _group(feature_category)
    categories = sorted(
        [
            {"category": k, "modality": v["modality"], "n_features": v["n"],
             "mean_abs_shap": v["sum"], "mean_per_feature": v["sum"] / v["n"]}
            for k, v in cats.items()
        ],
        key=lambda r: -r["mean_abs_shap"],
    )

    eeg_total = float(overall[:n_eeg].sum())
    fn_total = float(overall[n_eeg:].sum())
    n_hbo = len(schema.hbo_cols)
    hbo_total = float(overall[n_eeg:n_eeg + n_hbo].sum())
    hbr_total = float(overall[n_eeg + n_hbo:].sum())
    total = eeg_total + fn_total

    chan = _group(lambda nm: (feature_modality(nm), feature_channel(nm))
                  if feature_channel(nm) is not None else None)
    channels = sorted(
        [
            {"modality": k[0], "channel": k[1], "mean_abs_shap": v["sum"]}
            for k, v in chan.items()
        ],
        key=lambda r: -r["mean_abs_shap"],
    )

    return {
        "method": "shap.GradientExplainer",
        "n_background": int(len(bg_idx)),
        "n_explained": int(len(ex_idx)),
        "class_names": list(class_names),
        "ranking": ranking,
        "top_features": ranking[:40],
        "categories": categories,
        "channels": channels[:40],
        "modality_share": {
            "EEG": eeg_total / total if total else None,
            "fNIRS": fn_total / total if total else None,
            "HbO": hbo_total / total if total else None,
            "HbR": hbr_total / total if total else None,
        },
        "modality_total_abs_shap": {
            "EEG": eeg_total, "fNIRS": fn_total, "HbO": hbo_total, "HbR": hbr_total,
        },
    }


def compute_tsne(matrix, seed=42, perplexity=30, max_points=2500):
    """2-D t-SNE of a representation. Returns (coords, row indices used)."""
    from sklearn.manifold import TSNE

    n = len(matrix)
    rng = np.random.default_rng(seed)
    idx = (
        np.sort(rng.choice(n, max_points, replace=False)) if n > max_points
        else np.arange(n)
    )
    x = np.asarray(matrix, dtype=np.float64)[idx]
    perp = float(min(perplexity, max(5, (len(x) - 1) / 3)))
    coords = TSNE(
        n_components=2, perplexity=perp, init="pca",
        learning_rate="auto", random_state=seed, max_iter=750,
    ).fit_transform(x)
    return coords, idx


def subject_invariance_probe(emb, subjects, seed=42):
    """Can a fresh classifier still recover WHO from the representation?

    Lower accuracy = more subject-invariant. Compared against the chance rate
    of predicting the most frequent subject.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    uniq, counts = np.unique(subjects, return_counts=True)
    if len(uniq) < 2:
        return None
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return None
    # Stratify AND shuffle: the rows arrive grouped by subject, so a plain
    # KFold would test on subjects the probe never saw and score below chance,
    # which measures the split, not the representation.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, emb, subjects, cv=cv, scoring="accuracy")
    return {
        "probe": "logistic regression on the fused embedding",
        "cv_folds": n_splits,
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "chance_uniform": float(1.0 / len(uniq)),
        "chance_majority": float(counts.max() / counts.sum()),
        "n_subjects": int(len(uniq)),
        "interpretation": (
            "Accuracy near chance means subject identity is largely absent from "
            "the representation; well above chance means the encoder still "
            "carries who the person is."
        ),
    }
