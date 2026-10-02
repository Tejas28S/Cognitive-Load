"""
export_binary_shap.py - rewrite serving/frontend_data/shap.json with a REAL
feature-category importance for the binary (Low/High) model.

True SHAP (the `shap` package's DeepExplainer/GradientExplainer, used by
explain_model.py) can't run in this environment: its `numba` dependency fails
to import under an OS Application Control policy that blocks the native DLL
(see the export_diagnostics.py / ModelInsights.jsx notes from the same
investigation). Rather than leave the panel showing the old 3-class numbers
(a real bug fixed by clearing shap.json - see git history) or leave it empty
indefinitely, this computes Integrated Gradients (Sundararajan et al., 2017)
- a closely related, axiomatically-grounded attribution method that needs
only plain PyTorch autograd, no shap/numba. It answers the same question
("how much does each feature family move the model's output") on the exact
real served model (backend.model_service.ModelService + the current
serving/model_weights.pt) and the real held-out demo windows.

Per-feature attribution: IG_i(x) = (x_i - baseline_i) * average gradient of
the target class probability along the straight path from baseline to x.
Baseline = the feature-wise mean over the held-out demo set (a standard,
practical choice absent direct access to the training background here).

Aggregated into the same feature-category taxonomy explain_model.py uses
(EEG band / fNIRS HbO|HbR statistic), and written in the same JSON shape the
frontend already expects (categories + per_class), so no schema changes are
needed downstream - just real numbers.

    .venv/Scripts/python export_binary_shap.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "backend")
from model_service import ModelService  # noqa: E402
from explain_model import feature_category  # noqa: E402

SERVING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving")
OUT_PATH = os.path.join(SERVING_DIR, "frontend_data", "shap.json")
N_STEPS = 32
BATCH_SIZE = 32


def main():
    svc = ModelService(serving_dir=SERVING_DIR)
    with open(os.path.join(SERVING_DIR, "demo_samples.json"), encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    with open(os.path.join(SERVING_DIR, "feature_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)

    feature_names = list(schema["eeg_columns"]) + list(schema["fnirs_columns"])
    categories = [feature_category(n) for n in feature_names]

    X = np.asarray([s["features"] for s in samples], dtype=np.float32)
    n_eeg = svc.n_eeg
    n_classes = len(svc.class_names)
    n_features = X.shape[1]
    device = svc.device
    model = svc.model
    model.eval()

    baseline = X.mean(axis=0, keepdims=True)
    baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device)

    abs_ig_sum = np.zeros((n_classes, n_features), dtype=np.float64)
    n_done = 0

    for start in range(0, len(X), BATCH_SIZE):
        batch = X[start : start + BATCH_SIZE]
        x_t = torch.tensor(batch, dtype=torch.float32, device=device)
        b = batch.shape[0]
        diff = x_t - baseline_t

        for cls in range(n_classes):
            grad_accum = torch.zeros_like(x_t)
            for step in range(1, N_STEPS + 1):
                alpha = step / N_STEPS
                interp = (baseline_t + alpha * diff).clone().requires_grad_(True)
                eeg = interp[:, :n_eeg]
                fnirs = interp[:, n_eeg:]
                probs = torch.softmax(model(eeg, fnirs), dim=-1)
                target = probs[:, cls].sum()
                (grad,) = torch.autograd.grad(target, interp)
                grad_accum += grad
            avg_grad = grad_accum / N_STEPS
            ig = (diff * avg_grad).detach().cpu().numpy()
            abs_ig_sum[cls, :] += np.abs(ig).sum(axis=0)

        n_done += b
        print(f"  {n_done}/{len(X)} windows", end="\r")

    print()
    magnitudes = abs_ig_sum / len(X)  # (n_classes, n_features), mean |IG| per feature

    rows = []
    for cls_idx, cls_name in enumerate(svc.class_names):
        for category in sorted(set(categories)):
            mask = np.asarray([c == category for c in categories])
            rows.append({
                "class": cls_name,
                "category": category,
                "mean_abs_shap": float(magnitudes[cls_idx][mask].sum()),
                "features": int(mask.sum()),
            })
    df = pd.DataFrame(rows)
    total = df.groupby("class")["mean_abs_shap"].transform("sum")
    df["share"] = df["mean_abs_shap"] / total.replace(0, np.nan)

    overall = (
        df.groupby("category", as_index=False)["mean_abs_shap"].mean()
        .sort_values("mean_abs_shap", ascending=False)
    )
    per_class = {}
    for cls_name in svc.class_names:
        sub = df[df["class"] == cls_name].sort_values("mean_abs_shap", ascending=False)
        per_class[cls_name] = [
            {"category": r.category, "mean_abs_shap": float(r.mean_abs_shap),
             "share": float(r.share) if not pd.isna(r.share) else None}
            for r in sub.itertuples()
        ]

    out = {
        "method": "integrated_gradients",
        "method_note": (
            "True SHAP (DeepExplainer/GradientExplainer) is blocked in this "
            "environment (shap depends on numba, which fails to load under an "
            "OS Application Control policy here). These are Integrated Gradients "
            "(Sundararajan et al., 2017) computed directly on the real served "
            "binary model (serving/model_weights.pt) over the real held-out demo "
            "windows - a closely related, axiomatically-grounded attribution "
            "method, not a placeholder or carried-over 3-class value."
        ),
        "n_windows": len(samples),
        "baseline": "feature-wise mean over the held-out demo set",
        "categories": [
            {"category": r.category, "mean_abs_shap": float(r.mean_abs_shap)}
            for r in overall.itertuples()
        ],
        "per_class": per_class,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print("Top categories overall:")
    for r in overall.sort_values("mean_abs_shap", ascending=False).head(8).itertuples():
        print(f"  {r.category:<20} {r.mean_abs_shap:.4f}")


if __name__ == "__main__":
    main()
