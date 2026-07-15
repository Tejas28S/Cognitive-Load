import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import Config
from data_loader import CognitiveLoadDataLoader
from models.architectures import build_model


EEG_BANDS = ("delta", "theta", "alpha", "beta", "gamma", "engage")


def stratified_indices(labels, max_n, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per_class = max(1, int(np.ceil(max_n / len(classes))))
    chosen = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        take = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > max_n:
        chosen = rng.choice(chosen, size=max_n, replace=False)
    return np.sort(chosen)


def collect_loader(loader):
    eeg, fnirs, labels = [], [], []
    for eeg_batch, fnirs_batch, y_batch in loader:
        eeg.append(eeg_batch)
        fnirs.append(fnirs_batch)
        labels.append(y_batch)
    return torch.cat(eeg), torch.cat(fnirs), torch.cat(labels).numpy()


def feature_category(name):
    lower = name.lower()
    if lower.startswith("eeg_"):
        for band in EEG_BANDS:
            if re.search(rf"(^|_){band}($|_)", lower):
                return f"EEG {band}"
        return "EEG other"

    match = re.match(r"^(hbo|hbr)_ch\d+_(.+)$", lower)
    if match:
        chroma, stat = match.groups()
        return f"fNIRS {chroma.upper()} {stat}"
    return "fNIRS other"


def normalize_shap_output(raw_values):
    """Return [class][input] arrays for SHAP's several PyTorch return shapes."""
    if isinstance(raw_values, tuple):
        raw_values = list(raw_values)

    if isinstance(raw_values, list):
        if raw_values and isinstance(raw_values[0], (list, tuple)):
            return [[np.asarray(inp) for inp in class_values] for class_values in raw_values]
        if len(raw_values) == 2 and np.asarray(raw_values[0]).ndim >= 3:
            arrays = [np.asarray(v) for v in raw_values]
            class_axis = arrays[0].ndim - 1
            return [
                [np.take(arr, cls, axis=class_axis) for arr in arrays]
                for cls in range(arrays[0].shape[class_axis])
            ]
        return [[np.asarray(v)] for v in raw_values]

    arr = np.asarray(raw_values)
    if arr.ndim >= 3:
        class_axis = arr.ndim - 1
        return [[np.take(arr, cls, axis=class_axis)] for cls in range(arr.shape[class_axis])]
    return [[arr]]


class KernelWrapper:
    def __init__(self, model, n_eeg, device):
        self.model = model
        self.n_eeg = n_eeg
        self.device = device

    def __call__(self, flat):
        flat = np.asarray(flat, dtype=np.float32)
        eeg = torch.tensor(flat[:, : self.n_eeg], dtype=torch.float32, device=self.device)
        fnirs = torch.tensor(flat[:, self.n_eeg :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(eeg, fnirs)
            return torch.softmax(logits, dim=-1).cpu().numpy()


def compute_shap(model, bg_eeg, bg_fnirs, test_eeg, test_fnirs, device):
    import shap

    bg = [bg_eeg.to(device), bg_fnirs.to(device)]
    test = [test_eeg.to(device), test_fnirs.to(device)]

    for name, explainer_cls in (
        ("DeepExplainer", shap.DeepExplainer),
        ("GradientExplainer", shap.GradientExplainer),
    ):
        try:
            explainer = explainer_cls(model, bg)
            values = explainer.shap_values(test)
            return name, normalize_shap_output(values)
        except Exception as exc:
            print(f"{name} failed: {exc}")

    print("Falling back to KernelExplainer on a reduced sample.")
    n_bg = min(30, len(bg_eeg))
    n_test = min(60, len(test_eeg))
    bg_flat = torch.cat([bg_eeg[:n_bg], bg_fnirs[:n_bg]], dim=1).numpy()
    test_flat = torch.cat([test_eeg[:n_test], test_fnirs[:n_test]], dim=1).numpy()
    wrapper = KernelWrapper(model, bg_eeg.shape[1], device)
    explainer = shap.KernelExplainer(wrapper, bg_flat)
    values = explainer.shap_values(test_flat, nsamples=100)
    class_values = normalize_shap_output(values)
    split_values = []
    n_eeg = bg_eeg.shape[1]
    for class_item in class_values:
        flat_values = class_item[0]
        split_values.append([flat_values[:, :n_eeg], flat_values[:, n_eeg:]])
    return "KernelExplainer", split_values


def aggregate_shap(class_values, eeg_cols, fnirs_cols, class_names):
    feature_names = list(eeg_cols) + list(fnirs_cols)
    categories = [feature_category(name) for name in feature_names]
    rows = []

    for class_idx, inputs in enumerate(class_values):
        if len(inputs) != 2:
            raise ValueError("Expected SHAP values for EEG and fNIRS inputs")
        values = np.concatenate([np.asarray(inputs[0]), np.asarray(inputs[1])], axis=1)
        magnitudes = np.abs(values).mean(axis=0)
        for category in sorted(set(categories)):
            mask = np.asarray(categories) == category
            rows.append(
                {
                    "class": class_names[class_idx],
                    "category": category,
                    "mean_abs_shap": float(magnitudes[mask].sum()),
                    "features": int(mask.sum()),
                }
            )

    df = pd.DataFrame(rows)
    total = df.groupby("class")["mean_abs_shap"].transform("sum")
    df["share"] = df["mean_abs_shap"] / total.replace(0, np.nan)
    return df.sort_values(["class", "mean_abs_shap"], ascending=[True, False])


def plot_summary(agg, output_path, top_n=18):
    summary = (
        agg.groupby("category", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .sort_values("mean_abs_shap")
    )
    colors = ["#4C72B0" if cat.startswith("EEG") else "#55A868" for cat in summary["category"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(summary["category"], summary["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean absolute SHAP contribution")
    ax.set_title("Final Hybrid Model SHAP Summary by Feature Category")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_per_class(agg, output_path, top_n=10):
    classes = list(agg["class"].drop_duplicates())
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 4 * len(classes)))
    if len(classes) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, classes):
        subset = (
            agg[agg["class"] == class_name]
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .sort_values("mean_abs_shap")
        )
        colors = ["#4C72B0" if cat.startswith("EEG") else "#55A868" for cat in subset["category"]]
        ax.barh(subset["category"], subset["mean_abs_shap"], color=colors)
        ax.set_title(class_name)
        ax.set_xlabel("Mean absolute SHAP contribution")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Per-Class SHAP Drivers", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join("outputs", "phase3_per-subject_normalization_before_log.pt"),
    )
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--output-summary", default=os.path.join("outputs", "shap_summary.png"))
    parser.add_argument("--output-per-class", default=os.path.join("outputs", "shap_per_class.png"))
    parser.add_argument("--output-csv", default=os.path.join("outputs", "shap_category_importance.csv"))
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    dl = CognitiveLoadDataLoader(cfg)
    train_loader, _, test_loader, meta = dl.prepare()
    train_eeg, train_fnirs, train_labels = collect_loader(train_loader)
    test_eeg, test_fnirs, test_labels = collect_loader(test_loader)

    bg_idx = stratified_indices(train_labels, args.background_size, cfg.RANDOM_SEED)
    test_idx = stratified_indices(test_labels, args.test_size, cfg.RANDOM_SEED + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("hybrid_cnn_lstm", meta["n_eeg_features"], meta["n_fnirs_features"], cfg)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()

    method, class_values = compute_shap(
        model,
        train_eeg[bg_idx],
        train_fnirs[bg_idx],
        test_eeg[test_idx],
        test_fnirs[test_idx],
        device,
    )
    print(f"SHAP method used: {method}")

    agg = aggregate_shap(class_values, dl.eeg_cols, dl.fnirs_cols, cfg.CLASS_NAMES)
    agg.to_csv(args.output_csv, index=False)
    plot_summary(agg, args.output_summary)
    plot_per_class(agg, args.output_per_class)

    top = agg.groupby("category")["mean_abs_shap"].mean().sort_values(ascending=False).head(8)
    print("Top SHAP categories:")
    for category, value in top.items():
        print(f"  {category}: {value:.6f}")
    print(f"Saved {args.output_summary}")
    print(f"Saved {args.output_per_class}")
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
