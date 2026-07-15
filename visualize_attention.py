import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data_loader import CognitiveLoadDataLoader
from models.architectures import build_model


@torch.no_grad()
def collect_attention_weights(model, loader, device):
    model.eval()
    eeg_weights, fnirs_weights, labels_all = [], [], []

    for eeg, fnirs, labels in loader:
        eeg = eeg.to(device)
        fnirs = fnirs.to(device)
        eeg_w, fnirs_w = model.get_attention_weights(eeg, fnirs)
        eeg_weights.extend(eeg_w.cpu().numpy())
        fnirs_weights.extend(fnirs_w.cpu().numpy())
        labels_all.extend(labels.numpy())

    return np.asarray(eeg_weights), np.asarray(fnirs_weights), np.asarray(labels_all)


def save_attention_summary(eeg_w, fnirs_w, labels, cfg, output_path):
    rows = [
        {
            "class": "All",
            "n": int(len(labels)),
            "eeg_mean": float(eeg_w.mean()),
            "eeg_std": float(eeg_w.std()),
            "fnirs_mean": float(fnirs_w.mean()),
            "fnirs_std": float(fnirs_w.std()),
        }
    ]
    for idx, name in enumerate(cfg.CLASS_NAMES):
        mask = labels == idx
        rows.append(
            {
                "class": name,
                "n": int(mask.sum()),
                "eeg_mean": float(eeg_w[mask].mean()),
                "eeg_std": float(eeg_w[mask].std()),
                "fnirs_mean": float(fnirs_w[mask].mean()),
                "fnirs_std": float(fnirs_w[mask].std()),
            }
        )

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_attention(eeg_w, fnirs_w, labels, cfg, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(eeg_w, bins=30, alpha=0.7, label="EEG gate")
    axes[0].hist(fnirs_w, bins=30, alpha=0.7, label="fNIRS gate")
    axes[0].set_title("Attention Gate Distribution")
    axes[0].set_xlabel("Mean gate value")
    axes[0].set_ylabel("Samples")
    axes[0].legend()

    x = np.arange(cfg.NUM_CLASSES)
    eeg_means = [eeg_w[labels == i].mean() for i in range(cfg.NUM_CLASSES)]
    fnirs_means = [fnirs_w[labels == i].mean() for i in range(cfg.NUM_CLASSES)]
    width = 0.35
    axes[1].bar(x - width / 2, eeg_means, width, label="EEG gate")
    axes[1].bar(x + width / 2, fnirs_means, width, label="fNIRS gate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cfg.CLASS_NAMES, rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Mean Gate by True Class")
    axes[1].set_ylabel("Mean gate value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Path to a trained hybrid model")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--summary-csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    cfg = Config()
    if cfg.FUSION_TYPE != "attention":
        raise ValueError("visualize_attention.py requires Config.FUSION_TYPE='attention'")

    model_path = args.model_path or os.path.join(
        cfg.OUTPUT_DIR,
        "phase3_per-subject_normalization_before_log.pt",
    )
    output_path = args.output or os.path.join(cfg.OUTPUT_DIR, "attention_weights_final.png")
    summary_path = args.summary_csv or os.path.join(cfg.OUTPUT_DIR, "attention_weights_final.csv")

    dl = CognitiveLoadDataLoader(cfg)
    _, _, test_loader, meta = dl.prepare()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("hybrid_cnn_lstm", meta["n_eeg_features"], meta["n_fnirs_features"], cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    eeg_w, fnirs_w, labels = collect_attention_weights(model, test_loader, device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_attention(eeg_w, fnirs_w, labels, cfg, output_path)
    save_attention_summary(eeg_w, fnirs_w, labels, cfg, summary_path)
    print(f"Saved attention visualization to {output_path}")
    print(f"Saved attention summary to {summary_path}")


if __name__ == "__main__":
    main()
