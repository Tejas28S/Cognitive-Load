"""
Prototype the EEG+HbO+HbR (936-feature) served ensemble.

The full 26-fold LOSO ablation (outputs/hbr_ablation.json) showed adding HbR
back HELPS: Hybrid macro-F1 +0.087 (p=0.002), served ensemble +0.046
(t p=0.035, Wilcoxon p=0.059). This script builds a *prototype* of the served
Hybrid+RF ensemble on the 936 config so we can evaluate it end-to-end before
deciding whether to switch the live model.

It does NOT touch the live serving/ directory. Everything is written to
serving_936/, and a fresh Hybrid checkpoint is trained on the 936 config using
the SAME recipe as the frozen 468 checkpoint (fixed split, Config defaults:
per-subject z-score before log, attention fusion, weighted CE).

Usage: python prototype_936_serving.py [--skip-train]
Outputs:
    outputs/phase3_hbr936_before_log.pt   trained 936 Hybrid checkpoint
    serving_936/                          prototype serving artifacts
"""

import argparse
import os

import numpy as np
import torch

import config as config_mod
from part1_rigor import clone_config
from data_loader import CognitiveLoadDataLoader
from models.architectures import build_model
from trainer import train_model
from evaluate import evaluate_model
import export_for_serving as E

CKPT = os.path.join("outputs", "phase3_hbr936_before_log.pt")
SERVING_936 = "serving_936"
HBO_HBR = list(config_mod._hbo_cols) + list(config_mod._hbr_cols)


def cfg_936():
    """Config defaults (the frozen-checkpoint recipe) + HbR columns -> 936 fNIRS."""
    return clone_config({"MODALITY2_COLUMNS": HBO_HBR})


def train_checkpoint():
    cfg = cfg_936()
    cfg.MODEL_PATH = CKPT
    os.makedirs("outputs", exist_ok=True)
    dl = CognitiveLoadDataLoader(cfg)
    train_loader, val_loader, test_loader, meta = dl.prepare()
    print(f"Training 936 Hybrid: n_eeg={meta['n_eeg_features']} "
          f"n_fnirs={meta['n_fnirs_features']}")
    assert meta["n_fnirs_features"] == 936, meta["n_fnirs_features"]
    model = build_model("hybrid_cnn_lstm", meta["n_eeg_features"],
                        meta["n_fnirs_features"], cfg)
    model, _ = train_model(model, train_loader, val_loader, meta, cfg)
    torch.save(model.state_dict(), CKPT)
    evaluate_model(model, test_loader, cfg, model_name="Hybrid EEG+HbO+HbR (936)")
    print(f"Saved checkpoint -> {CKPT}")


def export_prototype():
    """Reuse export_for_serving against the 936 config + new checkpoint,
    redirected to serving_936/. Skips frontend_data (SHAP/attention/confusion
    are still the 468 model's; regenerating them is a separate step)."""
    device = torch.device("cpu")
    E.FROZEN_HYBRID = CKPT
    E.SERVING = SERVING_936
    E.FRONTEND = os.path.join(SERVING_936, "frontend_data")

    cfg = E.final_config()
    cfg.MODALITY2_COLUMNS = HBO_HBR
    fs = E.build_fixed_split(cfg)
    assert fs["n_fnirs"] == 936, fs["n_fnirs"]

    model, mc = E.export_model_and_scaler(cfg, fs, device)
    rf = E.export_rf(fs)
    E.export_demo_samples(fs)
    E.export_upload_schema(cfg, fs)
    E.export_upload_template(cfg)
    E.verify_ensemble(cfg, model, rf, fs, device)
    print(f"\nPrototype export complete -> {SERVING_936}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse an existing 936 checkpoint")
    args = ap.parse_args()
    if not args.skip_train:
        train_checkpoint()
    elif not os.path.exists(CKPT):
        raise SystemExit(f"--skip-train but no checkpoint at {CKPT}")
    export_prototype()


if __name__ == "__main__":
    main()
