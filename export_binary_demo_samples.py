"""
export_binary_demo_samples.py - rebuild serving/demo_samples.json from the
REAL, current held-out test subjects only.

The shipped demo_samples.json's subject list (1-17, sequential) does not match
any of the current train/val/test partitions - it mixes genuinely held-out
subjects with subjects the served model was trained on, and drops two of the
six real test subjects entirely. That makes the Live Demo dropdown look like
it covers more of the cohort than it honestly can: predictions on a trained
subject look artificially close to perfect, while the genuinely unseen
subjects show the real ~62% accuracy - a leakage-shaped illusion the rest of
this site goes out of its way to avoid (see the LOSO / lockbox framing
throughout Methodology and Research).

This rebuilds demo_samples.json from ONLY the current test-set subjects
(computed via the same dl.split_indices() the served model's split came from),
using the ALREADY-FITTED scaler baked into serving/feature_scaler.pkl (not a
freshly refit one, and not outputs/hybrid_best.pt, which has since drifted to
a 3-class checkpoint and would silently produce a wrong bundle) - so the
output is guaranteed consistent with what serving/model_weights.pt actually
expects.

    .venv/Scripts/python export_binary_demo_samples.py
"""
import copy
import json
import os

import joblib
import numpy as np
import pandas as pd

from config import Config
from data_loader import CognitiveLoadDataLoader

SERVING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving")
OUT_PATH = os.path.join(SERVING_DIR, "demo_samples.json")
LABELS = ["Low", "High"]


def binary_config():
    cfg = copy.copy(Config())
    cfg.BINARY_CLASSIFICATION = True
    cfg.CLASS_NAMES = ["Low (0-back)", "High (3-back)"]
    cfg.NUM_CLASSES = 2
    cfg.CLASS_WEIGHTS = [1.0, 1.0]
    cfg.PER_SUBJECT_NORMALIZE = True
    cfg.PER_SUBJECT_NORMALIZE_STAGE = "before_log"
    cfg.FUSION_TYPE = "attention"
    cfg.LOSS_FN = "weighted_ce"
    return cfg


def main():
    cfg = binary_config()
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, groups = dl.load_raw(return_groups=True)
    subjects = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])[
        cfg.SUBJECT_COLUMN
    ].astype(str).to_numpy()
    tr, va, te = dl.split_indices(labels, groups)

    test_subjects = sorted(set(int(s) for s in subjects[te]))
    print(f"Real held-out test subjects: {test_subjects}")

    scaler_bundle = joblib.load(os.path.join(SERVING_DIR, "feature_scaler.pkl"))
    eeg_scaler = scaler_bundle["eeg_scaler"]
    fnirs_scaler = scaler_bundle["fnirs_scaler"]

    eeg_te = eeg_scaler.transform(eeg_raw[te])
    fnirs_te = fnirs_scaler.transform(fnirs_raw[te])
    subj_te = subjects[te]
    y_te = labels[te]

    samples = []
    for s in test_subjects:
        mask = subj_te.astype(int) == s
        rows = np.where(mask)[0]
        for window_index, r in enumerate(rows):
            samples.append({
                "subject_id": s,
                "trial_id": window_index,
                "window_index": window_index,
                "timestamp_s": window_index * 10,
                "true_label": int(y_te[r]),
                "true_label_name": LABELS[int(y_te[r])],
                "features": (eeg_te[r].tolist() + fnirs_te[r].tolist()),
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_eeg": int(eeg_raw.shape[1]),
            "n_fnirs": int(fnirs_raw.shape[1]),
            "note": (
                "Real subject-held-out fixed-split test windows, fully "
                "preprocessed with the scaler baked into serving/feature_scaler.pkl "
                "(per-subject normalization is a demo-mode simplification using "
                "each known subject's own stats). Rebuilt to include every "
                "genuinely held-out test subject (previously only a partial, "
                "train/test-mixed subset of subjects 1-17 was shipped)."
            ),
            "samples": samples,
        }, f)

    print(f"Wrote {OUT_PATH}: {len(samples)} windows across {len(test_subjects)} held-out subjects")


if __name__ == "__main__":
    main()
