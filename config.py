"""
Tuned configuration for the windowed TU Berlin EEG+fNIRS dataset.

Use after generating data/tuberlin_sliding.csv:
    cp config.py config_backup.py
    cp config_improved.py config.py
"""

import os
import re
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _detect_cols(csv_path, prefix):
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, nrows=0)
    return sorted([c for c in df.columns if c.startswith(prefix)], key=_natural_key)


_CSV = os.environ.get("COGLOAD_CSV", "data/tuberlin_sliding_augmented.csv")
_eeg_cols = _detect_cols(_CSV, "eeg_")
_hbo_cols = _detect_cols(_CSV, "hbo_")
_hbr_cols = _detect_cols(_CSV, "hbr_")
_fnirs_cols = sorted(_hbo_cols + _hbr_cols, key=_natural_key)
_has_eeg = len(_eeg_cols) > 0
_has_eeg_sequence = _has_eeg and all(c.startswith("eeg_t") for c in _eeg_cols)

if os.path.exists(_CSV):
    print(f"[Config] CSV: {_CSV}")
    print(f"[Config] EEG features : {len(_eeg_cols)}")
    print(f"[Config] HbO features : {len(_hbo_cols)}")
    print(f"[Config] HbR features : {len(_hbr_cols)}")
    print("[Config] Split grouping: subject_id + trial_id")
else:
    print(f"[Config] CSV not found yet: {_CSV}")


class Config:
    DATA_MODE = "merged"
    MERGED_CSV = _CSV

    LABEL_COLUMN = "Cognitive_Load"
    SUBJECT_COLUMN = "subject_id"
    SPLIT_GROUP_COLUMNS = ["subject_id"]

    BINARY_CLASSIFICATION = False

    if BINARY_CLASSIFICATION:
        CLASS_NAMES = ["Low (0-back)", "High (3-back)"]
        NUM_CLASSES = 2
    else:
        CLASS_NAMES = ["Low (0-back)", "Medium (2-back)", "High (3-back)"]
        NUM_CLASSES = 3

    EEG_COLUMNS = _eeg_cols if _has_eeg else _hbo_cols
    MODALITY2_COLUMNS = _fnirs_cols if _has_eeg else _hbr_cols

    EEG_FEATURE_PREFIX = "eeg_" if _has_eeg else "hbo_"
    FNIRS_FEATURE_PREFIX = "hbo_" if _has_eeg else "hbr_"

    IGNORE_COLUMNS = [
        "subject_id",
        "trial_id",
        "window_id",
        "window_start_sec",
        "window_sec",
        "source_trial_start_ms",
        "task",
    ]

    TEST_SIZE = 0.20
    VAL_SIZE = 0.20
    RANDOM_SEED = 42
    NORMALIZE = True
    EEG_LOG_TRANSFORM = True
    EEG_LOG_MODE = "signed" if _has_eeg_sequence else "clip"
    FNIRS_SIGNED_LOG_TRANSFORM = True

    CHUNK_LOAD = False
    CHUNK_SIZE = 10_000

    MODEL_TYPE = "hybrid_cnn_lstm"

    CNN_FILTERS = [32, 16]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.5

    LSTM_UNITS = [32, 16]
    LSTM_DROPOUT = 0.5

    FUSION_TYPE = "attention"
    FUSION_DIM = 48

    FC_UNITS = [16]
    FC_DROPOUT = 0.6

    EPOCHS = 35
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    OPTIMIZER = "adamw"
    WEIGHT_DECAY = 0.01
    LR_PATIENCE = 4
    EARLY_STOP = 7
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHT_MODE = "manual"  # "manual", "balanced", or "balanced_medium_boost"
    if getattr(sys.modules[__name__], "BINARY_CLASSIFICATION", False): # Use getattr since we defined it inside class Config
        CLASS_WEIGHTS = [1.0, 1.0]
    else:
        CLASS_WEIGHTS = [1.08, 1.30, 1.10]
    MEDIUM_CLASS_WEIGHT_BOOST = 1.30
    LOSS_FN = "weighted_ce"  # "weighted_ce" or "focal"
    FOCAL_GAMMA = 2.0
    CHECKPOINT_METRIC = "val_loss"

    # DANN & Augmentation
    USE_DANN = True
    DANN_LAMBDA = 0.1
    AUGMENT_NOISE = 0.05

    RUN_CV = False
    CV_FOLDS = 5

    OUTPUT_DIR = "outputs/"
    SAVE_MODEL = True
    MODEL_PATH = "outputs/best_model_3class.pt"
    PRINT_EEG_LSTM_SHAPE = True
