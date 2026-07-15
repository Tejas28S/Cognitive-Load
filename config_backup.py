# ============================================================
#  config.py  –  TU Berlin Shin 2018  (654 samples)
#  Fixed for small dataset: tiny model + strong regularization
# ============================================================

import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _detect_cols(csv_path, prefix):
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, nrows=0)
    return sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda x: (int(x.split('_')[2]) if len(x.split('_')) > 2
                       and x.split('_')[2].isdigit() else 0, x)
    )

_CSV      = "data/tuberlin_eeg_fnirs.csv"
_eeg_cols = _detect_cols(_CSV, "eeg_")
_hbo_cols = _detect_cols(_CSV, "hbo_")
_hbr_cols = _detect_cols(_CSV, "hbr_")
_has_eeg  = len(_eeg_cols) > 0

if os.path.exists(_CSV):
    print(f"[Config] CSV: {_CSV}")
    print(f"[Config] EEG features  : {len(_eeg_cols)}  {'✅' if _has_eeg else '❌'}")
    print(f"[Config] HbO features  : {len(_hbo_cols)}")
    print(f"[Config] HbR features  : {len(_hbr_cols)}")
    if _has_eeg:
        print(f"[Config] Mode: TRUE MULTIMODAL (EEG + fNIRS)")
        print(f"[Config] Modality 1 → EEG ({len(_eeg_cols)} features)")
        print(f"[Config] Modality 2 → HbO ({len(_hbo_cols)} features)")
    else:
        print(f"[Config] Mode: fNIRS ONLY (HbO + HbR)")
        print(f"[Config] Modality 1 → HbO ({len(_hbo_cols)} features)")
        print(f"[Config] Modality 2 → HbR ({len(_hbr_cols)} features)")
else:
    print(f"[Config] ⚠️  CSV not found: {_CSV}")


class Config:
    # ── Dataset ──────────────────────────────────────────────
    DATA_MODE         = "merged"
    MERGED_CSV        = _CSV

    LABEL_COLUMN      = "Cognitive_Load"
    SUBJECT_COLUMN    = "subject_id"

    CLASS_NAMES       = ["Low (0-back)", "Medium (2-back)", "High (3-back)"]
    NUM_CLASSES       = 3

    EEG_COLUMNS       = _eeg_cols if _has_eeg else _hbo_cols
    MODALITY2_COLUMNS = _hbo_cols if _has_eeg else _hbr_cols

    EEG_FEATURE_PREFIX   = "eeg_" if _has_eeg else "hbo_"
    FNIRS_FEATURE_PREFIX = "hbo_" if _has_eeg else "hbr_"

    IGNORE_COLUMNS    = ["subject_id", "task"]

    # ── Preprocessing ─────────────────────────────────────────
    TEST_SIZE         = 0.20
    VAL_SIZE          = 0.20      # ← increased from 0.10 to 0.20
                                  #   gives 53→105 val samples (more reliable)
    RANDOM_SEED       = 42
    NORMALIZE         = True

    CHUNK_LOAD        = False
    CHUNK_SIZE        = 10_000

    # ── Model ─────────────────────────────────────────────────
    # CRITICAL: 654 samples needs a SMALL model to avoid overfitting
    # Old: 17M params → massively overfit (train=87%, test=29%)
    # New: ~200K params → right size for this dataset
    MODEL_TYPE        = "hybrid_cnn_lstm"

    CNN_FILTERS       = [64, 128]     # ← was [256, 512]
    CNN_KERNEL_SIZE   = 3
    CNN_DROPOUT       = 0.5           # ← was 0.3, stronger dropout

    LSTM_UNITS        = [64, 32]      # ← was [256, 128], much smaller
    LSTM_DROPOUT      = 0.4

    FUSION_TYPE       = "attention"
    FUSION_DIM        = 64            # ← was 256

    FC_UNITS          = [64, 32]      # ← was [256, 128, 64]
    FC_DROPOUT        = 0.5           # ← stronger dropout

    # ── Training ──────────────────────────────────────────────
    EPOCHS            = 200
    BATCH_SIZE        = 16            # ← was 32, smaller = more updates
    LEARNING_RATE     = 5e-4          # ← was 1e-3, slower learning
    WEIGHT_DECAY      = 1e-3          # ← was 1e-4, stronger L2 reg
    LR_PATIENCE       = 15
    EARLY_STOP        = 30
    USE_CLASS_WEIGHTS = True

    # ── Cross-validation ──────────────────────────────────────
    RUN_CV            = False
    CV_FOLDS          = 5

    # ── Output ────────────────────────────────────────────────
    OUTPUT_DIR        = "outputs/"
    SAVE_MODEL        = True
    MODEL_PATH        = "outputs/best_model.pt"
