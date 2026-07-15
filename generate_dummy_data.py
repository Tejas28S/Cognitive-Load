# ============================================================
#  utils/generate_dummy_data.py
#  Run this ONCE to create test CSVs so you can validate the
#  pipeline before plugging in your real dataset.
# ============================================================

import os
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

N_SAMPLES      = 600        # total samples
N_EEG_FEATS    = 40         # e.g. power bands × channels
N_FNIRS_FEATS  = 24         # e.g. HbO/HbR × channels
CLASS_NAMES    = [0, 1, 2]  # 0=low, 1=medium, 2=high

os.makedirs("data", exist_ok=True)

labels = np.random.choice(CLASS_NAMES, size=N_SAMPLES,
                           p=[0.35, 0.40, 0.25])   # slight imbalance

# EEG features: add class-correlated signal
eeg_data = np.random.randn(N_SAMPLES, N_EEG_FEATS)
for i, lbl in enumerate(labels):
    eeg_data[i, :10] += lbl * 0.6    # first 10 features carry class info

# fNIRS features
fnirs_data = np.random.randn(N_SAMPLES, N_FNIRS_FEATS)
for i, lbl in enumerate(labels):
    fnirs_data[i, :8] += lbl * 0.5

# ── Write MERGED CSV ─────────────────────────────────────────
eeg_cols   = [f"eeg_{i:02d}"   for i in range(N_EEG_FEATS)]
fnirs_cols = [f"fnirs_{i:02d}" for i in range(N_FNIRS_FEATS)]

df_merged = pd.DataFrame(
    np.hstack([eeg_data, fnirs_data]),
    columns=eeg_cols + fnirs_cols
)
df_merged["cognitive_load"] = labels
df_merged.to_csv("data/merged_features.csv", index=False)
print(f"✅ Wrote data/merged_features.csv  ({df_merged.shape})")

# ── Write SEPARATE CSVs ──────────────────────────────────────
df_eeg        = pd.DataFrame(eeg_data, columns=eeg_cols)
df_eeg["cognitive_load"] = labels
df_eeg.to_csv("data/eeg_features.csv", index=False)

df_fnirs      = pd.DataFrame(fnirs_data, columns=fnirs_cols)
df_fnirs["cognitive_load"] = labels
df_fnirs.to_csv("data/fnirs_features.csv", index=False)

print(f"✅ Wrote data/eeg_features.csv    ({df_eeg.shape})")
print(f"✅ Wrote data/fnirs_features.csv  ({df_fnirs.shape})")
print("\nClass distribution:")
for lbl, name in zip([0,1,2], ["Low","Medium","High"]):
    print(f"  {name:8s}: {(labels==lbl).sum():4d}  ({(labels==lbl).mean():.1%})")
