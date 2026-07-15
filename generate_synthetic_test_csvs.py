"""
Generate SYNTHETIC test fixtures for exercising the /api/predict/upload feature.

These are NOT real physiological data and must never be cited as evidence of
model performance. They exist only to drive the upload / validation / prediction
pipeline manually. Each row is a resample-and-perturb of a REAL raw feature row
(so column scale/shape stays physiologically plausible), never independent noise.

IMPORTANT: rows are drawn from the RAW feature source
(data/tuberlin_eeg_fnirs.csv), NOT serving/demo_samples.json — the upload
endpoint expects raw extracted features and applies the full preprocessing
pipeline itself. Feeding it already-scaled demo features would double-scale.

Output: test_data/synthetic/*.csv  (+ a README.md test plan)

Usage:  python generate_synthetic_test_csvs.py
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(ROOT, "data", "tuberlin_eeg_fnirs.csv")
SCHEMA = os.path.join(ROOT, "serving", "feature_schema.json")
OUT_DIR = os.path.join(ROOT, "test_data", "synthetic")


def load_source():
    with open(SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    feature_cols = schema["eeg_columns"] + schema["fnirs_columns"]  # 180 + 468
    df = pd.read_csv(RAW_CSV)
    real = df[feature_cols].astype(np.float64)
    stds = real.std(axis=0)
    return real, feature_cols, stds


def synth(real, feature_cols, stds, n_rows, noise_scale=0.15, seed=0):
    """Resample n_rows real rows and add per-column Gaussian noise scaled to the
    real column std — keeps realistic scale/shape while making 'new' rows."""
    rng = np.random.default_rng(seed)
    base = real.sample(n=n_rows, replace=True, random_state=seed).to_numpy()
    noise = rng.normal(0.0, noise_scale, size=base.shape) * stds.to_numpy()
    return pd.DataFrame(base + noise, columns=feature_cols)


def write(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  {name:<44} {df.shape[0]:>3} rows x {df.shape[1]:>4} cols")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    real, cols, stds = load_source()
    print(f"Source: {len(real)} real raw rows, {len(cols)} feature columns\n"
          f"Writing synthetic fixtures to {OUT_DIR}:")

    # 1 — 10 rows, no subject_id (global fallback path)
    write(synth(real, cols, stds, 10, 0.15, seed=1), "synthetic_01_small.csv")

    # 2 — 10 rows, single consistent fake subject_id (per-subject path)
    d2 = synth(real, cols, stds, 10, 0.15, seed=2)
    d2.insert(0, "subject_id", "SYN_A")
    write(d2, "synthetic_02_small_with_subject.csv")

    # 3 — 50 rows, no subject_id
    write(synth(real, cols, stds, 50, 0.15, seed=3), "synthetic_03_medium.csv")

    # 4 — 50 rows, 3 mixed fake subject_ids (per-subject grouping, multi-subject)
    d4 = synth(real, cols, stds, 50, 0.15, seed=4)
    rng = np.random.default_rng(4)
    d4.insert(0, "subject_id", rng.choice(["SYN_A", "SYN_B", "SYN_C"], size=len(d4)))
    write(d4, "synthetic_04_medium_multi_subject.csv")

    # 5 — 20 rows, very low noise (close to real -> varied/confident predictions)
    write(synth(real, cols, stds, 20, 0.05, seed=5), "synthetic_05_low_noise.csv")

    # 6 — 20 rows, high noise (further from real -> less confident, still valid)
    write(synth(real, cols, stds, 20, 0.40, seed=6), "synthetic_06_high_noise.csv")

    # 7 — 200 rows, mixed noise levels, no subject_id (larger stress file)
    parts = [
        synth(real, cols, stds, 70, 0.05, seed=71),
        synth(real, cols, stds, 70, 0.15, seed=72),
        synth(real, cols, stds, 60, 0.30, seed=73),
    ]
    write(pd.concat(parts, ignore_index=True), "synthetic_07_large.csv")

    # 8 — exactly 1 row (smallest valid upload edge case)
    write(synth(real, cols, stds, 1, 0.15, seed=8), "synthetic_08_single_row.csv")

    # 9 — INVALID: drop ~20 required feature columns (expect rejection)
    d9 = synth(real, cols, stds, 15, 0.15, seed=9)
    dropped = cols[:10] + cols[180:190]  # 10 EEG + 10 HbO columns
    d9 = d9.drop(columns=dropped)
    write(d9, "synthetic_09_missing_columns.csv")

    # 10 — INVALID: rename a few required columns (case/typo near-misses) + add
    #      unexpected extra columns (expect rejection: the originals are missing)
    d10 = synth(real, cols, stds, 15, 0.15, seed=10)
    renames = {
        cols[0]: cols[0].upper(),          # case change  eeg_ch00_* -> EEG_CH00_*
        cols[1]: cols[1] + "_x",           # typo/suffix
        cols[180]: cols[180].replace("hbo", "hb0"),  # o -> zero near-miss
    }
    d10 = d10.rename(columns=renames)
    d10["unexpected_extra_1"] = 0.0
    d10["notes"] = "synthetic"
    write(d10, "synthetic_10_extra_and_renamed_columns.csv")

    write_readme()
    print("\nDone. These are SYNTHETIC fixtures — not real data, not performance evidence.")


def write_readme():
    lines = [
        "# Synthetic Test Fixtures — Upload Feature",
        "",
        "**These are SYNTHETIC fixtures, not real physiological data and not sourced",
        "from any human subject.** They exist only to manually exercise the",
        "`/api/predict/upload` validation + prediction pipeline. **Never** cite any",
        "prediction on these files as evidence of model performance in `RESULTS.md`",
        "or the report.",
        "",
        "## How they were made",
        "",
        "Generated by `generate_synthetic_test_csvs.py`. Each row is a **resample of a",
        "real raw feature row** (from `data/tuberlin_eeg_fnirs.csv`) plus per-column",
        "Gaussian noise scaled to that column's real standard deviation — so column",
        "scale/shape stays realistic, but no row is identical to a real sample. Files",
        "carry only the 180 EEG + 468 HbO feature columns the model requires (raw,",
        "un-preprocessed) and are **unlabelled** (no `Cognitive_Load` column).",
        "",
        "## Files",
        "",
        "| # | File | Rows | subject_id | Expected | Tests |",
        "|---|------|-----:|------------|----------|-------|",
        "| 1 | `synthetic_01_small.csv` | 10 | none | ✅ pass | global-normalization fallback (no subject_id) |",
        "| 2 | `synthetic_02_small_with_subject.csv` | 10 | one fake ID | ✅ pass | per-subject normalization path |",
        "| 3 | `synthetic_03_medium.csv` | 50 | none | ✅ pass | medium batch, fallback |",
        "| 4 | `synthetic_04_medium_multi_subject.csv` | 50 | 3 fake IDs | ✅ pass | per-subject grouping with multiple subjects in one file |",
        "| 5 | `synthetic_05_low_noise.csv` | 20 | none | ✅ pass | low noise (0.05) — close to real, varied predictions |",
        "| 6 | `synthetic_06_high_noise.csv` | 20 | none | ✅ pass | high noise (0.40) — unusual inputs, less confident (fine) |",
        "| 7 | `synthetic_07_large.csv` | 200 | none | ✅ pass | larger file, mixed noise — pipeline stress |",
        "| 8 | `synthetic_08_single_row.csv` | 1 | none | ✅ pass | smallest valid upload edge case |",
        "| 9 | `synthetic_09_missing_columns.csv` | 15 | none | ❌ reject | ~20 required columns dropped → specific 'missing columns' error |",
        "| 10 | `synthetic_10_extra_and_renamed_columns.csv` | 15 | none | ❌ reject | near-miss renamed columns (case/typo) + extra columns → caught, not silently processed |",
        "",
        "Files 1–8 should return per-row predictions and a class distribution.",
        "Files 9–10 should be rejected with HTTP 422 and a specific message naming",
        "the missing/renamed columns — never silently processed.",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python generate_synthetic_test_csvs.py",
        "```",
        "",
    ]
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  {'README.md':<44} (test plan)")


if __name__ == "__main__":
    main()
