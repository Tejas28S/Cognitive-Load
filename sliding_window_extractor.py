"""
Create a windowed TU Berlin EEG+fNIRS feature CSV from raw per-subject .mat files.

The extractor reuses the signal loading, filtering, and feature functions in
preprocess_tuberlin.py, then slices each aligned trial into overlapping windows.
Each emitted row keeps subject_id + trial_id metadata so data_loader.py can keep
all windows from the same trial together during train/val/test splitting.

Example:
    python sliding_window_extractor.py \
        --nirs-dir data/NIRS_01-26_MATLAB \
        --eeg-dir data/EEG_01-26_MATLAB \
        --output data/tuberlin_sliding.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocess_tuberlin import (
    TASK,
    FNIRS_FS,
    EEG_FS,
    CLASS_NAMES,
    eeg_col_names,
    eeg_features,
    fnirs_col_names,
    fnirs_features,
    load_eeg,
    load_nirs,
)


def _window_starts(n_samples, fs, window_sec, stride_sec):
    window = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    if window <= 0:
        raise ValueError("window_sec must produce at least one sample")
    if stride <= 0:
        raise ValueError("stride_sec must produce at least one sample")
    if n_samples < window:
        return []
    return list(range(0, n_samples - window + 1, stride))


def _slice_window(trial, start, fs, window_sec):
    end = start + int(round(window_sec * fs))
    return trial[start:end]


def _default_stride(window_sec, overlap):
    if overlap < 0 or overlap >= 1:
        raise ValueError("--overlap must be in [0, 1)")
    return window_sec * (1.0 - overlap)


def run(
    nirs_dir,
    eeg_dir=None,
    output="data/tuberlin_sliding.csv",
    window_sec=10.0,
    stride_sec=None,
    overlap=0.5,
    include_hbr=True,
):
    explicit_stride = stride_sec is not None
    stride_sec = stride_sec if explicit_stride else _default_stride(window_sec, overlap)
    nirs_subjects = sorted(glob.glob(os.path.join(nirs_dir, "VP*-NIRS")))
    if not nirs_subjects:
        raise FileNotFoundError(
            f"No VP*-NIRS folders in '{nirs_dir}'. Expected paths like "
            f"{nirs_dir}/VP001-NIRS/cnt_nback.mat"
        )

    has_eeg = eeg_dir is not None and os.path.isdir(eeg_dir)

    print("=" * 72)
    print("  TU Berlin Shin 2018 - Sliding Window Extraction")
    print("=" * 72)
    print(f"  Task      : {TASK}")
    print(f"  Subjects  : {len(nirs_subjects)}")
    print(f"  Window    : {window_sec:g}s")
    print(f"  Stride    : {stride_sec:g}s")
    print("  Overlap   : derived from stride" if explicit_stride else f"  Overlap   : {overlap:.0%}")
    print(f"  EEG       : {'YES -> ' + eeg_dir if has_eeg else 'NO - fNIRS only'}")
    print(f"  Output    : {output}")
    print("=" * 72)

    all_rows = []
    hbo_names = hbr_names = eeg_names = None
    eeg_ok_count = 0

    for subject_index, nirs_path in enumerate(nirs_subjects, 1):
        vp_id = os.path.basename(nirs_path).replace("-NIRS", "")
        subject_id = int("".join(filter(str.isdigit, vp_id)))

        hbo_list, hbr_list, y_list, times_ms = load_nirs(nirs_path)
        if not hbo_list:
            print(f"  [{subject_index:2d}/{len(nirs_subjects)}] {vp_id} skipped")
            continue

        eeg_list = None
        if has_eeg:
            eeg_path = os.path.join(eeg_dir, vp_id + "-EEG")
            if os.path.isdir(eeg_path):
                eeg_trials, _ = load_eeg(eeg_path, times_ms, y_list)
                if eeg_trials:
                    n_common = min(len(hbo_list), len(eeg_trials))
                    hbo_list = hbo_list[:n_common]
                    hbr_list = hbr_list[:n_common]
                    y_list = y_list[:n_common]
                    times_ms = times_ms[:n_common]
                    eeg_list = eeg_trials[:n_common]
                    eeg_ok_count += 1

        if hbo_names is None:
            hbo_names = fnirs_col_names("hbo_", hbo_list[0].shape[1])
            hbr_names = fnirs_col_names("hbr_", hbr_list[0].shape[1])
        if eeg_list and eeg_names is None:
            eeg_names = eeg_col_names(eeg_list[0].shape[1])

        subject_rows = 0
        for trial_index, (hbo_trial, hbr_trial, label, start_ms) in enumerate(
            zip(hbo_list, hbr_list, y_list, times_ms)
        ):
            nirs_starts = _window_starts(hbo_trial.shape[0], FNIRS_FS, window_sec, stride_sec)
            if eeg_list:
                eeg_starts = _window_starts(
                    eeg_list[trial_index].shape[0],
                    EEG_FS,
                    window_sec,
                    stride_sec,
                )
                n_windows = min(len(nirs_starts), len(eeg_starts))
            else:
                eeg_starts = []
                n_windows = len(nirs_starts)

            for window_index in range(n_windows):
                row = {}
                if eeg_list:
                    eeg_window = _slice_window(
                        eeg_list[trial_index],
                        eeg_starts[window_index],
                        EEG_FS,
                        window_sec,
                    )
                    for col_name, value in zip(eeg_names, eeg_features(eeg_window, EEG_FS)):
                        row[col_name] = float(value)

                hbo_window = _slice_window(
                    hbo_trial,
                    nirs_starts[window_index],
                    FNIRS_FS,
                    window_sec,
                )
                for col_name, value in zip(hbo_names, fnirs_features(hbo_window, FNIRS_FS)):
                    row[col_name] = float(value)

                if include_hbr:
                    hbr_window = _slice_window(
                        hbr_trial,
                        nirs_starts[window_index],
                        FNIRS_FS,
                        window_sec,
                    )
                    for col_name, value in zip(hbr_names, fnirs_features(hbr_window, FNIRS_FS)):
                        row[col_name] = float(value)

                row["subject_id"] = subject_id
                row["trial_id"] = f"{subject_id:03d}_{trial_index:03d}"
                row["window_id"] = window_index
                row["window_start_sec"] = nirs_starts[window_index] / FNIRS_FS
                row["window_sec"] = float(window_sec)
                row["source_trial_start_ms"] = float(start_ms)
                row["task"] = TASK
                row["Cognitive_Load"] = int(label)
                all_rows.append(row)
                subject_rows += 1

        print(
            f"  [{subject_index:2d}/{len(nirs_subjects)}] {vp_id} -> "
            f"{len(hbo_list)} trials, {subject_rows} windows"
            f"{' + EEG' if eeg_list else ''}"
        )

    if not all_rows:
        raise RuntimeError("No windows were extracted. Check input paths and window settings.")

    df = pd.DataFrame(all_rows).fillna(0)
    eeg_cols = [c for c in df.columns if c.startswith("eeg_")]
    hbo_cols = [c for c in df.columns if c.startswith("hbo_")]
    hbr_cols = [c for c in df.columns if c.startswith("hbr_")]
    meta_cols = [
        "subject_id",
        "trial_id",
        "window_id",
        "window_start_sec",
        "window_sec",
        "source_trial_start_ms",
        "task",
        "Cognitive_Load",
    ]
    df = df[eeg_cols + hbo_cols + hbr_cols + meta_cols]

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    df.to_csv(output, index=False)

    print("=" * 72)
    print("  DONE")
    print(f"  Windows        : {len(df):,}")
    print(f"  Trials/groups  : {df['trial_id'].nunique():,}")
    print(f"  Subjects       : {df['subject_id'].nunique():,}")
    print(f"  EEG features   : {len(eeg_cols)} (subjects with EEG: {eeg_ok_count})")
    print(f"  HbO features   : {len(hbo_cols)}")
    print(f"  HbR features   : {len(hbr_cols)}")
    for label, name in CLASS_NAMES.items():
        n = int((df["Cognitive_Load"] == label).sum())
        print(f"  {name:25s}: {n:,} ({n / len(df):.1%})")
    print(f"  Saved -> {output}")
    print("=" * 72)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nirs-dir", required=True)
    parser.add_argument("--eeg-dir", default=None)
    parser.add_argument("--output", default="data/tuberlin_sliding.csv")
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--no-hbr", action="store_true", help="Only write HbO fNIRS features")
    args = parser.parse_args()

    run(
        nirs_dir=args.nirs_dir,
        eeg_dir=args.eeg_dir,
        output=args.output,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        overlap=args.overlap,
        include_hbr=not args.no_hbr,
    )
