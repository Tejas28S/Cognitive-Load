"""
Create v2 TU Berlin EEG+fNIRS sliding-window features.

This keeps fNIRS extraction identical to sliding_window_extractor.py, but changes
EEG to a real temporal sequence:
    (outer_window, eeg_subwindow, channel_band_feature)

EEG features are baseline-normalized per subject using that subject's 0-back
windows, then written as columns named eeg_t00_ch00_delta, ...

Example:
    python sliding_window_extractor_v2.py \
        --nirs-dir data/NIRS_01-26_MATLAB \
        --eeg-dir data/EEG_01-26_MATLAB \
        --output data/tuberlin_sliding_v2.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocess_tuberlin import (
    TASK,
    FNIRS_FS,
    EEG_FS,
    CLASS_NAMES,
    _eeg_band_features,
    fnirs_col_names,
    fnirs_features,
    load_eeg,
    load_nirs,
)


EEG_BANDS = ["delta", "theta", "alpha", "beta", "gamma", "engage"]


def _window_starts(n_samples, fs, window_sec, stride_sec):
    window = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    if window <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must produce positive samples")
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


def _subwindow_starts(n_samples, fs, sub_window_sec, sub_overlap):
    sub_len = int(round(sub_window_sec * fs))
    sub_stride = int(round(sub_len * (1.0 - sub_overlap)))
    if sub_len <= 0 or sub_stride <= 0:
        raise ValueError("EEG sub-window settings must produce positive samples")
    if n_samples < sub_len:
        return [0]
    return list(range(0, n_samples - sub_len + 1, sub_stride))


def eeg_sequence_col_names(n_channels, n_steps):
    base = [
        f"ch{ch:02d}_{band}"
        for ch in range(n_channels)
        for band in EEG_BANDS
    ]
    return [
        f"eeg_t{step:02d}_{feature}"
        for step in range(n_steps)
        for feature in base
    ]


def eeg_sequence_features(
    eeg_window,
    fs,
    sub_window_sec=2.0,
    sub_overlap=0.0,
    artifact_threshold_uv=None,
):
    starts = _subwindow_starts(eeg_window.shape[0], fs, sub_window_sec, sub_overlap)
    sub_len = int(round(sub_window_sec * fs))
    seq = []

    for start in starts:
        segment = eeg_window[start:start + sub_len]
        if segment.shape[0] < max(8, int(round(0.5 * fs))):
            pad = np.full((eeg_window.shape[1], len(EEG_BANDS)), np.nan, dtype=np.float32)
            seq.append(pad)
            continue

        if artifact_threshold_uv is not None:
            if np.nanmax(np.abs(segment)) > artifact_threshold_uv:
                pad = np.full((segment.shape[1], len(EEG_BANDS)), np.nan, dtype=np.float32)
                seq.append(pad)
                continue

        seq.append(_eeg_band_features(segment, fs))

    return np.stack(seq, axis=0).reshape(len(seq), -1).astype(np.float32)


def _baseline_normalize_subject(sequences, labels, mode="robust", clip=20.0):
    arr = np.stack(sequences, axis=0).astype(np.float32)
    baseline = arr[np.asarray(labels) == 0]
    if baseline.size == 0:
        baseline = arr

    flat_baseline = baseline.reshape(-1, arr.shape[-1])
    mode = mode.lower()
    if mode == "zscore":
        center = np.nanmean(flat_baseline, axis=0)
        scale = np.nanstd(flat_baseline, axis=0)
    elif mode == "robust":
        center = np.nanmedian(flat_baseline, axis=0)
        q75 = np.nanpercentile(flat_baseline, 75, axis=0)
        q25 = np.nanpercentile(flat_baseline, 25, axis=0)
        scale = (q75 - q25) / 1.349
    else:
        raise ValueError("EEG baseline mode must be 'zscore' or 'robust'")

    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    norm = (arr - center.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    if clip is not None and clip > 0:
        norm = np.clip(norm, -float(clip), float(clip))
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)


def run(
    nirs_dir,
    eeg_dir,
    output="data/tuberlin_sliding_v2.csv",
    window_sec=10.0,
    stride_sec=None,
    overlap=0.5,
    eeg_sub_window_sec=2.0,
    eeg_sub_overlap=0.0,
    artifact_threshold_uv=None,
    include_hbr=True,
    baseline_normalize=True,
    baseline_mode="robust",
    baseline_clip=20.0,
):
    explicit_stride = stride_sec is not None
    stride_sec = stride_sec if explicit_stride else _default_stride(window_sec, overlap)
    nirs_subjects = sorted(glob.glob(os.path.join(nirs_dir, "VP*-NIRS")))
    if not nirs_subjects:
        raise FileNotFoundError(f"No VP*-NIRS folders in '{nirs_dir}'")
    if not eeg_dir or not os.path.isdir(eeg_dir):
        raise FileNotFoundError("v2 extraction requires --eeg-dir")

    print("=" * 72)
    print("  TU Berlin Shin 2018 - Sliding Window Extraction v2")
    print("=" * 72)
    print(f"  Task          : {TASK}")
    print(f"  Subjects      : {len(nirs_subjects)}")
    print(f"  Outer window  : {window_sec:g}s")
    print(f"  Outer stride  : {stride_sec:g}s")
    print(f"  EEG subwindow : {eeg_sub_window_sec:g}s overlap={eeg_sub_overlap:.0%}")
    if baseline_normalize:
        clip_label = "none" if baseline_clip is None or baseline_clip <= 0 else f"+/-{baseline_clip:g}"
        baseline_label = f"per-subject 0-back {baseline_mode} (clip={clip_label})"
    else:
        baseline_label = "disabled"
    print(f"  EEG baseline  : {baseline_label}")
    print(f"  fNIRS         : unchanged")
    print(f"  Output        : {output}")
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

        eeg_path = os.path.join(eeg_dir, vp_id + "-EEG")
        if not os.path.isdir(eeg_path):
            print(f"  [{subject_index:2d}/{len(nirs_subjects)}] {vp_id} skipped (no EEG)")
            continue

        eeg_trials, _ = load_eeg(eeg_path, times_ms, y_list)
        if not eeg_trials:
            print(f"  [{subject_index:2d}/{len(nirs_subjects)}] {vp_id} skipped (EEG load failed)")
            continue

        n_common = min(len(hbo_list), len(eeg_trials))
        hbo_list = hbo_list[:n_common]
        hbr_list = hbr_list[:n_common]
        y_list = y_list[:n_common]
        times_ms = times_ms[:n_common]
        eeg_trials = eeg_trials[:n_common]
        eeg_ok_count += 1

        if hbo_names is None:
            hbo_names = fnirs_col_names("hbo_", hbo_list[0].shape[1])
            hbr_names = fnirs_col_names("hbr_", hbr_list[0].shape[1])

        subject_rows = []
        subject_sequences = []
        subject_labels = []
        subject_windows = 0

        for trial_index, (hbo_trial, hbr_trial, label, start_ms) in enumerate(
            zip(hbo_list, hbr_list, y_list, times_ms)
        ):
            nirs_starts = _window_starts(hbo_trial.shape[0], FNIRS_FS, window_sec, stride_sec)
            eeg_starts = _window_starts(eeg_trials[trial_index].shape[0], EEG_FS, window_sec, stride_sec)
            n_windows = min(len(nirs_starts), len(eeg_starts))

            for window_index in range(n_windows):
                eeg_window = _slice_window(
                    eeg_trials[trial_index],
                    eeg_starts[window_index],
                    EEG_FS,
                    window_sec,
                )
                eeg_seq = eeg_sequence_features(
                    eeg_window,
                    EEG_FS,
                    sub_window_sec=eeg_sub_window_sec,
                    sub_overlap=eeg_sub_overlap,
                    artifact_threshold_uv=artifact_threshold_uv,
                )
                if eeg_names is None:
                    eeg_names = eeg_sequence_col_names(eeg_window.shape[1], eeg_seq.shape[0])

                row = {}
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

                subject_rows.append(row)
                subject_sequences.append(eeg_seq)
                subject_labels.append(int(label))
                subject_windows += 1

        if subject_sequences:
            if baseline_normalize:
                output_sequences = _baseline_normalize_subject(
                    subject_sequences,
                    subject_labels,
                    mode=baseline_mode,
                    clip=baseline_clip,
                )
            else:
                output_sequences = np.nan_to_num(
                    np.stack(subject_sequences, axis=0).astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            for row, seq in zip(subject_rows, output_sequences):
                for col_name, value in zip(eeg_names, seq.reshape(-1)):
                    row[col_name] = float(value)

        all_rows.extend(subject_rows)
        print(
            f"  [{subject_index:2d}/{len(nirs_subjects)}] {vp_id} -> "
            f"{len(hbo_list)} trials, {subject_windows} windows + EEG sequence"
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
    parser.add_argument("--eeg-dir", required=True)
    parser.add_argument("--output", default="data/tuberlin_sliding_v2.csv")
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--eeg-sub-window-sec", type=float, default=2.0)
    parser.add_argument("--eeg-sub-overlap", type=float, default=0.0)
    parser.add_argument("--artifact-threshold-uv", type=float, default=None)
    parser.add_argument("--no-hbr", action="store_true", help="Only write HbO fNIRS features")
    parser.add_argument(
        "--no-eeg-baseline",
        action="store_true",
        help="Disable per-subject 0-back EEG baseline normalization",
    )
    parser.add_argument(
        "--eeg-baseline-mode",
        choices=["robust", "zscore"],
        default="robust",
        help="Per-subject EEG baseline normalization mode",
    )
    parser.add_argument(
        "--eeg-baseline-clip",
        type=float,
        default=20.0,
        help="Clip baseline-normalized EEG features to +/- this value; use 0 to disable",
    )
    args = parser.parse_args()

    run(
        nirs_dir=args.nirs_dir,
        eeg_dir=args.eeg_dir,
        output=args.output,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        overlap=args.overlap,
        eeg_sub_window_sec=args.eeg_sub_window_sec,
        eeg_sub_overlap=args.eeg_sub_overlap,
        artifact_threshold_uv=args.artifact_threshold_uv,
        include_hbr=not args.no_hbr,
        baseline_normalize=not args.no_eeg_baseline,
        baseline_mode=args.eeg_baseline_mode,
        baseline_clip=args.eeg_baseline_clip,
    )
