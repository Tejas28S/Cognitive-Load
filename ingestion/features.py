"""
features.py: bridge from a projected canonical recording to the bundle's
feature table.

This module deliberately contains NO feature mathematics of its own. It imports
`bandpass`, `baseline_correct`, `eeg_features`, `fnirs_features`,
`eeg_col_names` and `fnirs_col_names` from backend/mat_extractor.py and calls
them exactly as the existing TU Berlin path does, so a value computed here is
the same value the current pipeline would compute. Those functions are already
channel-count agnostic (they loop over `trial.shape[1]`), which is why no change
to them is needed.

What this module adds is only the plumbing the existing path got for free by
being TU Berlin-specific: windowing arbitrary sample rates, and preserving
"this channel could not be filled" as NaN instead of letting it become 0.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# The existing, unmodified extraction maths.
from mat_extractor import (            # noqa: E402
    bandpass,
    baseline_correct,
    eeg_col_names,
    eeg_features,
    fnirs_col_names,
    fnirs_features,
)


def _all_nan_channels(arr):
    if arr is None:
        return np.zeros(0, dtype=bool)
    return np.all(np.isnan(arr), axis=0)


def _safe_bandpass(data, lo, hi, fs, order=3):
    """Bandpass that tolerates unfillable (all-NaN) channels.

    filtfilt propagates NaN across an entire column, and cannot run at all on a
    column that is wholly NaN. Missing channels are therefore filtered as zeros
    and restored to NaN afterwards, so a gap stays visibly a gap.
    """
    nan_mask = _all_nan_channels(data)
    clean = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0)
    out = bandpass(clean, lo, hi, fs, order=order)
    if nan_mask.any():
        out[:, nan_mask] = np.nan
    return out


def extract_feature_frame(
    eeg=None, eeg_fs=None, hbo=None, hbr=None, fnirs_fs=None,
    window_sec=5.0, stride_sec=1.0, subject_id="UPLOAD", trial_id=None,
    eeg_band=(0.5, 45.0), fnirs_band=(0.01, 0.1), max_windows=None,
):
    """Window a projected recording and extract the bundle's feature columns.

    Returns a DataFrame whose feature columns are exactly the canonical names
    (`eeg_ch00_alpha_mean`, `hbo_ch00_mean`, ...), plus the metadata columns the
    downstream preprocessing needs (`subject_id`, `window_id`,
    `window_start_sec`).

    Mirrors backend/mat_extractor.process_raw_mats: same filters, same 5s
    windows at 1s stride, same baseline correction, same column ordering.
    """
    if eeg is None and hbo is None:
        raise ValueError("extract_feature_frame needs at least one modality.")

    eeg_f = _safe_bandpass(eeg, eeg_band[0], eeg_band[1], eeg_fs) \
        if eeg is not None else None
    hbo_f = _safe_bandpass(hbo, fnirs_band[0], fnirs_band[1], fnirs_fs) \
        if hbo is not None else None
    hbr_f = _safe_bandpass(hbr, fnirs_band[0], fnirs_band[1], fnirs_fs) \
        if hbr is not None else None

    # Usable duration is bounded by the shorter modality.
    durations = []
    if eeg_f is not None:
        durations.append(eeg_f.shape[0] / float(eeg_fs))
    if hbo_f is not None:
        durations.append(hbo_f.shape[0] / float(fnirs_fs))
    duration = min(durations)

    n_windows = int((duration - window_sec) / stride_sec) + 1
    if n_windows <= 0:
        raise ValueError(
            f"Recording is {duration:.1f}s, too short for a single "
            f"{window_sec}s window."
        )
    # Bound the extraction itself, not its output. A full-length recording is
    # thousands of windows, and computing them all only to throw most away costs
    # minutes per request.
    if max_windows:
        n_windows = min(n_windows, int(max_windows))

    eeg_nan = _all_nan_channels(eeg_f) if eeg_f is not None else None
    hbo_nan = _all_nan_channels(hbo_f) if hbo_f is not None else None
    hbr_nan = _all_nan_channels(hbr_f) if hbr_f is not None else None

    eeg_names = eeg_col_names(eeg_f.shape[1]) if eeg_f is not None else []
    hbo_names = fnirs_col_names("hbo_", hbo_f.shape[1]) if hbo_f is not None else []
    hbr_names = fnirs_col_names("hbr_", hbr_f.shape[1]) if hbr_f is not None else []

    n_eeg_per_ch = 18   # 6 bands x (mean, std, slope)
    n_fn_per_ch = 13

    rows = []
    for i in range(n_windows):
        start_sec = i * stride_sec
        row = {}

        if eeg_f is not None:
            s = int(round(start_sec * eeg_fs))
            e = s + int(round(window_sec * eeg_fs))
            win = np.nan_to_num(eeg_f[s:e])
            vals = eeg_features(win, eeg_fs)
            if eeg_nan is not None and eeg_nan.any():
                vals = vals.copy()
                for ch in np.flatnonzero(eeg_nan):
                    vals[ch * n_eeg_per_ch:(ch + 1) * n_eeg_per_ch] = np.nan
            row.update(dict(zip(eeg_names, (float(v) for v in vals))))

        if hbo_f is not None:
            s = int(round(start_sec * fnirs_fs))
            e = s + int(round(window_sec * fnirs_fs))
            win = baseline_correct(np.nan_to_num(hbo_f[s:e]), fnirs_fs)
            vals = fnirs_features(win, fnirs_fs)
            if hbo_nan is not None and hbo_nan.any():
                vals = vals.copy()
                for ch in np.flatnonzero(hbo_nan):
                    vals[ch * n_fn_per_ch:(ch + 1) * n_fn_per_ch] = np.nan
            row.update(dict(zip(hbo_names, (float(v) for v in vals))))

        if hbr_f is not None:
            s = int(round(start_sec * fnirs_fs))
            e = s + int(round(window_sec * fnirs_fs))
            win = baseline_correct(np.nan_to_num(hbr_f[s:e]), fnirs_fs)
            vals = fnirs_features(win, fnirs_fs)
            if hbr_nan is not None and hbr_nan.any():
                vals = vals.copy()
                for ch in np.flatnonzero(hbr_nan):
                    vals[ch * n_fn_per_ch:(ch + 1) * n_fn_per_ch] = np.nan
            row.update(dict(zip(hbr_names, (float(v) for v in vals))))

        row["window_id"] = i
        row["window_start_sec"] = start_sec
        row["subject_id"] = subject_id
        row["trial_id"] = trial_id or f"{subject_id}_001"
        rows.append(row)

    return pd.DataFrame(rows)


def label_windows_from_events(frame, events, label_map, window_sec=5.0,
                              label_column="Cognitive_Load"):
    """Attach labels to windows from trial events, where a dataset has them.

    A window takes the label of the event whose onset falls inside it. Windows
    containing events of more than one class, or none, are left unlabelled
    (-1) rather than assigned a majority guess.
    """
    if not events:
        return frame

    starts = frame["window_start_sec"].to_numpy(dtype=float)
    labels = np.full(len(frame), -1, dtype=np.int64)

    for i, s in enumerate(starts):
        inside = [
            label_map.get(str(e.get("type", "")).strip().lower(), -1)
            for e in events
            if s <= float(e.get("latency_sec", -1)) < s + window_sec
        ]
        inside = [v for v in inside if v >= 0]
        if inside and len(set(inside)) == 1:
            labels[i] = inside[0]

    out = frame.copy()
    out[label_column] = labels
    return out
