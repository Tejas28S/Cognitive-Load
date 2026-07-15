# ============================================================
#  preprocess_tuberlin.py  â€“  TU Berlin Shin 2018 dataset
#  EEG + fNIRS (HbO + HbR) preprocessing + feature extraction
#
#  Key design: EEG epochs use NIRS marker timestamps so both
#  modalities are always aligned and have matching trial counts.
#
#  Usage:
#    python preprocess_tuberlin.py \
#        --nirs-dir data/NIRS_01-26_MATLAB \
#        --eeg-dir  data/EEG_01-26_MATLAB \
#        --output   data/tuberlin_eeg_fnirs.csv
# ============================================================

import os, sys, glob, argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

TASK         = "nback"
FNIRS_FS     = 10
EEG_FS       = 200
TRIAL_SEC    = 60
BASELINE_SEC = 5
CLASS_NAMES  = {0: "Low (0-back)", 1: "Medium (2-back)", 2: "High (3-back)"}


# â”€â”€ Smart mat loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_mat_smart(path):
    try:
        return sio.loadmat(path), "scipy"
    except Exception as e:
        if HAS_H5PY and ("HDF" in str(e) or "bytes" in str(e).lower()):
            return h5py.File(path, 'r'), "h5py"
        raise


# â”€â”€ Signal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def bandpass(sig, lo, hi, fs, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

def baseline_correct(trial, fs, sec=5):
    n = int(sec * fs)
    return trial - trial[:n].mean(axis=0, keepdims=True)


# â”€â”€ Feature extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fnirs_features(trial, fs):
    """13 stats per channel. trial: (T, ch) â†’ flat array"""
    feats, n = [], trial.shape[0]
    t = np.arange(n) / fs
    for ch in range(trial.shape[1]):
        sig, mid = trial[:, ch], n // 2
        slope = np.polyfit(t, sig, 1)[0]
        feats += [
            np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
            np.max(sig) - np.min(sig), skew(sig), kurtosis(sig),
            slope, np.mean(sig[:mid]), np.mean(sig[mid:]),
            np.mean(sig[mid:]) - np.mean(sig[:mid]),
            np.sum(sig**2) / n,
            np.trapezoid(np.abs(sig), t),
        ]
    return np.array(feats, dtype=np.float32)

def _eeg_band_features(segment, fs):
    bands = [(0.5,4),(4,8),(8,13),(13,30),(30,45)]
    f, psd = welch(
        segment,
        fs=fs,
        axis=0,
        nperseg=min(int(fs*2), max(8, segment.shape[0]//2)),
    )
    band_power = [
        psd[(f>=lo)&(f<hi)].mean(axis=0) if np.any((f>=lo)&(f<hi)) else np.zeros(segment.shape[1])
        for lo, hi in bands
    ]
    band_power = np.vstack(band_power).T
    engagement = band_power[:, 3:4] / (band_power[:, 1:2] + band_power[:, 2:3] + 1e-12)
    return np.hstack([band_power, engagement]).astype(np.float32)


def eeg_features(trial, fs):
    """2s subwindow EEG dynamics: mean/std/slope of 6 band features per channel."""
    feats = []
    sub_len = int(round(2.0 * fs))
    starts = list(range(0, max(trial.shape[0] - sub_len + 1, 1), sub_len))
    sub_feats = []
    for start in starts:
        segment = trial[start:start + sub_len]
        if segment.shape[0] < max(8, int(0.5 * fs)):
            continue
        sub_feats.append(_eeg_band_features(segment, fs))
    if not sub_feats:
        sub_feats = [_eeg_band_features(trial, fs)]

    sub_feats = np.stack(sub_feats, axis=0)
    means = sub_feats.mean(axis=0)
    stds = sub_feats.std(axis=0)
    if sub_feats.shape[0] > 1:
        x = np.arange(sub_feats.shape[0], dtype=np.float32)
        x = x - x.mean()
        denom = np.sum(x ** 2) + 1e-12
        slopes = np.tensordot(x, sub_feats - sub_feats.mean(axis=0), axes=(0, 0)) / denom
    else:
        slopes = np.zeros_like(means)

    per_channel = np.concatenate([means, stds, slopes], axis=1)
    return per_channel.reshape(-1).astype(np.float32)

def fnirs_col_names(prefix, n_ch):
    stats = ['mean','std','max','min','ptp','skew','kurt',
             'slope','early','late','delta','power','auc']
    return [f"{prefix}ch{ch:02d}_{s}" for ch in range(n_ch) for s in stats]

def eeg_col_names(n_ch):
    bands = ['delta','theta','alpha','beta','gamma','engage']
    stats = ["mean", "std", "slope"]
    return [f"eeg_ch{ch:02d}_{b}_{s}" for ch in range(n_ch) for s in stats for b in bands]


# â”€â”€ NIRS loader â€” returns trials + marker times â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_nirs(subj_dir):
    """Returns hbo_list, hbr_list, y_list, times_ms"""
    cnt_path = os.path.join(subj_dir, f"cnt_{TASK}.mat")
    mrk_path = os.path.join(subj_dir, f"mrk_{TASK}.mat")
    if not os.path.exists(cnt_path):
        return None, None, None, None

    mat, _  = load_mat_smart(cnt_path)
    cnt     = mat[f"cnt_{TASK}"]
    oxy_sig   = cnt['oxy'][0,0]['x'][0,0].astype(np.float64)
    deoxy_sig = cnt['deoxy'][0,0]['x'][0,0].astype(np.float64)
    fs        = int(cnt['oxy'][0,0]['fs'][0,0].flat[0])

    mrk_mat, _ = load_mat_smart(mrk_path)
    mrk        = mrk_mat[f"mrk_{TASK}"][0,0]
    # mrk['time'] shape is (1, N) â†’ flatten() gives (N,) directly
    # mrk['y']   shape is (3, N) â†’ argmax(axis=0) gives (N,)
    times_ms   = mrk['time'].flatten()          # (N,) trial start times in ms
    labels     = mrk['y'].argmax(axis=0).flatten()  # (N,) class labels 0/1/2
    win        = TRIAL_SEC * fs

    oxy_f   = bandpass(oxy_sig,   0.01, 0.1, fs)
    deoxy_f = bandpass(deoxy_sig, 0.01, 0.1, fs)

    hbo_list, hbr_list, y_list, t_list = [], [], [], []
    for t_ms, lbl in zip(times_ms, labels):
        s = int((t_ms / 1000) * fs)
        e = s + win
        if e > oxy_sig.shape[0]:
            continue
        hbo_list.append(np.nan_to_num(baseline_correct(oxy_f[s:e],   fs)))
        hbr_list.append(np.nan_to_num(baseline_correct(deoxy_f[s:e], fs)))
        y_list.append(int(lbl))
        t_list.append(t_ms)   # keep track of extracted trial times

    return hbo_list, hbr_list, y_list, np.array(t_list)


# â”€â”€ EEG loader â€” uses NIRS marker times for alignment â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_eeg(subj_dir, nirs_times_ms, nirs_labels):
    """
    Extract EEG epochs using the same timestamps as NIRS.
    This guarantees trial-level alignment between modalities.
    """
    cnt_path = os.path.join(subj_dir, f"cnt_{TASK}.mat")
    if not os.path.exists(cnt_path):
        return None, None

    try:
        mat, fmt = load_mat_smart(cnt_path)

        if fmt == "scipy":
            cnt = mat[f"cnt_{TASK}"]
            if hasattr(cnt, 'dtype') and cnt.dtype.names:
                cnt = cnt[0, 0]
            if not hasattr(cnt, 'dtype') or 'x' not in cnt.dtype.names:
                return None, None
            x_raw  = cnt['x']
            sig    = (x_raw[0,0] if x_raw.shape==(1,1) else x_raw).astype(np.float64)
            fs_raw = cnt['fs']
            fs     = int((fs_raw[0,0] if fs_raw.shape==(1,1) else fs_raw).flat[0])
        else:
            cnt = mat[f"cnt_{TASK}"]
            sig = np.array(cnt['x']).T.astype(np.float64)
            fs  = int(np.array(cnt['fs']).flat[0])

        win    = TRIAL_SEC * fs
        sig_bp = bandpass(sig, 0.5, 45.0, fs)

        trials, y_list = [], []
        for t_ms, lbl in zip(nirs_times_ms, nirs_labels):
            s = int((t_ms / 1000) * fs)
            e = s + win
            if e > sig.shape[0]:
                continue     # skip trial that clips end of EEG recording
            trials.append(np.nan_to_num(sig_bp[s:e]))
            y_list.append(int(lbl))

        return trials, y_list

    except Exception as e:
        print(f"      EEG error: {e}")
        return None, None


# â”€â”€ Main pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run(nirs_dir, eeg_dir=None, output="data/tuberlin_eeg_fnirs.csv"):
    nirs_subjects = sorted(glob.glob(os.path.join(nirs_dir, "VP*-NIRS")))
    if not nirs_subjects:
        raise FileNotFoundError(
            f"No VP*-NIRS folders in '{nirs_dir}'\n"
            f"Expected: {nirs_dir}/VP001-NIRS/cnt_nback.mat ..."
        )

    has_eeg = eeg_dir is not None and os.path.isdir(eeg_dir)

    print("=" * 62)
    print("  TU Berlin Shin 2018 â€” Preprocessing")
    print("=" * 62)
    print(f"  Task     : {TASK}  (0/2/3-back â†’ Low/Med/High CL)")
    print(f"  Subjects : {len(nirs_subjects)}")
    print(f"  Window   : {TRIAL_SEC}s task + {BASELINE_SEC}s baseline")
    print(f"  EEG      : {'YES â†’ ' + eeg_dir if has_eeg else 'NO â€” fNIRS only'}")
    print("=" * 62)

    all_rows     = []
    hbo_names    = hbr_names = eeg_names = None
    eeg_ok_count = 0

    for idx, nirs_path in enumerate(nirs_subjects, 1):
        vp_id   = os.path.basename(nirs_path).replace("-NIRS", "")
        subj_id = int(''.join(filter(str.isdigit, vp_id)))

        hbo_list, hbr_list, y_list, times_ms = load_nirs(nirs_path)
        if not hbo_list:
            print(f"  [{idx:2d}/{len(nirs_subjects)}] {vp_id}  âš ï¸  Skipped")
            continue

        print(f"  [{idx:2d}/{len(nirs_subjects)}] {vp_id}  â†’ {len(hbo_list)} trials", end="")

        # Load EEG using NIRS marker times for alignment
        eeg_list = None
        if has_eeg:
            eeg_path = os.path.join(eeg_dir, vp_id + "-EEG")
            if os.path.isdir(eeg_path):
                eeg_trials, eeg_y = load_eeg(eeg_path, times_ms, y_list)
                if eeg_trials and len(eeg_trials) > 0:
                    # Trim both modalities to matching trial count
                    n_common = min(len(hbo_list), len(eeg_trials))
                    hbo_list  = hbo_list[:n_common]
                    hbr_list  = hbr_list[:n_common]
                    y_list    = y_list[:n_common]
                    eeg_list  = eeg_trials[:n_common]
                    eeg_ok_count += 1
                    print(f" + EEG âœ… ({n_common} paired)", end="")
                else:
                    print(f" + EEG âŒ", end="")

        # Build column names once from first subject
        if hbo_names is None:
            hbo_names = fnirs_col_names("hbo_", hbo_list[0].shape[1])
            hbr_names = fnirs_col_names("hbr_", hbr_list[0].shape[1])
            print(f"\n     HbO: {hbo_list[0].shape[1]} ch Ã— 13 = {len(hbo_names)} feats")
            print(f"     HbR: {hbr_list[0].shape[1]} ch Ã— 13 = {len(hbr_names)} feats", end="")
            if eeg_list:
                eeg_names = eeg_col_names(eeg_list[0].shape[1])
                print(f"\n     EEG: {eeg_list[0].shape[1]} ch Ã— 6  = {len(eeg_names)} feats", end="")
        print()

        for i in range(len(hbo_list)):
            row = {}
            if eeg_list and i < len(eeg_list):
                if eeg_names is None:
                    eeg_names = eeg_col_names(eeg_list[i].shape[1])
                for cn, v in zip(eeg_names, eeg_features(eeg_list[i], EEG_FS)):
                    row[cn] = float(v)
            for cn, v in zip(hbo_names, fnirs_features(hbo_list[i], FNIRS_FS)):
                row[cn] = float(v)
            for cn, v in zip(hbr_names, fnirs_features(hbr_list[i], FNIRS_FS)):
                row[cn] = float(v)
            row["subject_id"]     = subj_id
            row["task"]           = TASK
            row["Cognitive_Load"] = y_list[i]
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    eeg_c  = [c for c in df.columns if c.startswith("eeg_")]
    hbo_c  = [c for c in df.columns if c.startswith("hbo_")]
    hbr_c  = [c for c in df.columns if c.startswith("hbr_")]
    meta_c = ["subject_id","task","Cognitive_Load"]
    df = df[eeg_c + hbo_c + hbr_c + meta_c].fillna(0)

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\n{'=' * 62}")
    print(f"  DONE!")
    print(f"  Samples        : {len(df)}")
    print(f"  EEG features   : {len(eeg_c)}  (subjects with EEG: {eeg_ok_count})")
    print(f"  HbO features   : {len(hbo_c)}")
    print(f"  HbR features   : {len(hbr_c)}")
    print(f"  Total features : {len(eeg_c)+len(hbo_c)+len(hbr_c)}")
    for lbl, name in CLASS_NAMES.items():
        n = (df['Cognitive_Load'] == lbl).sum()
        print(f"  {name:25s}: {n}  ({n/len(df):.1%})")
    print(f"  Saved â†’ {output}")
    print("=" * 62)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nirs-dir", required=True)
    p.add_argument("--eeg-dir",  default=None)
    p.add_argument("--output",   default="data/tuberlin_eeg_fnirs.csv")
    args = p.parse_args()
    run(args.nirs_dir, args.eeg_dir, args.output)
