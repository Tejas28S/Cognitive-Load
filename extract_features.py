# ============================================================
#  extract_features.py
#  Extracts EEG + fNIRS features from Shin 2019 .mat files
#  and saves a ready-to-train CSV
#
#  Dataset structure per subject file:
#    ch1..ch32  →  fNIRS signals (16 HbO + 16 HbR), fs=13.33Hz
#    mrk        →  trial markers + class labels
#    Classes:   1=Mental Arithmetic (High), 2=Motor Imagery (Medium), 3=Idle (Low)
#
#  Usage:
#    python extract_features.py                        # fNIRS only
#    python extract_features.py --eeg-dir data/EEG/   # fNIRS + EEG
# ============================================================

import os, sys, glob, argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch, butter, filtfilt
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────

FNIRS_DIR   = "data/fNIRS/"       # folder with subject_01.mat ... subject_18.mat
EEG_DIR     = None                 # set via --eeg-dir if available
OUTPUT_CSV  = "data/features_extracted.csv"

FNIRS_FS    = 13.333              # fNIRS sampling rate (Hz)
EEG_FS      = 200.0               # EEG sampling rate (Hz) - typical for this dataset
TRIAL_SEC   = 30                  # trial window in seconds

# Class mapping → cognitive load
# 1=Mental Arithmetic → High (2)
# 2=Motor Imagery     → Medium (1)
# 3=Idle State        → Low (0)
CLASS_MAP   = {1: 2, 2: 1, 3: 0}
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}

# fNIRS channel names (32 total: 16 HbO + 16 HbR)
HBO_CHANNELS  = [f"Ch{i}oxy"   for i in [1,2,3,4,5,7,8,9,10,11,12,14,15,16,17,18]]
HBRO_CHANNELS = [f"Ch{i}deoxy" for i in [1,2,3,4,5,7,8,9,10,11,12,14,15,16,17,18]]
ALL_FNIRS_CH  = HBO_CHANNELS + HBRO_CHANNELS  # 32 channels


# ── Signal processing helpers ─────────────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """Butterworth bandpass filter."""
    nyq  = fs / 2.0
    low  = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


def compute_fnirs_features(trial_signal, fs):
    """
    Extract features from one fNIRS trial.
    trial_signal: (n_samples, n_channels)
    Returns flat feature vector.
    """
    features = []
    for ch in range(trial_signal.shape[1]):
        sig = trial_signal[:, ch]

        # ── Statistical features ────────────────────────────
        features.append(np.mean(sig))           # mean amplitude
        features.append(np.std(sig))            # variability
        features.append(np.max(sig))            # peak response
        features.append(np.min(sig))            # trough
        features.append(np.max(sig) - np.min(sig))  # peak-to-peak
        features.append(skew(sig))              # asymmetry
        features.append(kurtosis(sig))          # peakedness

        # ── Slope (linear trend) ────────────────────────────
        t = np.arange(len(sig))
        slope = np.polyfit(t, sig, 1)[0]
        features.append(slope)

        # ── Mean of first half vs second half ───────────────
        mid = len(sig) // 2
        features.append(np.mean(sig[:mid]))     # early response
        features.append(np.mean(sig[mid:]))     # late response
        features.append(np.mean(sig[mid:]) - np.mean(sig[:mid]))  # change

    return np.array(features, dtype=np.float32)


def compute_eeg_features(trial_signal, fs):
    """
    Extract frequency-band power features from EEG trial.
    trial_signal: (n_samples, n_channels)
    Returns flat feature vector.
    """
    bands = {
        "delta": (0.5, 4),
        "theta": (4,   8),
        "alpha": (8,  13),
        "beta" : (13, 30),
        "gamma": (30, 45),
    }
    features = []
    for ch in range(trial_signal.shape[1]):
        sig = trial_signal[:, ch]
        freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)//2))

        for band, (lo, hi) in bands.items():
            idx  = np.where((freqs >= lo) & (freqs < hi))[0]
            power = np.mean(psd[idx]) if len(idx) > 0 else 0.0
            features.append(power)

        # Engagement index: beta / (alpha + theta)
        theta_p = np.mean(psd[np.where((freqs>=4) & (freqs<8))[0]])  + 1e-10
        alpha_p = np.mean(psd[np.where((freqs>=8) & (freqs<13))[0]]) + 1e-10
        beta_p  = np.mean(psd[np.where((freqs>=13)& (freqs<30))[0]]) + 1e-10
        features.append(beta_p / (alpha_p + theta_p))

    return np.array(features, dtype=np.float32)


# ── Load one subject's fNIRS data ─────────────────────────────

def load_fnirs_subject(filepath):
    """
    Returns:
        trials  : list of (n_samples, 32) arrays  [one per trial]
        labels  : list of cognitive load labels (0/1/2)
        subject : subject index
    """
    mat    = sio.loadmat(filepath)
    mrk    = mat['mrk']
    times  = mrk['time'][0,0].flatten()          # ms
    labels = mrk['event'][0,0]['desc'][0,0].flatten()
    fs     = FNIRS_FS
    win    = int(TRIAL_SEC * fs)

    # Stack all 32 channels → (total_samples, 32)
    ch_keys = sorted([k for k in mat.keys() if k.startswith('ch')],
                     key=lambda x: int(x[2:]))
    signal  = np.hstack([mat[k] for k in ch_keys])  # (T, 32)

    trials, y = [], []
    for t_ms, lbl in zip(times, labels):
        start = int((t_ms / 1000.0) * fs)
        end   = start + win
        if end > signal.shape[0]:
            continue
        trial = signal[start:end, :]   # (win, 32)

        # Simple baseline correction: subtract mean of first 2s
        baseline_end = int(2 * fs)
        trial = trial - trial[:baseline_end, :].mean(axis=0)

        trials.append(trial)
        y.append(CLASS_MAP[int(lbl)])

    return trials, y


# ── Load one subject's EEG data (if available) ───────────────

def load_eeg_subject(filepath, eeg_fs=200.0):
    """
    Loads EEG .mat file (same marker structure as fNIRS file).
    Returns trials list and labels.
    """
    try:
        mat    = sio.loadmat(filepath)
        mrk    = mat['mrk']
        times  = mrk['time'][0,0].flatten()
        labels = mrk['event'][0,0]['desc'][0,0].flatten()
        win    = int(TRIAL_SEC * eeg_fs)

        # EEG data likely in 'dat' or channel arrays
        if 'dat' in mat:
            dat_sig = mat['dat']['signal'][0,0]
            if hasattr(dat_sig, 'shape') and dat_sig.ndim == 2:
                signal = dat_sig
            else:
                return None, None
        else:
            ch_keys = sorted([k for k in mat.keys() if k.startswith('ch')],
                             key=lambda x: int(x[2:]))
            signal = np.hstack([mat[k] for k in ch_keys])

        # Bandpass 0.5–45 Hz
        signal = bandpass_filter(signal, 0.5, 45, eeg_fs)

        trials, y = [], []
        for t_ms, lbl in zip(times, labels):
            start = int((t_ms / 1000.0) * eeg_fs)
            end   = start + win
            if end > signal.shape[0]:
                continue
            trial = signal[start:end, :]
            trials.append(trial)
            y.append(CLASS_MAP[int(lbl)])

        return trials, y
    except Exception as e:
        print(f"    ⚠️  EEG load failed: {e}")
        return None, None


# ── Main extraction loop ──────────────────────────────────────

def extract_all(fnirs_dir, eeg_dir=None):
    fnirs_files = sorted(glob.glob(os.path.join(fnirs_dir, "subject_*.mat")))
    if not fnirs_files:
        raise FileNotFoundError(
            f"No subject_*.mat files found in '{fnirs_dir}'.\n"
            f"Place your subject_01.mat ... subject_18.mat files there."
        )

    print(f"Found {len(fnirs_files)} fNIRS subject files")
    has_eeg = eeg_dir is not None and os.path.isdir(eeg_dir)
    if has_eeg:
        print(f"EEG directory: {eeg_dir}")
    else:
        print("EEG directory: NOT provided → fNIRS-only extraction")
        print("(Add --eeg-dir path/to/EEG/ to include EEG features)")

    all_rows = []
    fnirs_feat_names = None
    eeg_feat_names   = None

    for subj_idx, fnirs_path in enumerate(fnirs_files, 1):
        subj_name = os.path.basename(fnirs_path).replace('.mat','')
        print(f"\n  [{subj_idx:2d}/{len(fnirs_files)}] {subj_name}", end="")

        # ── fNIRS ──────────────────────────────────────────
        fnirs_trials, fnirs_labels = load_fnirs_subject(fnirs_path)
        print(f"  → {len(fnirs_trials)} fNIRS trials", end="")

        # ── EEG (optional) ─────────────────────────────────
        eeg_trials = None
        if has_eeg:
            eeg_path = os.path.join(eeg_dir, os.path.basename(fnirs_path))
            if os.path.exists(eeg_path):
                eeg_trials, eeg_labels = load_eeg_subject(eeg_path, EEG_FS)
                if eeg_trials:
                    print(f"  + {len(eeg_trials)} EEG trials", end="")

        # ── Build feature names (first subject only) ───────
        if fnirs_feat_names is None:
            sample_fn = compute_fnirs_features(fnirs_trials[0], FNIRS_FS)
            n_fn = len(sample_fn)
            fnirs_feat_names = [f"fNIRS_{i}" for i in range(n_fn)]
            print(f"\n     fNIRS features per trial: {n_fn}")

            if eeg_trials:
                sample_eg = compute_eeg_features(eeg_trials[0], EEG_FS)
                n_eg = len(sample_eg)
                eeg_feat_names = [f"EEG_{i}" for i in range(n_eg)]
                print(f"     EEG features per trial  : {n_eg}")

        # ── Extract features per trial ─────────────────────
        use_eeg = eeg_trials is not None and len(eeg_trials) == len(fnirs_trials)
        for i, (fn_trial, label) in enumerate(zip(fnirs_trials, fnirs_labels)):
            fn_feats = compute_fnirs_features(fn_trial, FNIRS_FS)
            row = {}

            if use_eeg and i < len(eeg_trials):
                eg_feats = compute_eeg_features(eeg_trials[i], EEG_FS)
                for j, v in enumerate(eg_feats):
                    row[f"EEG_{j}"] = v

            for j, v in enumerate(fn_feats):
                row[f"fNIRS_{j}"] = v

            row["subject_id"]      = subj_idx
            row["Cognitive_Load"]  = label
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # ── Reorder columns: EEG first, then fNIRS, then meta ──
    eeg_cols   = sorted([c for c in df.columns if c.startswith('EEG_')],
                        key=lambda x: int(x.split('_')[1]))
    fnirs_cols = sorted([c for c in df.columns if c.startswith('fNIRS_')],
                        key=lambda x: int(x.split('_')[1]))
    other_cols = ["subject_id", "Cognitive_Load"]
    df = df[eeg_cols + fnirs_cols + other_cols]

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n\n{'='*55}")
    print(f"  ✅ Feature extraction complete!")
    print(f"     Total samples : {len(df):,}")
    print(f"     EEG features  : {len(eeg_cols)}")
    print(f"     fNIRS features: {len(fnirs_cols)}")
    print(f"     Subjects      : {df['subject_id'].nunique()}")
    print(f"\n  Class distribution:")
    for lbl, name in CLASS_NAMES.items():
        n = (df['Cognitive_Load'] == lbl).sum()
        print(f"     {lbl} ({name:6s}): {n:,}  ({n/len(df):.1%})")
    print(f"\n  Saved → {OUTPUT_CSV}")
    print(f"{'='*55}\n")
    return df


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnirs-dir", default="data/fNIRS/",
                        help="Folder with subject_*.mat fNIRS files")
    parser.add_argument("--eeg-dir",   default=None,
                        help="Folder with subject_*.mat EEG files (optional)")
    parser.add_argument("--output",    default="data/features_extracted.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    OUTPUT_CSV = args.output
    df = extract_all(args.fnirs_dir, args.eeg_dir)
