import io
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis

# Constants
FNIRS_FS = 10
EEG_FS = 200
WINDOW_SEC = 5.0
STRIDE_SEC = 1.0  # 1-second sliding stride

def bandpass(sig, lo, hi, fs, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

def baseline_correct(trial, fs, sec=5):
    n = int(sec * fs)
    if n == 0 or len(trial) < n:
        return trial - trial.mean(axis=0, keepdims=True)
    return trial - trial[:n].mean(axis=0, keepdims=True)

def fnirs_features(trial, fs):
    """13 stats per channel. trial: (T, ch)"""
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

def process_raw_mats(eeg_bytes, nirs_bytes):
    # Load NIRS
    nirs_mat = sio.loadmat(io.BytesIO(nirs_bytes))
    cnt_nirs = None
    for k in nirs_mat.keys():
        if k.startswith('cnt_'):
            cnt_nirs = nirs_mat[k]
            break
            
    if cnt_nirs is None:
        raise ValueError("No cnt_* struct found in NIRS file.")
        
    oxy_sig = cnt_nirs['oxy'][0,0]['x'][0,0].astype(np.float64)
    deoxy_sig = cnt_nirs['deoxy'][0,0]['x'][0,0].astype(np.float64)
    fs_nirs = int(cnt_nirs['oxy'][0,0]['fs'][0,0].flat[0])
    oxy_f = bandpass(oxy_sig, 0.01, 0.1, fs_nirs)
    deoxy_f = bandpass(deoxy_sig, 0.01, 0.1, fs_nirs)
    
    # Load EEG
    eeg_mat = sio.loadmat(io.BytesIO(eeg_bytes))
    cnt_eeg = None
    for k in eeg_mat.keys():
        if k.startswith('cnt_'):
            cnt_eeg = eeg_mat[k]
            break
            
    if cnt_eeg is None:
        raise ValueError("No cnt_* struct found in EEG file.")
        
    if hasattr(cnt_eeg, 'dtype') and cnt_eeg.dtype.names:
        cnt_eeg = cnt_eeg[0, 0]
    
    x_raw = cnt_eeg['x']
    eeg_sig = (x_raw[0,0] if x_raw.shape==(1,1) else x_raw).astype(np.float64)
    fs_raw = cnt_eeg['fs']
    fs_eeg = int((fs_raw[0,0] if fs_raw.shape==(1,1) else fs_raw).flat[0])
    eeg_f = bandpass(eeg_sig, 0.5, 45.0, fs_eeg)

    # Alignment and slicing
    duration = min(oxy_f.shape[0] / fs_nirs, eeg_f.shape[0] / fs_eeg)
    n_windows = int((duration - WINDOW_SEC) / STRIDE_SEC) + 1
    
    if n_windows <= 0:
        raise ValueError("Recording too short for a single sliding window.")

    n_eeg_ch = eeg_f.shape[1]
    n_hbo_ch = oxy_f.shape[1]
    
    eeg_names = eeg_col_names(n_eeg_ch)
    hbo_names = fnirs_col_names("hbo_", n_hbo_ch)
    hbr_names = fnirs_col_names("hbr_", n_hbo_ch)
    
    all_rows = []
    
    for i in range(n_windows):
        start_sec = i * STRIDE_SEC
        
        # EEG slice
        e_start = int(round(start_sec * fs_eeg))
        e_end = e_start + int(round(WINDOW_SEC * fs_eeg))
        e_window = np.nan_to_num(eeg_f[e_start:e_end])
        
        # NIRS slice
        n_start = int(round(start_sec * fs_nirs))
        n_end = n_start + int(round(WINDOW_SEC * fs_nirs))
        
        n_window_o = np.nan_to_num(oxy_f[n_start:n_end])
        n_window_o = baseline_correct(n_window_o, fs_nirs)
        
        n_window_r = np.nan_to_num(deoxy_f[n_start:n_end])
        n_window_r = baseline_correct(n_window_r, fs_nirs)
        
        row = {}
        for col_name, value in zip(eeg_names, eeg_features(e_window, fs_eeg)):
            row[col_name] = float(value)
        for col_name, value in zip(hbo_names, fnirs_features(n_window_o, fs_nirs)):
            row[col_name] = float(value)
        for col_name, value in zip(hbr_names, fnirs_features(n_window_r, fs_nirs)):
            row[col_name] = float(value)
            
        # Optional metadata
        row['window_id'] = i
        row['window_start_sec'] = start_sec
        row['subject_id'] = 'LIVE_MAT'
        row['trial_id'] = 'LIVE_MAT_001'
        
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df
