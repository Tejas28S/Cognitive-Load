import os, sys
sys.path.insert(0, '.')
import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt

TASK = "nback"
TRIAL_SEC = 60

def bandpass(sig, lo, hi, fs, order=3):
    nyq = fs/2.0
    b,a = butter(order,[lo/nyq,hi/nyq],btype='band')
    return filtfilt(b,a,sig,axis=0)

# Use the first EEG folder — adjust path as needed
eeg_dir = sys.argv[1] if len(sys.argv) > 1 else "data/EEG_01-26_MATLAB"
subj_dirs = sorted([d for d in os.listdir(eeg_dir) if d.startswith("VP") and d.endswith("-EEG")])
if not subj_dirs:
    print("No VP*-EEG folders found in", eeg_dir)
    sys.exit(1)

subj_path = os.path.join(eeg_dir, subj_dirs[0])
cnt_path  = os.path.join(subj_path, f"cnt_{TASK}.mat")
mrk_path  = os.path.join(subj_path, f"mrk_{TASK}.mat")

print(f"Testing: {subj_dirs[0]}")
print(f"cnt path: {cnt_path}  exists={os.path.exists(cnt_path)}")

mat = sio.loadmat(cnt_path)
keys = [k for k in mat.keys() if not k.startswith('_')]
print(f"Keys: {keys}")

cnt_raw = mat[keys[0]]
print(f"cnt_raw.shape={cnt_raw.shape}  dtype.names={cnt_raw.dtype.names}")

cnt = cnt_raw
if hasattr(cnt,'dtype') and cnt.dtype.names:
    cnt = cnt[0,0]
    print(f"After [0,0]: type={type(cnt)}, dtype.names={cnt.dtype.names}")

print(f"Has 'x': {'x' in cnt.dtype.names}")
x_raw = cnt['x']
print(f"x_raw type={type(x_raw)}, shape={x_raw.shape}, dtype={x_raw.dtype}")

# Unwrap x
sig = (x_raw[0,0] if x_raw.shape==(1,1) else x_raw).astype(np.float64)
print(f"sig.shape={sig.shape}")

fs_raw = cnt['fs']
fs = int((fs_raw[0,0] if fs_raw.shape==(1,1) else fs_raw).flat[0])
print(f"fs={fs}")

win = TRIAL_SEC * fs
print(f"win={win} samples")

mrk_mat = sio.loadmat(mrk_path)
mrk_raw = mrk_mat[[k for k in mrk_mat if not k.startswith('_')][0]]
mrk = mrk_raw[0,0] if hasattr(mrk_raw,'dtype') and mrk_raw.dtype.names else mrk_raw
t_raw = mrk['time']
times = (t_raw[0,0] if t_raw.shape==(1,1) else t_raw).flatten()
y_raw = mrk['y']
y_mat = (y_raw[0,0] if y_raw.shape==(1,1) else y_raw)
labels = y_mat.argmax(axis=0)

print(f"Trials: {len(times)}, label dist: {np.bincount(labels)}")
print(f"Signal length: {sig.shape[0]}, need for all trials: {int((times[-1]/1000)*fs)+win}")
print(f"Fits: {int((times[-1]/1000)*fs)+win <= sig.shape[0]}")

extracted = 0
for t_ms, lbl in zip(times, labels):
    s = int((t_ms/1000)*fs)
    e = s + win
    if e <= sig.shape[0]:
        extracted += 1

print(f"Extractable trials: {extracted}/{len(times)}")
