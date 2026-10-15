"""
within_subject_experiment.py
Evaluates:
  1) Fast Linear Ensemble (k=50) [LDA + LR + SVC with SelectKBest(k=50)]
  2) Baseline Hybrid CNN-LSTM (MLP encoders + Attention Fusion, 1,476 features)
Under within-subject evaluation protocols:
  A) Pooled Within-Subject Split (Stratified 5-Fold across all 21,216 windows)
  B) Per-Subject Personalized Within-Subject (5-Fold CV per subject, averaged over 26 subjects)
"""

import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from config import Config

# --- 1. Load Data ---
cfg = Config()
df = pd.read_csv(cfg.MERGED_CSV)

eeg_cols = [c for c in df.columns if c.startswith("eeg_")]
fnirs_cols = [c for c in df.columns if c.startswith(("hbo_", "hbr_"))]
all_feats = eeg_cols + fnirs_cols

print(f"Loaded dataset: {len(df)} total rows")
print(f"Features: {len(eeg_cols)} EEG + {len(fnirs_cols)} fNIRS = {len(all_feats)} total")

# Binary filter: 0 = Low, 2 = High (drop 1 = Medium)
mask = df["Cognitive_Load"].values != 1
df_bin = df[mask].copy()

X_eeg_raw = df_bin[eeg_cols].values.astype(np.float32)
X_fnirs_raw = df_bin[fnirs_cols].values.astype(np.float32)
X_all_raw = df_bin[all_feats].values.astype(np.float32)

y_raw = df_bin["Cognitive_Load"].values
y = np.where(y_raw == 2, 1, 0).astype(np.int64)
subjects = df_bin["subject_id"].values

print(f"Binary rows: {len(y)} ({np.sum(y == 0)} Low, {np.sum(y == 1)} High across {len(np.unique(subjects))} subjects)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Model Definitions ---

def make_fast_linear_ensemble(k=50):
    return Pipeline([
        ("scaler", RobustScaler()),
        ("sel", SelectKBest(score_func=f_classif, k=k)),
        ("clf", VotingClassifier(estimators=[
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ("lr", LogisticRegression(C=0.1, max_iter=500, random_state=42)),
            ("svc", CalibratedClassifierCV(LinearSVC(C=0.01, random_state=42, max_iter=2000), cv=3))
        ], voting="soft"))
    ])

# Hybrid CNN-LSTM Architecture (matching served baseline)
class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class ModalityMLP(nn.Module):
    def __init__(self, n_features, proj_dim, out_dim, n_blocks=2, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, proj_dim), nn.BatchNorm1d(proj_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResMLPBlock(proj_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.output_proj(self.blocks(self.input_proj(x)))

class AttentionFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.gate_eeg = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.gate_fnirs = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.BatchNorm1d(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        combined = torch.cat([eeg, fnirs], dim=-1)
        weighted = torch.cat(
            [eeg * self.gate_eeg(combined), fnirs * self.gate_fnirs(combined)], dim=-1
        )
        return self.fc(weighted)

class HybridCNNLSTM(nn.Module):
    def __init__(self, n_eeg=540, n_fnirs=936, proj_dim=96, enc_dim=48, n_blocks=2, encoder_dropout=0.5, fc_units=(48, 24), fc_dropout=0.6, num_classes=2):
        super().__init__()
        self.eeg_encoder = ModalityMLP(n_eeg, proj_dim, enc_dim, n_blocks, encoder_dropout)
        self.fnirs_encoder = ModalityMLP(n_fnirs, proj_dim, enc_dim, n_blocks, encoder_dropout)
        self.fusion = AttentionFusion(enc_dim, enc_dim, encoder_dropout)
        head, in_d = [], enc_dim
        for units in fc_units:
            head += [nn.Linear(in_d, units), nn.BatchNorm1d(units), nn.ReLU(), nn.Dropout(fc_dropout)]
            in_d = units
        head.append(nn.Linear(in_d, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, eeg, fnirs):
        return self.head(self.fusion(self.eeg_encoder(eeg), self.fnirs_encoder(fnirs)))

def train_eval_hybrid(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te, y_te, epochs=25, batch_size=128, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Scale inputs
    es = RobustScaler()
    fs = RobustScaler()
    eeg_tr_s = es.fit_transform(eeg_tr)
    eeg_te_s = es.transform(eeg_te)
    fnirs_tr_s = fs.fit_transform(fnirs_tr)
    fnirs_te_s = fs.transform(fnirs_te)
    
    model = HybridCNNLSTM(n_eeg=eeg_tr.shape[1], n_fnirs=fnirs_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_ds = TensorDataset(torch.tensor(eeg_tr_s, dtype=torch.float32),
                             torch.tensor(fnirs_tr_s, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Train loop
    model.train()
    for ep in range(epochs):
        for e_b, f_b, y_b in train_loader:
            e_b, f_b, y_b = e_b.to(device), f_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(e_b, f_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    with torch.no_grad():
        # Train acc
        tr_preds = []
        for e_b, f_b, _ in DataLoader(train_ds, batch_size=batch_size, shuffle=False):
            out = model(e_b.to(device), f_b.to(device))
            tr_preds.append(out.argmax(dim=-1).cpu().numpy())
        tr_pred = np.concatenate(tr_preds)
        train_acc = accuracy_score(y_tr, tr_pred)
        
        # Test acc
        te_ds = TensorDataset(torch.tensor(eeg_te_s, dtype=torch.float32),
                              torch.tensor(fnirs_te_s, dtype=torch.float32),
                              torch.tensor(y_te, dtype=torch.long))
        te_preds = []
        for e_b, f_b, _ in DataLoader(te_ds, batch_size=batch_size, shuffle=False):
            out = model(e_b.to(device), f_b.to(device))
            te_preds.append(out.argmax(dim=-1).cpu().numpy())
        te_pred = np.concatenate(te_preds)
        test_acc = accuracy_score(y_te, te_pred)
        test_f1 = f1_score(y_te, te_pred, average="macro")
        
    return train_acc, test_acc, test_f1

# --- 3. EXPERIMENT A: Pooled Within-Subject 5-Fold Split ---
print("\n" + "="*70)
print("EXPERIMENT A: POOLED WITHIN-SUBJECT 5-FOLD SPLIT (All 26 Subjects Mixed)")
print("="*70)
print("Every subject has 80% of their windows in Train and 20% in Test.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

linear_train_accs = []
linear_test_accs = []
linear_test_f1s = []

hybrid_train_accs = []
hybrid_test_accs = []
hybrid_test_f1s = []

t0 = time.time()
for fold, (train_idx, test_idx) in enumerate(skf.split(X_all_raw, y)):
    print(f"\n--- Fold {fold+1}/5 ---")
    
    # 1. Fast Linear Ensemble (k=50)
    t_lin_0 = time.time()
    pipe = make_fast_linear_ensemble(k=50)
    pipe.fit(X_all_raw[train_idx], y[train_idx])
    
    tr_pred_lin = pipe.predict(X_all_raw[train_idx])
    te_pred_lin = pipe.predict(X_all_raw[test_idx])
    
    tr_acc_lin = accuracy_score(y[train_idx], tr_pred_lin)
    te_acc_lin = accuracy_score(y[test_idx], te_pred_lin)
    te_f1_lin = f1_score(y[test_idx], te_pred_lin, average="macro")
    
    linear_train_accs.append(tr_acc_lin)
    linear_test_accs.append(te_acc_lin)
    linear_test_f1s.append(te_f1_lin)
    print(f"[Fast Linear Ensemble (k=50)] Train Acc: {tr_acc_lin:.4f} | Test Acc: {te_acc_lin:.4f} | F1: {te_f1_lin:.4f} ({time.time()-t_lin_0:.1f}s)")
    
    # 2. Hybrid CNN-LSTM Baseline
    t_hyb_0 = time.time()
    tr_acc_hyb, te_acc_hyb, te_f1_hyb = train_eval_hybrid(
        X_eeg_raw[train_idx], X_fnirs_raw[train_idx], y[train_idx],
        X_eeg_raw[test_idx], X_fnirs_raw[test_idx], y[test_idx],
        epochs=20, batch_size=128, lr=1e-3, seed=42 + fold
    )
    hybrid_train_accs.append(tr_acc_hyb)
    hybrid_test_accs.append(te_acc_hyb)
    hybrid_test_f1s.append(te_f1_hyb)
    print(f"[Hybrid CNN-LSTM Baseline  ] Train Acc: {tr_acc_hyb:.4f} | Test Acc: {te_acc_hyb:.4f} | F1: {te_f1_hyb:.4f} ({time.time()-t_hyb_0:.1f}s)")

print("\n" + "-"*70)
print(f"POOLED WITHIN-SUBJECT 5-FOLD SUMMARY:")
print(f"  Fast Linear Ensemble (k=50):")
print(f"    Train Acc: {np.mean(linear_train_accs):.4f} ± {np.std(linear_train_accs):.4f}")
print(f"    Test Acc : {np.mean(linear_test_accs):.4f} ± {np.std(linear_test_accs):.4f}")
print(f"    Test F1  : {np.mean(linear_test_f1s):.4f} ± {np.std(linear_test_f1s):.4f}")
print(f"    Gap      : {np.mean(linear_train_accs) - np.mean(linear_test_accs):.4f}")
print(f"  Hybrid CNN-LSTM Baseline (1,476 features):")
print(f"    Train Acc: {np.mean(hybrid_train_accs):.4f} ± {np.std(hybrid_train_accs):.4f}")
print(f"    Test Acc : {np.mean(hybrid_test_accs):.4f} ± {np.std(hybrid_test_accs):.4f}")
print(f"    Test F1  : {np.mean(hybrid_test_f1s):.4f} ± {np.std(hybrid_test_f1s):.4f}")
print(f"    Gap      : {np.mean(hybrid_train_accs) - np.mean(hybrid_test_accs):.4f}")
print("-"*70)

# --- 4. EXPERIMENT B: Per-Subject Personalized Within-Subject (5-Fold CV per individual subject) ---
print("\n" + "="*70)
print("EXPERIMENT B: INDIVIDUAL PERSONALIZED WITHIN-SUBJECT (5-Fold CV per Subject)")
print("="*70)
print("For each of the 26 subjects, train on 80% of their trials, test on 20% of their trials.")

unique_subjs = np.unique(subjects)
subj_linear_accs = []
subj_linear_gaps = []
subj_hybrid_accs = []
subj_hybrid_gaps = []

for s_idx, s in enumerate(unique_subjs):
    s_mask = subjects == s
    X_s_all = X_all_raw[s_mask]
    X_s_eeg = X_eeg_raw[s_mask]
    X_s_fnirs = X_fnirs_raw[s_mask]
    y_s = y[s_mask]
    
    # 5-fold CV for this subject
    skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_lin_tr, s_lin_te = [], []
    s_hyb_tr, s_hyb_te = [], []
    
    for tr_i, te_i in skf_s.split(X_s_all, y_s):
        # Linear
        p = make_fast_linear_ensemble(k=50)
        p.fit(X_s_all[tr_i], y_s[tr_i])
        s_lin_tr.append(accuracy_score(y_s[tr_i], p.predict(X_s_all[tr_i])))
        s_lin_te.append(accuracy_score(y_s[te_i], p.predict(X_s_all[te_i])))
        
        # Hybrid
        tr_acc_h, te_acc_h, _ = train_eval_hybrid(
            X_s_eeg[tr_i], X_s_fnirs[tr_i], y_s[tr_i],
            X_s_eeg[te_i], X_s_fnirs[te_i], y_s[te_i],
            epochs=20, batch_size=64, lr=1e-3, seed=42
        )
        s_hyb_tr.append(tr_acc_h)
        s_hyb_te.append(te_acc_h)
        
    mean_lin_te = np.mean(s_lin_te)
    mean_lin_gap = np.mean(s_lin_tr) - mean_lin_te
    subj_linear_accs.append(mean_lin_te)
    subj_linear_gaps.append(mean_lin_gap)
    
    mean_hyb_te = np.mean(s_hyb_te)
    mean_hyb_gap = np.mean(s_hyb_tr) - mean_hyb_te
    subj_hybrid_accs.append(mean_hyb_te)
    subj_hybrid_gaps.append(mean_hyb_gap)
    
    print(f"Subject {s:02d} ({s_idx+1:02d}/26) | Fast Linear Acc: {mean_lin_te:.4f} (gap {mean_lin_gap:.4f}) | Hybrid CNN-LSTM Acc: {mean_hyb_te:.4f} (gap {mean_hyb_gap:.4f})")

print("\n" + "="*70)
print("FINAL PERSONALIZED WITHIN-SUBJECT SUMMARY (Average across all 26 Subjects):")
print(f"  Fast Linear Ensemble (k=50):")
print(f"    Personalized Test Acc: {np.mean(subj_linear_accs):.4f} ± {np.std(subj_linear_accs):.4f}")
print(f"    Average Overfit Gap  : {np.mean(subj_linear_gaps):.4f}")
print(f"  Hybrid CNN-LSTM Baseline (1,476 features):")
print(f"    Personalized Test Acc: {np.mean(subj_hybrid_accs):.4f} ± {np.std(subj_hybrid_accs):.4f}")
print(f"    Average Overfit Gap  : {np.mean(subj_hybrid_gaps):.4f}")
print("="*70)
