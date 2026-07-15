# ============================================================
#  diagnose_data.py  –  Run this BEFORE training to check if
#  your features actually contain cognitive load signal
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, ".")
from config import Config
cfg = Config()

print("=" * 60)
print("  DATA DIAGNOSTICS")
print("=" * 60)

# ── 1. Load ───────────────────────────────────────────────────
print("\n📂 Loading data...")
needed = cfg.EEG_COLUMNS + cfg.MODALITY2_COLUMNS + [cfg.LABEL_COLUMN]
df     = pd.read_csv(cfg.MERGED_CSV, usecols=needed)

X_eeg   = df[cfg.EEG_COLUMNS].values
X_fnirs = df[cfg.MODALITY2_COLUMNS].values
X_all   = np.hstack([X_eeg, X_fnirs])
y       = df[cfg.LABEL_COLUMN].values

print(f"  Samples : {len(y):,}")
print(f"  Features: {X_all.shape[1]} total ({X_eeg.shape[1]} EEG + {X_fnirs.shape[1]} fNIRS)")

# ── 2. Label distribution ─────────────────────────────────────
print(f"\n📊 Label distribution:")
for cls, name in enumerate(cfg.CLASS_NAMES):
    n = (y == cls).sum()
    print(f"  {cls} ({name:6s}): {n:,}  ({n/len(y):.1%})")

# ── 3. Feature statistics per class ───────────────────────────
print(f"\n📈 Mean feature values per class (first 5 EEG features):")
df_check = pd.DataFrame(X_eeg[:, :5],
                         columns=[f"EEG_{i}" for i in range(5)])
df_check['label'] = y
print(df_check.groupby('label').mean().round(4).to_string())

# ── 4. ANOVA F-test – do features differ across classes? ──────
print(f"\n🔬 ANOVA F-test (are features statistically different across classes?)")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
f_vals, p_vals = f_classif(X_scaled, y)
sig_feats = (p_vals < 0.05).sum()
print(f"  Significant features (p<0.05): {sig_feats} / {X_all.shape[1]}")
print(f"  Top 5 most discriminative features:")
top5 = np.argsort(f_vals)[::-1][:5]
all_cols = cfg.EEG_COLUMNS + cfg.MODALITY2_COLUMNS
for idx in top5:
    print(f"    {all_cols[idx]:20s}  F={f_vals[idx]:.2f}  p={p_vals[idx]:.4f}")

# ── 5. Quick ML sanity checks ─────────────────────────────────
print(f"\n🤖 Quick ML sanity checks (3-fold CV):")
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=500, C=1.0),
    "Random Forest (100)" : RandomForestClassifier(n_estimators=100,
                                                    n_jobs=-1,
                                                    random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_tr, y_tr, cv=3,
                              scoring='accuracy', n_jobs=-1)
    model.fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, model.predict(X_te))
    results[name] = test_acc
    print(f"  {name:25s}  CV: {scores.mean():.2%} ± {scores.std():.2%}  "
          f"Test: {test_acc:.2%}")

# ── 6. Verdict ────────────────────────────────────────────────
print(f"\n{'=' * 60}")
best_acc = max(results.values())
random_chance = 1 / cfg.NUM_CLASSES

if best_acc <= random_chance + 0.02:
    print("  ❌ VERDICT: Features have NO discriminative signal.")
    print("     Even Random Forest can't beat random chance.")
    print("     → The labels and features may not be aligned,")
    print("       or the dataset needs different preprocessing.")
elif best_acc < 0.50:
    print(f"  ⚠️  VERDICT: Weak signal detected ({best_acc:.2%}).")
    print("     Some information present but features need work.")
    print("     → Try feature engineering or different normalization.")
else:
    print(f"  ✅ VERDICT: Signal detected! RF achieves {best_acc:.2%}.")
    print("     Deep learning model should match or exceed this.")
    print("     → Proceed with DL training, tune hyperparameters.")
print(f"{'=' * 60}\n")

# ── 7. Feature importance (if RF learned something) ───────────
rf = models["Random Forest (100)"]
importances = rf.feature_importances_
top10_idx = np.argsort(importances)[::-1][:10]
print("  🏆 Top 10 most important features (Random Forest):")
for rank, idx in enumerate(top10_idx, 1):
    print(f"    {rank:2d}. {all_cols[idx]:25s}  importance={importances[idx]:.4f}")