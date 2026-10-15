import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from config import Config

cfg = Config()
df = pd.read_csv(cfg.MERGED_CSV)
meta = ["subject_id", "task", "Cognitive_Load"]
feats = [c for c in df.columns if c not in meta]
X_raw = df[feats].values.astype(float)
y = df["Cognitive_Load"].values
groups = df["subject_id"].values

mask = y != 1
X_raw = X_raw[mask]
y = y[mask]
groups = groups[mask]
y[y == 2] = 1

def make_slda():
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])

logo = LeaveOneGroupOut()

# Run all folds and compute average to give a comprehensive result
train_accs = []
test_accs = []

print("Running Leave-One-Group-Out CV for Shrinkage LDA...")
for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y, groups=groups)):
    X_tr_raw = X_raw[train_idx]
    X_te_raw = X_raw[test_idx]

    mu = X_tr_raw.mean(axis=0)
    sd = X_tr_raw.std(axis=0) + 1e-8
    X_tr = (X_tr_raw - mu) / sd
    X_te = (X_te_raw - mu) / sd
    y_tr = y[train_idx]
    y_te = y[test_idx]
    
    pipe = make_slda()
    pipe.fit(X_tr, y_tr)
    
    y_tr_pred = pipe.predict(X_tr)
    y_te_pred = pipe.predict(X_te)
    
    train_acc = accuracy_score(y_tr, y_tr_pred)
    test_acc = accuracy_score(y_te, y_te_pred)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Fold {fold+1} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

print("-" * 30)
print(f"Average Train Accuracy: {np.mean(train_accs):.4f}")
print(f"Average Test Accuracy: {np.mean(test_accs):.4f}")
