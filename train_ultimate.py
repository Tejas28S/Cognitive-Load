import numpy as np
import pandas as pd
import warnings, os, time
warnings.filterwarnings("ignore")

from sklearn.preprocessing    import StandardScaler, RobustScaler
from sklearn.pipeline         import Pipeline
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection  import LeaveOneGroupOut
from sklearn.metrics          import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from config import Config

cfg = Config()
print("=====================================================")
print("  Ultimate Binary Classifier (Subject-Normalized)")
print("=====================================================")

df = pd.read_csv(cfg.MERGED_CSV)
feats = [c for c in df.columns if c.startswith(("eeg_", "hbo_", "hbr_"))]

X_raw = df[feats].values.astype(float)
y = df["Cognitive_Load"].values
groups = df["subject_id"].values

# Filter for Binary (0 = Low, 2 = High)
mask = y != 1
X_raw = X_raw[mask]
y = y[mask]
groups = groups[mask]

# Remap 2 -> 1
y[y == 2] = 1

labels = ["Low (0-back)", "High (3-back)"]

print(f"Samples : {len(y)} | Features : {X_raw.shape[1]} | Subjects : {len(np.unique(groups))}")

def make_ensemble(k=50):
    return Pipeline([
        ("scaler", RobustScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
        ("clf", VotingClassifier(estimators=[
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=42)),
            ("gbm", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42))
        ], voting="soft", n_jobs=-1))
    ])

logo = LeaveOneGroupOut()
y_pred_all = np.empty(len(y), dtype=int)
train_accs = []

t0 = time.time()
print("▶ Training Soft Ensemble with Leave-One-Subject-Out CV (this will take a minute)...")
for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y, groups=groups)):
    X_tr_raw = X_raw[train_idx]
    X_te_raw = X_raw[test_idx]

    # strict subject-wise normalization using ONLY training subjects' stats
    mu = X_tr_raw.mean(axis=0)
    sd = X_tr_raw.std(axis=0) + 1e-8

    X_tr = (X_tr_raw - mu) / sd
    X_te = (X_te_raw - mu) / sd

    y_tr = y[train_idx]
    
    pipe = make_ensemble(k=50)
    pipe.fit(X_tr, y_tr)
    
    y_pred_tr = pipe.predict(X_tr)
    y_pred_te = pipe.predict(X_te)
    
    train_acc = accuracy_score(y_tr, y_pred_tr)
    test_acc = accuracy_score(y[test_idx], y_pred_te)
    train_accs.append(train_acc)
    y_pred_all[test_idx] = y_pred_te
    print(f"  Fold {fold+1:02d}/{len(np.unique(groups))} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

acc = accuracy_score(y, y_pred_all)
mean_train = np.mean(train_accs)
print(f"Done in {time.time()-t0:.0f}s")
print(f"\n[RESULTS] Ultimate Ensemble LOSO Accuracy: {acc:.4f} ({acc:.2%})")
print(f"[RESULTS] Average Train Accuracy:        {mean_train:.4f} ({mean_train:.2%})")
print(f"[RESULTS] Generalization Overfit Gap:    {mean_train - acc:.4f} ({(mean_train - acc):.2%})")
print("\nClassification Report:")
print(classification_report(y, y_pred_all, target_names=labels))
