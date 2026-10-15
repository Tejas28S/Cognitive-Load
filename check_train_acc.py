import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from config import Config

cfg = Config()
df = pd.read_csv(cfg.MERGED_CSV)
feats = [c for c in df.columns if c.startswith(("eeg_", "hbo_", "hbr_"))]
X_raw = df[feats].values.astype(float)
y = df["Cognitive_Load"].values
groups = df["subject_id"].values

mask = y != 1
X_raw = X_raw[mask]
y = y[mask]
groups = groups[mask]
y[y == 2] = 1

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
# Run all folds and compute average
train_accs = []
test_accs = []

print("Running Leave-One-Group-Out CV for Voting Classifier Ensemble...")
for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y, groups=groups)):
    X_tr_raw = X_raw[train_idx]
    X_te_raw = X_raw[test_idx]

    mu = X_tr_raw.mean(axis=0)
    sd = X_tr_raw.std(axis=0) + 1e-8
    X_tr = (X_tr_raw - mu) / sd
    X_te = (X_te_raw - mu) / sd
    y_tr = y[train_idx]
    y_te = y[test_idx]
    
    pipe = make_ensemble()
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
