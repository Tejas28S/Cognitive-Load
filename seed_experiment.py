import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

# Load trial-level multimodal dataset
df = pd.read_csv('data/tuberlin_features.csv')
meta = ['subject_id', 'task', 'Cognitive_Load']
feats = [c for c in df.columns if c not in meta]
X = df[feats].values.astype(float)
y = df['Cognitive_Load'].values
groups = df['subject_id'].values

# Binary classification: 0 (Low, 0-back) vs 2 (High, 3-back)
mask = y != 1
X_b = X[mask]
y_b = y[mask].copy()
groups_b = groups[mask]
y_b[y_b == 2] = 1

SEEDS = [0, 7, 42, 123, 999]

def evaluate_seed(seed):
    logo = LeaveOneGroupOut()
    y_preds = np.empty(len(y_b), dtype=int)
    train_accs = []
    
    for train_idx, test_idx in logo.split(X_b, y_b, groups=groups_b):
        X_tr = X_b[train_idx]
        X_te = X_b[test_idx]
        y_tr = y_b[train_idx]
        
        # Local normalization using train fold stats
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-8
        X_tr_norm = (X_tr - mu) / sd
        X_te_norm = (X_te - mu) / sd
        
        model = Pipeline([
            ("scaler", RobustScaler()),
            ("sel", SelectKBest(score_func=f_classif, k=50)),
            ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=1))
        ])
        
        model.fit(X_tr_norm, y_tr)
        y_preds[test_idx] = model.predict(X_te_norm)
        train_accs.append(accuracy_score(y_tr, model.predict(X_tr_norm)))
        
    test_acc = accuracy_score(y_b, y_preds)
    test_f1 = f1_score(y_b, y_preds, average="macro")
    mean_train = np.mean(train_accs)
    gap = mean_train - test_acc
    
    return {
        "Seed": seed,
        "Train Acc": mean_train,
        "Test Acc": test_acc,
        "Macro F1": test_f1,
        "Gen Gap": gap
    }

print("=" * 65, flush=True)
print("  MULTI-SEED EVALUATION EXPERIMENT (Leave-One-Subject-Out CV)", flush=True)
print("=" * 65, flush=True)
print(f"Dataset: N={len(y_b)} samples, {X_b.shape[1]} features, {len(np.unique(groups_b))} subjects", flush=True)
print(f"Testing seeds: {SEEDS}\n", flush=True)

results = []
t0 = time.time()
for s in SEEDS:
    st0 = time.time()
    res = evaluate_seed(s)
    results.append(res)
    print(f"Seed {s:3d} | Test Acc: {res['Test Acc']*100:.2f}% | Macro F1: {res['Macro F1']:.4f} | Train Acc: {res['Train Acc']*100:.2f}% | Gap: {res['Gen Gap']*100:.2f}% (in {time.time()-st0:.1f}s)", flush=True)

res_df = pd.DataFrame(results)
print("\n" + "=" * 65, flush=True)
print("  MULTI-SEED BENCHMARK SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"Mean Test Accuracy : {res_df['Test Acc'].mean()*100:.2f}% ± {res_df['Test Acc'].std()*100:.2f}%", flush=True)
print(f"Min / Max Test Acc : {res_df['Test Acc'].min()*100:.2f}% / {res_df['Test Acc'].max()*100:.2f}% (Spread: {(res_df['Test Acc'].max() - res_df['Test Acc'].min())*100:.2f}%)", flush=True)
print(f"Mean Macro F1      : {res_df['Macro F1'].mean():.4f} ± {res_df['Macro F1'].std():.4f}", flush=True)
print(f"Mean Generalization Gap: {res_df['Gen Gap'].mean()*100:.2f}%", flush=True)
print("=" * 65, flush=True)
print(f"Total time elapsed : {time.time()-t0:.1f}s", flush=True)
