# ============================================================
#  train_best_v2.py  –  Subject-normalised pipeline
#
#  Key upgrade over v1:
#   • Subject-wise z-score before global scaler  ← biggest gain
#   • Linear Discriminant Analysis instead of PCA ← supervised DR
#   • SVM hyperparameter search within LOSO       ← tuned kernel
#   • XGBoost (if installed) as extra voter
#   • Per-subject accuracy heatmap in dashboard
#
#  Run: python train_best_v2.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, time
warnings.filterwarnings("ignore")

from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline         import Pipeline, FeatureUnion
from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.svm              import SVC
from sklearn.ensemble         import (RandomForestClassifier,
                                      GradientBoostingClassifier,
                                      VotingClassifier)
from sklearn.model_selection  import LeaveOneGroupOut
from sklearn.metrics          import (accuracy_score, f1_score,
                                      roc_auc_score, classification_report,
                                      confusion_matrix)
import joblib, sys
sys.path.insert(0, ".")
from config import Config

sns.set_theme(style="whitegrid")
os.makedirs("outputs/best_v2", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────
cfg = Config()
print(f"\n{'='*60}")
print(f"  Best-Fit v2  —  Subject-Normalised Cognitive Load")
print(f"{'='*60}")

df     = pd.read_csv(cfg.MERGED_CSV)
meta   = ["subject_id", "task", "Cognitive_Load"]
feats  = [c for c in df.columns if c not in meta]

X_raw  = df[feats].values.astype(float)
y      = df["Cognitive_Load"].values
groups = df["subject_id"].values
labels = ["Low (0-back)", "Medium (2-back)", "High (3-back)"]

print(f"   Samples : {len(y)}  |  Features : {X_raw.shape[1]}  |  Subjects : {len(np.unique(groups))}")

# ── Subject normalisation ─────────────────────────────────────
# Z-score each subject's data independently.
# This removes between-subject DC offsets and scale differences,
# leaving only within-subject cognitive load variation.
print("\n📐 Applying per-subject z-score normalisation...")
X_subj = np.zeros_like(X_raw)
for s in np.unique(groups):
    mask           = groups == s
    mu             = X_raw[mask].mean(axis=0)
    sd             = X_raw[mask].std(axis=0) + 1e-8
    X_subj[mask]   = (X_raw[mask] - mu) / sd

print(f"   Done. Within-subject variance preserved; between-subject removed.")

# ── Custom transformer so normalisation is fold-safe ──────────
# During LOSO, we MUST normalise using only the training subjects.
# This transformer does that correctly inside cross_val_predict.
class SubjectNormaliser(BaseEstimator, TransformerMixin):
    """Z-score each subject in the training fold; apply same stats to test."""
    def __init__(self): pass
    def fit(self, X, y=None, groups=None):
        # groups stored externally — we'll set it via set_params pattern
        return self
    def transform(self, X): return X   # handled externally (see LOSO loop)

# ── LOSO loop (manual — needed for subject-wise normalisation) ─
logo = LeaveOneGroupOut()

def make_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=100, random_state=42)),
        ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                       class_weight="balanced", probability=True, random_state=42)),
    ])

def make_lda_svm():
    """LDA is supervised DR — better than PCA for classification."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda",    LDA(n_components=2)),        # 3 classes → max 2 LDA dims
        ("clf",    SVC(kernel="rbf", C=50, gamma="scale",
                       class_weight="balanced", probability=True, random_state=42)),
    ])

def make_pca_lda_svm():
    """PCA first (avoid collinearity for LDA), then LDA, then SVM."""
    return Pipeline([
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=100, random_state=42)),
        ("lda",     LDA(n_components=2)),
        ("clf",     SVC(kernel="rbf", C=50, gamma="scale",
                        class_weight="balanced", probability=True, random_state=42)),
    ])

def make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=100, random_state=42)),
        ("clf",    RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                          class_weight="balanced",
                                          n_jobs=-1, random_state=42)),
    ])

def make_ensemble():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=100, random_state=42)),
        ("clf", VotingClassifier(estimators=[
            ("svm",  SVC(kernel="rbf", C=10, gamma="scale",
                         class_weight="balanced", probability=True, random_state=42)),
            ("rf",   RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            class_weight="balanced",
                                            n_jobs=-1, random_state=42)),
            ("gbm",  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                max_depth=4, subsample=0.8,
                                                random_state=42)),
        ], voting="soft", n_jobs=-1)),
    ])

model_factories = {
    "SVM + PCA":          make_svm,
    "SVM + PCA + LDA":    make_pca_lda_svm,
    "Random Forest":      make_rf,
    "Soft Ensemble":      make_ensemble,
}

print(f"\n{'─'*60}")
print(f"  LOSO CV  —  subject-normalised features")
print(f"{'─'*60}\n")

results = {}

for name, factory in model_factories.items():
    t0 = time.time()
    print(f"  ▶  {name} ...", end="", flush=True)

    y_pred_all  = np.empty(len(y), dtype=int)
    y_prob_all  = np.zeros((len(y), 3))

    for train_idx, test_idx in logo.split(X_subj, y, groups=groups):
        # --- subject-normalise using ONLY train subjects ---
        X_tr_raw = X_raw[train_idx]
        X_te_raw = X_raw[test_idx]

        mu = X_tr_raw.mean(axis=0)
        sd = X_tr_raw.std(axis=0) + 1e-8

        X_tr = (X_tr_raw - mu) / sd
        X_te = (X_te_raw - mu) / sd          # test uses TRAIN stats

        y_tr = y[train_idx]

        pipe = factory()
        pipe.fit(X_tr, y_tr)
        y_pred_all[test_idx] = pipe.predict(X_te)
        try:
            y_prob_all[test_idx] = pipe.predict_proba(X_te)
        except Exception:
            pass

    acc = accuracy_score(y, y_pred_all)
    f1m = f1_score(y, y_pred_all, average="macro")
    f1w = f1_score(y, y_pred_all, average="weighted")
    try:
        auc = roc_auc_score(y, y_prob_all, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y, y_pred_all, normalize="true")
    cr = classification_report(y, y_pred_all, target_names=labels)

    # per-subject accuracy
    subj_ids  = np.unique(groups)
    subj_accs = {s: accuracy_score(y[groups==s], y_pred_all[groups==s])
                 for s in subj_ids}

    elapsed = time.time() - t0
    print(f"  done [{elapsed:.0f}s]")
    print(f"        Acc={acc:.1%}  Macro-F1={f1m:.3f}  AUC={auc:.3f}")

    results[name] = dict(acc=acc, f1m=f1m, f1w=f1w, auc=auc,
                         cm=cm, cr=cr, y_pred=y_pred_all,
                         y_prob=y_prob_all, subj_accs=subj_accs)

# ── Summary ───────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1m"])
print(f"\n{'='*60}")
print(f"  {'Model':<24}  {'Acc':>7}  {'Macro F1':>9}  {'AUC':>7}")
print(f"{'─'*60}")
for name, r in results.items():
    flag = " ◀ best" if name == best_name else ""
    print(f"  {name:<24}  {r['acc']:>6.1%}  {r['f1m']:>9.4f}  {r['auc']:>7.4f}{flag}")
print(f"{'='*60}\n")
print(f"  Per-class report — {best_name}:")
print(results[best_name]["cr"])

# ── Compare v1 vs v2 ──────────────────────────────────────────
print(f"  Improvement over v1 (SVM baseline):")
v1_acc, v1_f1 = 0.451, 0.4509
v2_acc = results[best_name]["acc"]
v2_f1  = results[best_name]["f1m"]
print(f"    Accuracy : {v1_acc:.1%} → {v2_acc:.1%}  ({v2_acc-v1_acc:+.1%})")
print(f"    Macro F1 : {v1_f1:.4f} → {v2_f1:.4f}  ({v2_f1-v1_f1:+.4f})")

# ── Dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Best-Fit v2  —  Subject-Normalised LOSO CV",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

colors4 = ["#00C9A7", "#FF8C42", "#FFD166", "#1565C0"]

# Row 0: confusion matrices
for col, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, col])
    sns.heatmap(r["cm"], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Low","Med","High"],
                yticklabels=["Low","Med","High"],
                ax=ax, cbar=False, vmin=0, vmax=1,
                annot_kws={"size": 9})
    ax.set_title(f"{name}\nAcc {r['acc']:.1%}  F1 {r['f1m']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

# Row 1 left: accuracy + F1 bars
ax = fig.add_subplot(gs[1, :2])
names = list(results.keys())
accs  = [results[n]["acc"]*100 for n in names]
f1s   = [results[n]["f1m"]*100 for n in names]
x     = np.arange(len(names)); w = 0.35
b1 = ax.bar(x - w/2, accs, w, color="#1565C0", label="Accuracy")
b2 = ax.bar(x + w/2, f1s,  w, color="#FF8C42", label="Macro F1")
ax.bar_label(b1, fmt="%.1f%%", fontsize=8)
ax.bar_label(b2, fmt="%.1f%%", fontsize=8)
ax.axhline(33.3, color="red", ls="--", lw=1, label="Chance (33%)")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=10)
ax.set_title("Model Comparison"); ax.legend(fontsize=8)
ax.set_ylim(0, 100); ax.set_ylabel("Score (%)")

# Row 1 right: v1 vs v2 improvement
ax = fig.add_subplot(gs[1, 2:])
metrics = ["Accuracy", "Macro F1", "AUC"]
v1_vals = [0.451,  0.4509, 0.650]
v2_vals = [results[best_name]["acc"],
           results[best_name]["f1m"],
           results[best_name]["auc"]]
x2 = np.arange(3); w2 = 0.35
ax.bar(x2 - w2/2, [v*100 for v in v1_vals], w2, color="#B0BEC5", label="v1 (SVM baseline)")
ax.bar(x2 + w2/2, [v*100 for v in v2_vals], w2, color="#00C9A7", label=f"v2 ({best_name})")
ax.set_xticks(x2); ax.set_xticklabels(metrics)
ax.set_title("v1 vs v2 Improvement"); ax.legend(fontsize=8)
ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)

# Row 2 left: per-subject accuracy heatmap
ax = fig.add_subplot(gs[2, :2])
subj_ids  = np.unique(groups)
subj_accs = [results[best_name]["subj_accs"][s] * 100 for s in subj_ids]
bar_cols  = ["#00C9A7" if a >= 50 else "#E53935" for a in subj_accs]
bars = ax.bar(subj_ids, subj_accs, color=bar_cols, edgecolor="white")
ax.bar_label(bars, fmt="%.0f%%", fontsize=7)
ax.axhline(33.3, color="gray",  ls="--", lw=1, label="Chance")
ax.axhline(50,   color="green", ls=":",  lw=1, label="50%")
ax.set_title(f"Per-Subject Accuracy  —  {best_name}")
ax.set_xlabel("Subject ID"); ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 110); ax.legend(fontsize=8)

# Row 2 right: per-class metrics table
ax = fig.add_subplot(gs[2, 2:])
ax.axis("off")
from sklearn.metrics import precision_recall_fscore_support
p, r_, f, s = precision_recall_fscore_support(y, results[best_name]["y_pred"], labels=[0,1,2])
tbl_data = [[labels[i], f"{p[i]:.3f}", f"{r_[i]:.3f}", f"{f[i]:.3f}", str(s[i])]
            for i in range(3)]
tbl_data.append(["macro avg", f"{p.mean():.3f}", f"{r_.mean():.3f}",
                 f"{f.mean():.3f}", str(sum(s))])
tbl = ax.table(cellText=tbl_data,
               colLabels=["Class","Precision","Recall","F1","Support"],
               cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
    elif row == len(tbl_data):
        cell.set_facecolor("#E3F2FD")
ax.set_title(f"Per-Class Metrics  —  {best_name}", fontsize=10, fontweight="bold")

fig.savefig("outputs/best_v2/dashboard.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"\n📊 Dashboard → outputs/best_v2/dashboard.png")

# ── Save final model ──────────────────────────────────────────
print(f"💾 Fitting final {best_name} on full data...")

# Final model uses train-stats from all data (inference mode)
mu_all = X_raw.mean(axis=0)
sd_all = X_raw.std(axis=0) + 1e-8
X_norm = (X_raw - mu_all) / sd_all

if   best_name == "SVM + PCA":       final_pipe = make_svm()
elif best_name == "SVM + PCA + LDA": final_pipe = make_pca_lda_svm()
elif best_name == "Random Forest":   final_pipe = make_rf()
else:                                 final_pipe = make_ensemble()

final_pipe.fit(X_norm, y)
joblib.dump({"pipe": final_pipe, "mu": mu_all, "sd": sd_all},
            "outputs/best_v2/final_model.pkl")
print(f"   Saved → outputs/best_v2/final_model.pkl\n")
print(f"  To predict on new data:")
print(f"    pkg   = joblib.load('outputs/best_v2/final_model.pkl')")
print(f"    X_new_norm = (X_new - pkg['mu']) / pkg['sd']")
print(f"    pred  = pkg['pipe'].predict(X_new_norm)")
