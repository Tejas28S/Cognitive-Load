# ============================================================
#  train_best.py  –  Best-fit pipeline for 654-sample
#                    multimodal EEG+fNIRS cognitive load data
#
#  Strategy:
#   • Classical ML (SVM / RF / GBM) — right-sized for N=654
#   • Leave-One-Subject-Out CV      — honest cross-subject eval
#   • Balanced class weights        — fixes class collapse
#   • Voting ensemble               — squeezes last few % out
#   • Full reporting + dashboard
#
#  Run: python train_best.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, time
warnings.filterwarnings("ignore")

from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics         import (accuracy_score, f1_score,
                                     roc_auc_score, classification_report,
                                     confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
import sys
sys.path.insert(0, ".")
from config import Config

sns.set_theme(style="whitegrid")
os.makedirs("outputs/best", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────
cfg = Config()
print(f"\n{'='*60}")
print(f"  Best-Fit Cognitive Load Classifier")
print(f"{'='*60}")
print(f"\n📂 Loading {cfg.MERGED_CSV} ...")

df    = pd.read_csv(cfg.MERGED_CSV)
meta  = ["subject_id", "task", "Cognitive_Load"]
feats = [c for c in df.columns if c not in meta]

X      = df[feats].values.astype(float)
y      = df["Cognitive_Load"].values
groups = df["subject_id"].values          # for LOSO

labels     = ["Low (0-back)", "Medium (2-back)", "High (3-back)"]
n_subjects = len(np.unique(groups))

print(f"   Samples   : {len(y)}")
print(f"   Features  : {X.shape[1]}")
print(f"   Subjects  : {n_subjects}")
print(f"   Classes   : {np.unique(y)}  →  {dict(zip(*np.unique(y, return_counts=True)))}")

# ── Build pipelines ───────────────────────────────────────────
# PCA n_components chosen from EDA: 95% variance ≈ 120 components
# Keep it at 100 to maximise sample:feature ratio after LOSO split
PCA_N = 100

def make_pipeline(clf):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=PCA_N, random_state=42)),
        ("clf",    clf),
    ])

svm = make_pipeline(
    SVC(kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True, random_state=42)
)

rf = make_pipeline(
    RandomForestClassifier(
        n_estimators=500, max_depth=None,
        min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=42)
)

gbm = make_pipeline(
    GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        random_state=42)
)

# Soft-voting ensemble (best of all worlds)
ensemble = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=PCA_N, random_state=42)),
    ("clf", VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        class_weight="balanced", probability=True, random_state=42)),
            ("rf",  RandomForestClassifier(
                        n_estimators=500, min_samples_leaf=2,
                        class_weight="balanced", n_jobs=-1, random_state=42)),
            ("gbm", GradientBoostingClassifier(
                        n_estimators=300, learning_rate=0.05,
                        max_depth=4, subsample=0.8, random_state=42)),
        ],
        voting="soft", n_jobs=-1
    )),
])

models = {
    "SVM (RBF)":        svm,
    "Random Forest":    rf,
    "Gradient Boost":   gbm,
    "Soft Ensemble":    ensemble,
}

# ── Leave-One-Subject-Out CV ──────────────────────────────────
logo    = LeaveOneGroupOut()
results = {}

print(f"\n{'─'*60}")
print(f"  Leave-One-Subject-Out Cross-Validation  ({n_subjects} folds)")
print(f"{'─'*60}\n")

for name, pipe in models.items():
    t0 = time.time()
    print(f"  ▶  {name} ...", end="", flush=True)

    # cross_val_predict gives per-sample predictions across all folds
    y_pred = cross_val_predict(pipe, X, y, groups=groups,
                               cv=logo, n_jobs=-1)

    # For AUC we need probabilities — re-run with method="predict_proba"
    try:
        y_prob = cross_val_predict(pipe, X, y, groups=groups,
                                   cv=logo, n_jobs=-1,
                                   method="predict_proba")
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    acc  = accuracy_score(y, y_pred)
    f1m  = f1_score(y, y_pred, average="macro")
    f1w  = f1_score(y, y_pred, average="weighted")
    cm   = confusion_matrix(y, y_pred, normalize="true")
    cr   = classification_report(y, y_pred, target_names=labels)

    elapsed = time.time() - t0
    print(f"  done  [{elapsed:.0f}s]")
    print(f"        Acc={acc:.1%}  Macro-F1={f1m:.3f}  AUC={auc:.3f}")

    results[name] = dict(acc=acc, f1m=f1m, f1w=f1w, auc=auc,
                         cm=cm, cr=cr, y_pred=y_pred,
                         y_prob=y_prob if not np.isnan(auc) else None)

# ── Console summary ───────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {'Model':<22}  {'Acc':>7}  {'Macro F1':>9}  {'AUC':>7}")
print(f"{'─'*60}")
best_name = max(results, key=lambda k: results[k]["f1m"])
for name, r in results.items():
    flag = " ◀ best" if name == best_name else ""
    print(f"  {name:<22}  {r['acc']:>6.1%}  {r['f1m']:>9.4f}  {r['auc']:>7.4f}{flag}")
print(f"{'='*60}\n")

print(f"  Per-class report — {best_name}:")
print(results[best_name]["cr"])

# ── Dashboard ─────────────────────────────────────────────────
colors4 = ["#00C9A7", "#FF8C42", "#FFD166", "#1565C0"]
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Best-Fit Classifier Results  –  Leave-One-Subject-Out CV",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# Row 0: confusion matrices (4 models)
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

# Row 1 left: bar comparison
ax = fig.add_subplot(gs[1, :2])
names = list(results.keys())
accs  = [results[n]["acc"]*100 for n in names]
f1s   = [results[n]["f1m"]*100 for n in names]
x     = np.arange(len(names))
w     = 0.35
b1 = ax.bar(x - w/2, accs, w, color="#1565C0", label="Accuracy")
b2 = ax.bar(x + w/2, f1s,  w, color="#FF8C42", label="Macro F1")
ax.bar_label(b1, fmt="%.1f%%", fontsize=8)
ax.bar_label(b2, fmt="%.1f%%", fontsize=8)
ax.axhline(33.3, color="red", ls="--", lw=1, label="Chance (33.3%)")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_title("Accuracy & Macro F1 — All Models"); ax.legend(fontsize=8)
ax.set_ylim(0, 100); ax.set_ylabel("Score (%)")

# Row 1 right: AUC bar
ax = fig.add_subplot(gs[1, 2:])
aucs = [results[n]["auc"] for n in names]
bars = ax.bar(names, aucs, color=colors4, edgecolor="white")
ax.bar_label(bars, fmt="%.3f", fontsize=9)
ax.axhline(0.5, color="red", ls="--", lw=1, label="Chance AUC (0.5)")
ax.set_title("ROC-AUC (macro OvR) — All Models"); ax.legend(fontsize=8)
ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUC")
ax.set_xticklabels(names, fontsize=9)

# Row 2: per-class F1 heatmap for best model + per-subject accuracy
ax = fig.add_subplot(gs[2, :2])
# Per-subject accuracy for best model
y_pred_best = results[best_name]["y_pred"]
subj_ids    = np.unique(groups)
subj_accs   = [accuracy_score(y[groups==s], y_pred_best[groups==s]) for s in subj_ids]
bar_colors  = ["#00C9A7" if a >= 0.5 else "#E53935" for a in subj_accs]
bars = ax.bar(subj_ids, [a*100 for a in subj_accs], color=bar_colors, edgecolor="white")
ax.bar_label(bars, fmt="%.0f%%", fontsize=7)
ax.axhline(33.3, color="gray", ls="--", lw=1, label="Chance")
ax.axhline(50,   color="green", ls=":",  lw=1, label="50%")
ax.set_title(f"Per-Subject Accuracy  —  {best_name}\n"
             f"(green ≥ 50%,  red < 50%)")
ax.set_xlabel("Subject ID"); ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105); ax.legend(fontsize=8)

# Row 2 right: per-class precision / recall / f1 table
ax = fig.add_subplot(gs[2, 2:])
ax.axis("off")
from sklearn.metrics import precision_recall_fscore_support
p, r, f, s = precision_recall_fscore_support(
    y, y_pred_best, labels=[0,1,2])
table_data = [[labels[i],
               f"{p[i]:.3f}", f"{r[i]:.3f}", f"{f[i]:.3f}", str(s[i])]
              for i in range(3)]
table_data.append(["macro avg",
                   f"{p.mean():.3f}", f"{r.mean():.3f}",
                   f"{f.mean():.3f}", str(sum(s))])
tbl = ax.table(
    cellText=table_data,
    colLabels=["Class", "Precision", "Recall", "F1", "Support"],
    cellLoc="center", loc="center",
    bbox=[0, 0.1, 1, 0.85]
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
    elif row == len(table_data):
        cell.set_facecolor("#E3F2FD")
ax.set_title(f"Per-Class Metrics  —  {best_name}", fontsize=10, fontweight="bold")

out_path = "outputs/best/dashboard.png"
fig.savefig(out_path, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"📊 Dashboard saved → {out_path}")

# ── Train final model on ALL data & save ──────────────────────
import joblib
print(f"\n💾 Fitting final {best_name} on full dataset and saving...")
best_pipe = models[best_name]
best_pipe.fit(X, y)
joblib.dump(best_pipe, "outputs/best/final_model.pkl")
print(f"   Saved → outputs/best/final_model.pkl")

print(f"\n{'='*60}")
print(f"  ✅  Done.  Best model: {best_name}")
print(f"       Acc={results[best_name]['acc']:.1%}  "
      f"Macro-F1={results[best_name]['f1m']:.3f}  "
      f"AUC={results[best_name]['auc']:.3f}")
print(f"{'='*60}\n")
print(f"  Output files:")
print(f"    outputs/best/dashboard.png")
print(f"    outputs/best/final_model.pkl")
print(f"\n  To load and predict later:")
print(f"    import joblib, numpy as np")
print(f"    model = joblib.load('outputs/best/final_model.pkl')")
print(f"    pred  = model.predict(X_new)")
