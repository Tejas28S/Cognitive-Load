# ============================================================
#  eda.py  –  Exploratory Data Analysis
#  Generates plots showing why PCA is needed
#  Run: python eda.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif
import sys, os
sys.path.insert(0, ".")
from config import Config

sns.set_theme(style="whitegrid")
cfg = Config()
os.makedirs("outputs/eda", exist_ok=True)

print("📂 Loading data...")
df      = pd.read_csv(cfg.MERGED_CSV)
meta    = ["subject_id", "task", "Cognitive_Load"]
feats   = [c for c in df.columns if c not in meta]
eeg_f   = [c for c in feats if c.startswith("eeg_")]
hbo_f   = [c for c in feats if c.startswith("hbo_")]
hbr_f   = [c for c in feats if c.startswith("hbr_")]

X = df[feats].values.astype(float)
y = df["Cognitive_Load"].values
Xs = StandardScaler().fit_transform(X)
labels_map = {0:"Low (0-back)", 1:"Medium (2-back)", 2:"High (3-back)"}
colors = ["#00C9A7", "#FF8C42", "#FFD166"]
print(f"   Samples: {len(y)} | Features: {X.shape[1]} | Classes: {np.unique(y)}")

# ── Figure 1: Overview Dashboard ─────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("EDA — Multimodal Cognitive Load Dataset", fontsize=16, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 1a: Class distribution
ax = fig.add_subplot(gs[0, 0])
counts = [( y==i).sum() for i in range(3)]
bars = ax.bar([l.split()[0] for l in labels_map.values()], counts, color=colors)
ax.bar_label(bars, fmt="%d")
ax.set_title("Class Distribution"); ax.set_ylabel("Count")

# 1b: Feature group sizes
ax = fig.add_subplot(gs[0, 1])
ax.pie([len(eeg_f), len(hbo_f), len(hbr_f)],
       labels=["EEG\n(180)", "HbO\n(468)", "HbR\n(468)"],
       colors=["#1565C0","#00C9A7","#028090"], autopct="%1.0f%%", startangle=90)
ax.set_title("Feature Breakdown\n(1,116 total)")

# 1c: Sample-to-feature ratio — THE CURSE
ax = fig.add_subplot(gs[0, 2])
datasets = ["Ideal DL\n(10:1)", "This project\n(0.58:1)", "After PCA\n(~3:1)"]
ratios   = [10, 654/1116, 654/200]
bar_colors = ["#4CAF50", "#E53935", "#FF8C42"]
b = ax.bar(datasets, ratios, color=bar_colors)
ax.bar_label(b, fmt="%.2f")
ax.axhline(1, color="red", ls="--", lw=1, label="Minimum (1:1)")
ax.set_title("Sample:Feature Ratio\n← Curse of Dimensionality"); ax.legend(fontsize=8)

# 1d: Per-subject sample count
ax = fig.add_subplot(gs[0, 3])
subj_counts = df["subject_id"].value_counts().sort_index()
ax.bar(subj_counts.index, subj_counts.values, color="#1B2D45", edgecolor="#00C9A7")
ax.set_title("Samples per Subject"); ax.set_xlabel("Subject ID"); ax.set_ylabel("Trials")

# 1e: Mean feature value per class (first 30 EEG features)
ax = fig.add_subplot(gs[1, :2])
for i, (cls, name) in enumerate(labels_map.items()):
    means = df[eeg_f[:30]][y==cls].mean().values
    ax.plot(means, label=name, color=colors[i], alpha=0.8)
ax.set_title("Mean EEG Feature Values per Class (first 30 features)\n"
             "→ Overlapping means = high dimensionality hurts separability")
ax.legend(fontsize=8); ax.set_xlabel("Feature index"); ax.set_ylabel("Mean value")

# 1f: Mean HbO feature values per class
ax = fig.add_subplot(gs[1, 2:])
for i, (cls, name) in enumerate(labels_map.items()):
    means = df[hbo_f[:40]][y==cls].mean().values
    ax.plot(means, label=name, color=colors[i], alpha=0.8)
ax.set_title("Mean HbO Feature Values per Class (first 40 features)\n"
             "→ More separation visible in fNIRS vs EEG")
ax.legend(fontsize=8); ax.set_xlabel("Feature index"); ax.set_ylabel("Mean value")

# 1g: ANOVA F-scores (top 50 features)
ax = fig.add_subplot(gs[2, :2])
f_vals, p_vals = f_classif(Xs, y)
sig = (p_vals < 0.05).sum()
top50_idx = np.argsort(f_vals)[::-1][:50]
ax.bar(range(50), f_vals[top50_idx],
       color=["#00C9A7" if feats[i].startswith("hbo_") or feats[i].startswith("hbr_")
              else "#FF8C42" for i in top50_idx])
ax.set_title(f"Top 50 Features by ANOVA F-Score\n"
             f"Significant (p<0.05): {sig}/{len(feats)} = {sig/len(feats):.1%} of features")
ax.set_xlabel("Feature rank"); ax.set_ylabel("F-score")
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#00C9A7",label="fNIRS"),
                   Patch(color="#FF8C42",label="EEG")], fontsize=8)

# 1h: Variance explained by top features
ax = fig.add_subplot(gs[2, 2:])
pca_tmp = PCA().fit(Xs)
cumvar  = np.cumsum(pca_tmp.explained_variance_ratio_) * 100
ax.plot(cumvar, color="#00C9A7", lw=2)
for thr, style in [(80,"--"),(90,"-."),(95,":")]:
    n = np.argmax(cumvar >= thr) + 1
    ax.axvline(n, color="gray", ls=style, lw=1)
    ax.axhline(thr, color="gray", ls=style, lw=1)
    ax.text(n+5, thr-4, f"{thr}% @ {n} PCs", fontsize=8, color="gray")
ax.set_title("PCA: Cumulative Variance Explained\n← Choose n_components here")
ax.set_xlabel("Number of Principal Components"); ax.set_ylabel("Cumulative Variance (%)")
ax.set_xlim(0, 300)

fig.savefig("outputs/eda/eda_overview.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print("✅ outputs/eda/eda_overview.png")

# ── Figure 2: t-SNE before PCA ───────────────────────────────
print("🔄 Running t-SNE (before PCA)...")
Xpca50 = PCA(n_components=50, random_state=42).fit_transform(Xs)  # pre-reduce for speed
tsne_before = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(Xpca50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("t-SNE Visualization: Class Separability", fontsize=14, fontweight="bold")

for ax, (title, emb) in zip(axes, [
    ("Before PCA (1,116 features → 50 PC pre-reduction)", tsne_before),
]):
    for i, (cls, name) in enumerate(labels_map.items()):
        mask = y == cls
        ax.scatter(emb[mask,0], emb[mask,1], c=colors[i], label=name,
                   alpha=0.6, s=30, edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

# Right: per-subject coloring
sc = axes[1].scatter(tsne_before[:,0], tsne_before[:,1],
                     c=df["subject_id"].values, cmap="tab20", alpha=0.6, s=30)
axes[1].set_title("Same t-SNE — colored by Subject ID\n→ Subject variability dominates class signal")
axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
plt.colorbar(sc, ax=axes[1], label="Subject ID")

fig.tight_layout()
fig.savefig("outputs/eda/tsne_separability.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print("✅ outputs/eda/tsne_separability.png")

# ── Figure 3: Feature correlation heatmap ────────────────────
print("🔄 Computing feature correlations...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Feature Correlation Heatmap (random sample of 40 features per modality)",
             fontsize=13, fontweight="bold")

np.random.seed(42)
for ax, (name, cols) in zip(axes, [
    ("EEG (40 random feats)",  np.random.choice(eeg_f, 40, replace=False)),
    ("HbO (40 random feats)",  np.random.choice(hbo_f, 40, replace=False)),
    ("HbR (40 random feats)",  np.random.choice(hbr_f, 40, replace=False)),
]):
    corr = df[cols].corr().values
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, square=True,
                cbar_kws={"shrink":0.7})
    high_corr = (np.abs(corr - np.eye(40)) > 0.8).sum() // 2
    ax.set_title(f"{name}\nHighly correlated pairs (|r|>0.8): {high_corr}\n→ Redundant features that PCA removes")

fig.tight_layout()
fig.savefig("outputs/eda/correlation_heatmap.png", bbox_inches="tight", dpi=130)
plt.close(fig)
print("✅ outputs/eda/correlation_heatmap.png")

# ── Summary stats ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  EDA SUMMARY")
print(f"{'─'*55}")
print(f"  Samples           : {len(y)}")
print(f"  Total features    : {X.shape[1]}")
print(f"  Sample:feat ratio : {len(y)/X.shape[1]:.2f}  (ideal ≥ 10)")
print(f"  Significant feats : {sig}/{X.shape[1]} ({sig/X.shape[1]:.1%})")

cumvar_all = np.cumsum(pca_tmp.explained_variance_ratio_) * 100
for thr in [80, 90, 95]:
    n = np.argmax(cumvar_all >= thr) + 1
    ratio = len(y) / n
    print(f"  PCA for {thr}% var    : {n:3d} components  "
          f"(sample:feat = {ratio:.1f}{'  ✅' if ratio >= 3 else '  ⚠️'})")
print(f"{'='*55}")
print(f"\n  Recommended: PCA with n_components=100-150")
print(f"  Run next   : python train_with_pca.py")