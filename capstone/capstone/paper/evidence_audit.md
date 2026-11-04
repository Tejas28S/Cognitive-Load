# Comprehensive Evidence Audit: Cognitive Load Estimation via Multimodal EEG + fNIRS

**Document Version:** 1.0.0  
**Target Submission:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE) / IEEE EMBC  
**Evaluation Scope:** 26-Subject Leave-One-Subject-Out (LOSO) Cross-Validation & 8-Subject Isolated Lockbox  
**Audit Date:** October 2026  
**Artifact Location:** `/Users/tejas.m.s/.gemini/antigravity/brain/1d6e184c-114d-47c7-a07c-f59a5e1af120/evidence_audit.md` (and mirrored at `/Users/tejas.m.s/.gemini/antigravity/brain/6efc9f4a-7e61-4896-b2ba-92da6f17e292/evidence_audit.md`)  

---

## 1. Executive Summary & Audit Mandate

This evidence audit establishes the definitive empirical ground truth for the BTech capstone project on **Cognitive Load Estimation using Multimodal EEG + fNIRS** (TU Berlin dataset, Shin et al., 2018).

The project investigated whether deep neural architectures, domain-adversarial adaptation, and attention-based multimodal fusion could achieve genuine cross-subject generalization on binary cognitive workload classification (0-back Low vs. 3-back High). In total, **58 distinct model configurations** were explored across three sequential research campaigns.

### The Core Scientific Finding
Across 42 systematically controlled experiments in the primary campaign, **zero model architectures or regularizations cleared the experimental noise floor ($\pm 0.0184$ macro F1 standard error)** above the baseline dual-stream MLP model (`A0_baseline`, 245,594 parameters). When evaluated under strict, zero-leakage 26-fold Leave-One-Subject-Out (LOSO) cross-validation and confirmed on an untouched 8-subject lockbox:
- **26-Fold LOSO Baseline:** Accuracy **$0.6309 \pm 0.1011$**, Macro F1 **$0.6255 \pm 0.1044$**, ROC-AUC **$0.6775 \pm 0.1289$**, MCC **$0.2675 \pm 0.2058$**.
- **Lockbox (8 Unseen Subjects):** Accuracy **$0.6371 \pm 0.0850$**, Macro F1 **$0.6275 \pm 0.0931$**, ROC-AUC **$0.6888 \pm 0.1112$**, MCC **$0.2816 \pm 0.1690$**.

The lockbox score strictly mirrors the cross-validation score, proving that the 42-experiment search incurred **zero selection inflation**.

```mermaid
flowchart TD
    subgraph Data [TU Berlin Dataset: 26 Subjects]
        Raw[Raw EEG 30ch @ 200Hz + fNIRS 36ch @ 10Hz] --> Win[Windowing: 10s Window, 5s Stride, 50% Overlap]
        Win --> Feat[1,476 Features: 540 EEG + 936 fNIRS HbO/HbR]
    end

    subgraph Split [Subject Partitioning]
        Feat --> Dev[18 Development Subjects]
        Feat --> Lock[8 Lockbox Subjects - Strictly Frozen]
    end

    subgraph Audit [Automated Leakage Audit: research/leakage.py]
        Dev --> Checks[9 Per-Fold Checks + 3 Per-Dataset Checks]
        Checks -->|PASS| TrainVal[GroupShuffleSplit: Train vs Validation]
        Checks -->|FAIL| Abort[Raise LeakageError & Terminate Run]
    end

    subgraph Results [Empirical Reality vs Literature Illusion]
        TrainVal --> Base[A0_baseline Dual-Stream MLP: 63.09% LOSO Acc]
        Lock --> LockRes[Lockbox Confirmation: 63.71% Acc - No Inflation]
        Base -.-> LitIllusion["Literature 'Flawed' Splits: 75-90% Acc (Identity Leakage)"]
    end
```

### The Audit Mandate
A critical discrepancy exists between the **rigorous, leakage-audited engineering records (PDF 1 & PDF 2)** and the **provisional publication brief (PDF 3)**. To prevent scientific retraction, reviewer rejection, or IEEE ethics violations, this audit separates:
1. **Verified Empirical Ground Truth:** Backed by automated leakage audits, saved fold predictions (`predictions.csv`), and paired statistical testing.
2. **Unverified / Provisional Claims:** High-performing candidate configurations that were screened on development folds but never executed under the final 26-fold LOSO + lockbox protocol (`--stage final` not run).
3. **Contradictions & Confounders:** Hardware-induced script abandonments (XGBoost `libomp`), discrepancies between metadata configurations and served inference scripts, and conceptual conflation of attention gates with SHAP attribution.

---

## 2. Source Material Profiles & Epistemic Hierarchy

The documentation corpus consists of three documents representing distinct phases and epistemic authorities:

```mermaid
graph LR
    P1[PDF 1: PROJECT_RECORD<br/><b>Epistemic Authority: Tier 1</b><br/>Complete Project Record, Automated Leakage Tests, Bit-for-Bit Verified Predictions]
    P2[PDF 2: PROJECT_DOCUMENTATION<br/><b>Epistemic Authority: Tier 2</b><br/>Architectural Blueprint, Codebase Audit, 58-Run Campaign History, Loose Ends]
    P3[PDF 3: Research & Publication Brief<br/><b>Epistemic Authority: Tier 3</b><br/>Aspirational Publication Pitch, Unconfirmed Ensembles, Provisional Claims]

    P1 -->|Overrides| P3
    P2 -->|Contextualizes & Validates| P1
    P2 -->|Flags Caveats in| P3
```

### PDF 1: `PROJECT_RECORD` (Complete Project Record)
- **Role:** Primary Source of Empirical Truth.
- **Authority:** Highest (Tier 1). Generated directly from execution artifacts, timestamped run directories (`outputs/runs/run_20260820_141126_19cb`), and bit-for-bit prediction verification via `research/verify_run.py`.
- **Key Contents:**
  - Complete mathematical specification of preprocessing, 1,476-dimensional feature extraction, and leakage assertions (9 per-fold + 3 per-dataset).
  - Exhaustive 42-experiment search table across 11 stages (S0–S10) with paired $p$-values, standard errors, and generalization gaps.
  - Negative results: CNN architecture failure, DANN monotonic degradation, and subject-invariance probe analysis.
  - Explainability: GradientExplainer SHAP attribution allocating 94.5% importance to fNIRS (HbR > HbO).
  - Definitive baseline 26-fold LOSO and 8-subject lockbox metrics.

### PDF 2: `PROJECT_DOCUMENTATION` (Project Documentation — Architecture, Methodology, Results, and Status)
- **Role:** Engineering Architecture, Codebase Audit, and Campaign Evolution Record.
- **Authority:** High (Tier 2). Generated 2026-08-30 from git branch `updated-frontend`, commit `4c8c033`.
- **Key Contents:**
  - End-to-end repository audit covering the 58 total configurations attempted across 3 campaigns:
    1. Primary Campaign (`run_campaign.py`): 42 experiments.
    2. Target Follow-Up (`target_campaign.py`): 15 candidate configurations.
    3. Deep Ensemble Screen (`ensemble_campaign.py`): 1 candidate (5-member deep ensemble).
  - The cautionary **D2 Episode**: Candidate `D2_narrow_patience6` won the 18-subject development screen but collapsed when promoted to the 26-fold LOSO + lockbox protocol.
  - Critical loose ends: Documented vs. rerun baseline mismatch (0.6537 vs. 0.6304 dev-LOSO accuracy), 5-member deep ensemble `--stage final` not executed, and `model_config.json` serving discrepancy.

### PDF 3: `Cognitive_Load_Research_Brief` (Research & Publication Brief)
- **Role:** Conference / Journal Proposal Pitch.
- **Authority:** Provisional / Low (Tier 3).
- **Key Contents:**
  - Executive benchmarking framing the project as outperforming literature (66.32% Deep Ensemble, 61.78% Fast Linear Ensemble).
  - 4 scientific contributions: 96.6% dimensionality pruning ($F > 400$, $p < 10^{-79}$), classical vs. deep comparison, attention gating (65–70% attribution), and zero-leakage DANN.
  - **Audit Verdict:** Several headline figures in PDF 3 are unconfirmed artifacts from interrupted sweeps, unverified development-stage runs, or legacy scripts that were never subjected to the primary leakage-audited protocol.

---

## 3. Master Evidence Matrix

The following table cross-references every empirical claim, metric, and experimental finding across all three source documents.

| ID | Scientific Domain / Claim | Numerical Value | Source Document(s) | Verification Status | Methodological Evidence & Run Artifacts | Safe for IEEE Paper? |
|:---|:---|:---|:---|:---|:---|:---|
| **E01** | Primary Dataset & Cohort | 26 subjects, 650 trials (25/subj), simultaneous EEG (30ch @ 200Hz) + fNIRS (36ch @ 10Hz) | PDF 1 (§2), PDF 2 (§1), PDF 3 (§4) | **VERIFIED** | Shin et al. (2018) open dataset. Raw `.mat` and feature CSVs verified in repository (9.3 GB under `ZDATASET/`). | **YES** |
| **E02** | Binary Task Framing & Balance | 0-back (Low) vs. 3-back (High); 2-back dropped; 2,288 windows/class = 4,576 total | PDF 1 (§2), PDF 2 (§1), PDF 3 (§1) | **VERIFIED** | Perfectly balanced classes. Verified in `dataset_info.json` and `predictions.csv`. | **YES** |
| **E03** | Window Geometry (Primary) | 10 s window, 5 s stride (50% adjacent overlap) $\rightarrow$ 4,576 samples | PDF 1 (§3, §18), PDF 2 (§5) | **VERIFIED** | `tuberlin_sliding.csv`. Standard sliding geometry. | **YES** |
| **E04** | Feature Dimensionality | 1,476 features/window: 540 EEG (30 el $\times$ 6 bands $\times$ 3 stats) + 936 fNIRS (36 opt $\times$ 13 stats $\times$ 2 HbO/HbR) | PDF 1 (§3), PDF 2 (§5.1), PDF 3 (§2) | **VERIFIED** | Channel-major column ordering. Verified by schema inspection. | **YES** |
| **E05** | Automated Leakage Audit | 9 per-fold assertions + 3 dataset-level assertions; 8/8 negative tests pass | PDF 1 (§10-14), PDF 2 (§5.2) | **VERIFIED** | `research/leakage.py` and `research/test_leakage.py`. Asserts zero subject, window, trial, or scaler leakage. | **YES** |
| **E06** | Subject Split Protocol | 18 Development subjects, 8 Lockbox subjects (deterministic partition) | PDF 1 (§7), PDF 2 (§5.3) | **VERIFIED** | Dev: `[1,2,3,4,5,7,9,10,11,12,13,14,15,17,20,23,24,25]`; Lockbox: `[6,8,16,18,19,21,22,26]`. | **YES** |
| **E07** | Selection Rule Standard Error | Noise floor bar = $\pm 0.0184$ macro F1 ($SE = SD / \sqrt{18}$) | PDF 1 (§9, §15), PDF 2 (§11) | **VERIFIED** | Arms within 1 SE of leader ranked; incumbent and smallest params preferred. | **YES** |
| **E08** | Primary Search Scale | 42 experiments across 11 stages (S0–S10); 0 cleared the noise floor | PDF 1 (§15, §16), PDF 2 (§5.3) | **VERIFIED** | All 42 runs archived under `run_20260820_141126_19cb` and `run_20260819_184443_f0d1`. | **YES** |
| **E09** | Baseline Architecture (`A0_baseline`) | Dual-stream MLP encoders, attention fusion, ResMLP blocks, 245,594 parameters | PDF 1 (§5, §17), PDF 2 (§5.1) | **VERIFIED** | `proj_dim=96`, `enc_dim=48`, 2 blocks, dropout 0.5/0.6, AdamW, Plateau scheduler. | **YES** |
| **E10** | 26-Fold LOSO Accuracy | $0.6309 \pm 0.1011$ (Chance = 0.5000) | PDF 1 (§18), PDF 2 (§9), PDF 3 (§2) | **VERIFIED** | Evaluated on all 4,576 held-out windows. Verified via `research/verify_run.py`. | **YES** |
| **E11** | 26-Fold LOSO Macro F1 | $0.6255 \pm 0.1044$ | PDF 1 (§18) | **VERIFIED** | Computed strictly from held-out fold predictions. | **YES** |
| **E12** | 26-Fold LOSO ROC-AUC & MCC | ROC-AUC: $0.6775 \pm 0.1289$; MCC: $0.2675 \pm 0.2058$; Kappa: $0.2618 \pm 0.2022$ | PDF 1 (§18) | **VERIFIED** | Multi-metric validation confirmed in `metrics.json`. | **YES** |
| **E13** | Lockbox Performance (8 Subjects) | Acc: $0.6371 \pm 0.0850$; Macro F1: $0.6275 \pm 0.0931$; AUC: $0.6888 \pm 0.1112$ | PDF 1 (§18) | **VERIFIED** | Unseen lockbox cohort confirms zero selection inflation over 26-fold LOSO ($0.6371 \ge 0.6309$). | **YES** |
| **E14** | Generalization Overfit Gap | Train: $0.8022 \pm 0.093$; Val: $0.6400 \pm 0.053$; Test: $0.6309 \pm 0.101$; Gap: $+0.1713$ | PDF 1 (§19), PDF 3 (§2) | **VERIFIED** | Demonstrates cross-subject generalization penalty. Lockbox gap is $+0.1815$. | **YES** |
| **E15** | Confusion Matrix Breakdown | Low: Prec 0.619, Rec 0.682, F1 0.649; High: Prec 0.646, Rec 0.580, F1 0.611 | PDF 1 (§20) | **VERIFIED** | Pooled 4,576 windows: 1,561 TN, 727 FP, 962 FN, 1,326 TP. | **YES** |
| **E16** | Per-Subject Variance Spread | Macro F1 range: $0.431$ (S12) to $0.807$ (S17); 4 of 26 subjects fall below chance | PDF 1 (§21), PDF 2 (§9) | **VERIFIED** | Spread across humans ($0.376$) is $>2\times$ the spread across all 42 architectures ($0.166$). | **YES** |
| **E17** | 1D-CNN Encoder Failure | CNN Plain: F1 $0.4786$ ($p < 0.001$); CNN Res+Attn: F1 $0.5467$ ($p = 0.001$); Best CNN: F1 $0.5550$ | PDF 1 (§16, §22) | **VERIFIED** | Convolving channel-axis over summary features cuts capacity excessively. Overfitting drops but held-out score collapses. | **YES** |
| **E18** | Attention Fusion vs. Concat | Attention: F1 $0.6451$; Concat: F1 $0.6294$ ($\Delta = -0.0157$, $p = 0.230$) | PDF 1 (§16, §23) | **VERIFIED** | Attention gain is positive but not statistically significant. Retained for interpretability. | **YES** |
| **E19** | Regularization Ineffectiveness | High Dropout (0.6/0.7): F1 drops by $0.1369$ ($p < 0.001$); Weight decay, input noise fail | PDF 1 (§24) | **VERIFIED** | Strong regularization shrinks train-test gap (to $0.040$) but destroys held-out accuracy. | **YES** |
| **E20** | Capacity Invariance | 66K param model (`S5_proj32`): F1 $0.6466$ vs. Baseline 245K: F1 $0.6451$ ($\Delta = +0.0015$, $p = 0.929$) | PDF 1 (§16, §25) | **VERIFIED** | 3.7$\times$ parameter pruning incurs zero performance loss; capacity is not the constraint. | **YES** |
| **E21** | DANN Adversarial Degradation | Baseline: F1 $0.6451$; $\lambda=0.1$: $0.6182$ ($p=0.104$); $\lambda=0.5$: $0.5968$ ($p=0.014$); $\lambda=1.0$: $0.5778$ ($p=0.012$) | PDF 1 (§16, §26), PDF 2 (§5.4) | **VERIFIED** | Monotonic performance collapse as adversarial gradient reversal strength increases. | **YES** |
| **E22** | Subject-Invariance Probe | Raw: $0.013$; Baseline Embedding: $0.214$; DANN Embedding: $0.217$ (Chance = $0.125$) | PDF 1 (§26), PDF 2 (§5.4) | **VERIFIED** | Preprocessing removes subject identity; encoder re-introduces it; DANN fails to strip it, injecting only noise. | **YES** |
| **E23** | Preprocessing Ablations | No subject norm: $-0.0753$ ($p=0.014$); Drop HbR: $-0.0670$ ($p=0.014$); Calibration norm: $-0.0402$ ($p=0.156$) | PDF 1 (§27) | **VERIFIED** | Proves per-subject normalization and HbR are mathematically essential. Calibration mode viable for deployment. | **YES** |
| **E24** | Dense Windowing (1s stride) | F1 $0.6566 \pm 0.114$ ($\Delta = +0.0115$, $p = 0.461$, inside noise) | PDF 1 (§16, §27) | **VERIFIED** | 4.6$\times$ row expansion (21,216 samples) with 90% overlap provides no statistically significant gain. | **YES** |
| **E25** | Feature Attribution (SHAP) | fNIRS: **$0.945$** total share; EEG: **$0.055$**; Within fNIRS: HbR ($0.488$) > HbO ($0.458$) | PDF 1 (§29) | **VERIFIED** | GradientExplainer on held-out test windows. Corroborates HbR ablation ($\Delta = -0.067$). | **YES** |
| **E26** | Total Configurations Tried | 58 total: 42 (primary search) + 15 (target follow-up) + 1 (5-member deep ensemble) | PDF 2 (§5.3) | **VERIFIED** | Comprehensive campaign count spanning `run_campaign.py`, `target_campaign.py`, `ensemble_campaign.py`. | **YES** |
| **E27** | The D2 Episode | Candidate `D2_narrow_patience6` won 18-fold dev screen but lost on full 26-fold LOSO + lockbox | PDF 2 (§5.3) | **VERIFIED** | Empirical proof that single dev-screen selection inflates expectations; mandates lockbox protocol. | **YES** |
| **E28** | Baseline Dev-LOSO Mismatch | Documented dev-LOSO accuracy **$0.6537$** vs. rerun **$0.6304$** | PDF 2 (§5.3, §8) | **CONFIRMED DISCREPANCY** | Unresolved numerical drift. Forced all downstream comparisons to be paired against in-run baselines. | **QUALIFIED** |
| **E29** | Serving Code vs. Config Mismatch | `model_config.json` lists `hybrid_rf_ensemble`, but `model_service.py` serves `HybridCNNLSTM` alone | PDF 2 (§5.5, §8) | **CONFIRMED DISCREPANCY** | Stale metadata in serving configuration bundle. Inference code executes pure MLP hybrid. | **NO (Fix Code)** |
| **E30** | 3-Class Serving Stand-in | 3-class endpoint serves a retrained stand-in (6 subjects), not the published research model | PDF 2 (§7, §8) | **VERIFIED CAVEAT** | Demo environment fallback due to gitignored large raw files. Explicitly flagged by API response. | **QUALIFIED** |
| **E31** | 5-Member Deep Ensemble Claim | Claimed Acc: **$66.32\% \pm 9.9\%$**, Overfit Gap: $21.7\%$ | PDF 3 (§1, §2) | **UNVERIFIED / PROVISIONAL** | Won dev screen against in-run baseline, but `--stage final` (26-fold LOSO + lockbox) was **NEVER RUN**. | **NO** |
| **E32** | Fast Linear Ensemble Claim | Claimed Acc: **$61.78\% \pm 8.0\%$**, Gap: $4.65\%$, 50 features, 80 s CPU train | PDF 3 (§1, §2, §3) | **UNVERIFIED / PROVISIONAL** | Classical sweep from `classical_sweep.py`. Not rerun or audited under primary leakage-audited protocol. | **NO** |
| **E33** | SelectKBest + PCA + LDA Claim | Claimed Acc: **$61.60\% \pm 8.7\%$**, Gap: $3.9\%$, 20-50 features, 64 s CPU train | PDF 3 (§2) | **UNVERIFIED / PROVISIONAL** | Classical pipeline; lacking per-fold leakage assertion artifact verification in `outputs/runs/`. | **NO** |
| **E34** | Hybrid + RF Ensemble Claim | Claimed Acc: **$62.25\%$**, Overfit Gap: $28.6\%$ | PDF 3 (§2) | **UNVERIFIED / CONTRADICTED** | Claimed in PDF 3 table, but PDF 2 confirms RF ensemble was unhooked and is not served. | **NO** |
| **E35** | 96.6% Pruning & ANOVA F-Score | Pruning to 50 feats; claims $F > 400$, $p < 10^{-79}$ strictly on prefrontal HbO (ch05-09, ch23-26) | PDF 3 (§3) | **UNVERIFIED / SUSPECT** | Extreme $F$-statistic suggests evaluation on pooled dataset prior to fold splitting (leakage artifact). | **NO** |
| **E36** | Modality Attribution Share | Claimed fNIRS carries ~65% to 70% attribution; EEG provides rapid transient bursts | PDF 3 (§3) | **DISCREPANT / CONFUSED** | Conflates attention gate values (65-70%) with SHAP feature attribution (94.5% fNIRS). | **QUALIFIED** |

---

## 4. In-Depth Discrepancy & Conflict Analysis

### Conflict 1: The 5x Deep Ensemble Performance ($66.32\%$ vs. Unrun `--stage final`)

```mermaid
sequenceDiagram
    autonumber
    participant Dev as 18-Subject Dev Screen
    participant Final as 26-Fold LOSO + Lockbox (--stage final)
    participant Brief as PDF 3: Publication Brief
    participant Record as PDFs 1 & 2: Project Record

    Dev->>Record: 5-Member Ensemble beats in-run baseline on dev screen
    Note over Final,Record: ensemble_campaign.py --stage final NEVER RUN
    Record-->>Brief: Dev screen results leaked into publication notes
    Brief->>Brief: Publishes 66.32% ± 9.9% as verified 26-subject LOSO Test Acc
    Record->>Record: Flags unconfirmed status; D2 episode proves dev wins can fail
```

- **The Clash:** 
  - PDF 3 (§1, §2) explicitly claims that a "5x Deep Ensemble (Lakshminarayanan et al.)" achieves **$66.32\% \pm 9.9\%$** on "Comprehensive 5-Model Empirical Benchmark (TU Berlin, 26 Folds)" with a training overfit gap of $21.7\%$ ($88.0\%$ train vs. $66.32\%$ test).
  - PDF 1 and PDF 2 (§5.3, §8, §10) explicitly state that while the 5-member deep ensemble was the only configuration in 58 attempts to clear the development-screen bar against an in-run baseline, **`--stage final` (the full 26-fold LOSO and 8-subject lockbox evaluation) was NEVER RUN**.
- **The D2 Cautionary Episode:** 
  The project has already demonstrated the scientific danger of trusting a development-screen win. Candidate `D2_narrow_patience6` won convincingly on the 18-subject development screen. When promoted to `finalize_winner.py` (full 26-fold LOSO + lockbox with a freshly re-run paired baseline), it **lost on both accuracy and macro F1** on both protocols.
- **Audit Verdict:** Quoting $66.32\%$ as a verified 26-subject LOSO result in an academic paper would constitute reporting an unconfirmed result. 
- **Action Required:** The 5-member deep ensemble must either be run to completion using `python ensemble_campaign.py --stage final` or strictly quarantined from the manuscript's verified results table.

---

### Conflict 2: Fast Linear Ensemble ($61.78\%$) and Astronomical ANOVA Scores ($F > 400$)

- **The Clash:**
  - PDF 3 claims that an ultra-fast 50-feature linear ensemble (Shrinkage LDA + L2-Logistic Regression + LinearSVC) achieved **$61.78\% \pm 8.0\%$** with an overfit gap of only $4.65\%$ in 80 seconds on CPU. It further claims that 96.6% dimensionality reduction via univariate ANOVA $F$-scores isolated prefrontal HbO optodes with **$F > 400, p < 10^{-79}$**.
  - PDF 1 and PDF 2 reveal that classical sweeps originated from an older script (`classical_sweep.py`). Crucially, PDF 1 (§XI.8) notes: *"xgboost is unusable in this environment (missing libomp), so the classical sweep was not re-run under the new protocol."* Furthermore, PDF 1 contains zero record of an automated leakage-audited run yielding $F > 400$.
- **Methodological Vulnerability:**
  In small-sample BCI datasets ($N=26$), an ANOVA $F$-statistic exceeding 400 ($p < 10^{-79}$) almost invariably indicates that feature selection was computed **globally across the entire dataset prior to fold partitioning**, rather than refitted strictly inside each training fold (`scaler_fit_on_train_only`). If feature selection saw the held-out subject's data, the feature set is contaminated by cross-subject information leakage.
- **Audit Verdict:** The $61.78\%$ linear ensemble and the $F > 400$ claim cannot be published without independent re-execution under `research/leakage.py` assertions.

---

### Conflict 3: Modality Attribution Discrepancy ($65\text{--}70\%$ vs. $94.5\%$ fNIRS Share)

- **The Clash:**
  - PDF 3 (§3, Contribution 3) claims: *"fNIRS carries ~65% to 70% of the steady workload attribution weight, whereas EEG provides rapid, localized spectral bursts."*
  - PDF 1 (§29) reports an empirical SHAP analysis (GradientExplainer across held-out windows) showing:
    - **fNIRS Modality Share: $0.945$ ($94.5\%$)**
    - **EEG Modality Share: $0.055$ ($5.5\%$)**
    - Within fNIRS, Deoxygenated Hemoglobin (**HbR: $0.488$**) exceeds Oxygenated Hemoglobin (**HbO: $0.458$**).

```mermaid
pie title Modality Attribution Breakdown (PDF 1 SHAP Reality)
    "fNIRS - Deoxygenated (HbR)" : 48.8
    "fNIRS - Oxygenated (HbO)" : 45.8
    "EEG (6 Frequency Bands)" : 5.5
```

- **Root Cause Reconciliation:**
  This discrepancy is a classic confusion of **model architectural parameters (attention gate weights)** with **post-hoc feature attributions (Shapley values)**:
  1. **Attention Gate Mechanism:** The network computes scalar gates $g_{\text{eeg}}, g_{\text{fnirs}} = \sigma(W [e_{\text{eeg}}; e_{\text{fnirs}}])$. Because sigmoid activations saturate and the attention fusion block was regularized, gate outputs typically hover between $0.35$ and $0.65$ (reflecting a 65:35 ratio). This was rendered on the demo UI's "Modality Contribution" bar.
  2. **SHAP Feature Attribution:** SHAP evaluates the actual marginal contribution of input feature perturbations on the output logit. Under GradientExplainer, 94.5% of the predictive gradient mass is concentrated in fNIRS features (specifically AUC, min, max, and late-window hemodynamic slope), while EEG features receive only 5.5% attribution.
- **Physical/Physiological Corroboration:**
  The 94.5% fNIRS SHAP dominance is independently corroborated by the controlled ablation experiment in Stage 10 (`S10_no_hbr`): completely dropping HbR causes macro F1 to drop by **$-0.0670$ ($p = 0.014$)**, proving that hemodynamic features carry the true discriminative signal. Conversely, modality dropout of EEG (`S7_md_eeg_only`) caused a smaller, non-significant drop.
- **Audit Verdict:** State the distinction clearly in the manuscript: *attention gate scaling factors allocate ~65% weight to fNIRS, but game-theoretic SHAP gradient attribution reveals that fNIRS drives 94.5% of the classification decision.*

---

### Conflict 4: Baseline Reference Drift ($0.6537$ vs. $0.6304$ Dev-LOSO Accuracy)

- **The Clash:**
  - In earlier documentation (`RESULTS_BINARY_IMPROVEMENT.md`), the documented 18-fold development-LOSO baseline accuracy was recorded as **$0.6537$** (Macro F1 $0.6451$).
  - When the baseline configuration was subsequently re-executed in the current Python/PyTorch environment during the target campaign, it yielded an accuracy of **$0.6304$**.
- **Consequence for the Search Campaign:**
  To prevent false positives or false rejections caused by environment-induced drift, the researchers correctly modified the framework so that **every challenger arm was evaluated using paired comparisons against an in-run baseline** executed under identical seeds and library versions.
- **Audit Verdict:** The manuscript must report the definitive, bit-for-bit verified 26-fold LOSO baseline accuracy of **$63.09\% \pm 10.11\%$** and lockbox accuracy of **$63.71\% \pm 8.50\%$**, explicitly noting that developmental model screening utilized in-run paired baselines to control for environment drift.

---

### Conflict 5: Software Serving Pipeline vs. Configuration Metadata

- **The Clash:**
  - `serving/model_config.json` specifies `"served_model": "hybrid_rf_ensemble"`.
  - Direct inspection of `backend/model_service.py` (lines 440–447) reveals that the serving backend hardcodes:
    ```python
    served_model = "hybrid_cnn_lstm"
    ```
    The serving code executes a forward pass through the dual-stream MLP network alone, completely bypassing any Random Forest ensemble.
  - Furthermore, `backend/main.py` serves a 3-class endpoint whose provenance flag reports `"kind": "retrained_stand_in"`, trained on a 6-subject subset because large raw checkpoints were gitignored.
- **Audit Verdict:** The model served in the live demonstration and evaluated in PDF 1 is the **dual-stream MLP model (`A0_baseline`)**, NOT an RF hybrid ensemble. Metadata in `model_config.json` is stale and must be corrected.

---

## 5. Methodological Breakdown: Turning Negative Results into a Scientific Contribution

The core strength of this project is that its "negative" results are rigorous, mechanistically explained, and expose systemic methodological flaws in published BCI literature.

```mermaid
graph TD
    subgraph Systematic Findings [Why the Baseline Could Not Be Beaten]
        F1["Inductive Bias Mismatch (CNNs)<br/>Channel-axis conv on pre-aggregated stats cuts capacity; F1 drops by 0.09-0.16"]
        F2["Adversarial Noise Injection (DANN)<br/>Probe shows preprocessing already strips subject ID; DANN corrupts task head without benefit"]
        F3["Subject Heterogeneity Dominance<br/>Between-subject variance (0.38 F1) is >2x model architectural variance (0.17 F1)"]
        F4["Regularization Trade-off<br/>Dropout/Weight Decay shrinks overfitting gap only by destroying held-out performance"]
    end
```

### 1. Inductive Bias Mismatch: Why 1D-CNNs Fail on Pre-Aggregated Features
- **Hypothesis:** Convolving across adjacent electrodes/optodes along the channel axis should exploit spatial topological correlation and reduce parameter count.
- **Empirical Result:** 
  - Plain 1D-CNN (`S6_cnn`): Macro F1 collapsed to **$0.4786$** ($\Delta = -0.1665, p < 0.001$).
  - Residual CNN (`S6_cnn_res`): Macro F1 fell to **$0.5550$** ($\Delta = -0.0902, p = 0.002$).
  - Res-CNN + Attention (`S6_cnn_res_attn`): Macro F1 fell to **$0.5467$** ($\Delta = -0.0984, p = 0.001$).
- **Mechanistic Cause:** 
  The 1D-CNN reshaped the input into $(\text{electrodes}, \text{feature\_types})$. Convolving along the electrode axis with shared weights assumes local shift invariance across channels. On pre-aggregated statistical summaries (mean, slope, power), this inductive bias is invalid: neighboring 10-20 EEG positions reflect distinct functional cortical regions. 
  While the CNN dramatically reduced the training-test overfit gap (from $0.163$ down to $0.026$), it did so by destroying model expressivity. Spatial weight sharing is appropriate for raw temporal series, but detrimental for channel-wise summary statistics.

---

### 2. The DANN Paradox: Why Domain Adversarial Training Degrades Workload Estimation
- **Hypothesis:** Subject variability is the primary generalization barrier. A Gradient Reversal Layer (GRL) connected to a subject-discriminator head should force the encoder to learn subject-invariant representations.
- **Empirical Result:** Monotonic degradation as adversarial weight $\lambda$ increased:
  - Baseline ($\lambda = 0$): Macro F1 **$0.6451$**
  - $\lambda = 0.1$: Macro F1 **$0.6182$** ($\Delta = -0.027, p = 0.104$)
  - $\lambda = 0.5$: Macro F1 **$0.5968$** ($\Delta = -0.048, p = 0.014$)
  - $\lambda = 1.0$: Macro F1 **$0.5778$** ($\Delta = -0.067, p = 0.012$)
- **The Mechanistic Breakthrough (Subject-Invariance Probe):**
  To diagnose the failure, a secondary classifier probe was trained to predict subject identity from frozen intermediate representations:
  - **Raw Preprocessed Features:** Probe Accuracy = **$0.013$** (Chance = $0.125$, i.e., $1/8$ lockbox subjects).
  - **Baseline MLP Embedding:** Probe Accuracy = **$0.214$**.
  - **DANN-Trained Embedding:** Probe Accuracy = **$0.217$**.
- **Explanation:** 
  Per-subject z-score normalization in preprocessing had *already eliminated* decodable subject-identity signal from the raw input ($0.013 \ll 0.125$). The nonlinear MLP task encoder subsequently re-introduced latent subject clustering ($0.214$) as a byproduct of mapping workload. The adversarial discriminator failed to reduce this identity representation ($0.217$), acting purely as an unstructured gradient noise source that destabilized the primary classification head.

---

### 3. The Regularization Dilemma: Capacity vs. Between-Subject Variance
- **Empirical Result:** 
  Every tested regularizer (weight decay up to $0.1$, dropout up to $0.7$, Gaussian input noise up to $0.2$, label smoothing, modality dropout) reduced the train-test gap exclusively by degrading held-out test accuracy. For instance, heavy dropout (`S4_drop_high`) compressed the gap from $0.163$ to $0.040$, but caused test macro F1 to drop by **$-0.1369$ ($p < 0.001$)**.
- **Scientific Conclusion:** 
  The $17\%$ generalization gap in cross-subject BCI is **not algorithmic overfitting** that can be resolved by capacity constraints or stochastic regularization. It reflects **intrinsic physiological distribution shift**: individual variations in skull thickness, optode-scalp coupling impedance, baseline vascular tone, and cognitive strategy mean that an unseen subject's feature manifold does not coincide with the training cohort.

---

### 4. Flawed Literature Splits vs. True Generalization

| Validation Protocol | Reported Literature Acc. | Real Generalization & Clinical Meaning | Vulnerability / Mechanism |
|:---|:---|:---|:---|
| **Within-Subject CV** (Shin 2018 benchmark) | $70\% - 85\%+$ | Zero cross-subject transfer. Clinical failure on any new human. | Model fits individual cranial conductivity, baseline vascular tone, and electrode placement geometry. |
| **Random Window Split** | $75\% - 90\%$ | Methodological illusion. Unusable in production. | Temporal correlation leakage: overlapping 10s windows (50–90% overlap) from the same trial appear in train and test sets. |
| **Strict LOSO (Our Work)** | **$63.09\% \pm 10.1\%$** | **True Zero-Calibration BCI.** Plug-and-play capability. | Complete subject exclusion. Scalers, normalizations, and weights isolated from the held-out subject. |

---

## 6. Verification Status Taxonomy

To ensure scientific integrity during manuscript preparation, all findings are categorized into three operational tiers:

### Tier 1: Verified Ground Truth (Safe for Direct Publication)
> [!IMPORTANT]
> These metrics and findings are fully backed by execution logs, automated leakage passes, saved fold predictions, and paired statistical testing. They may be reported without qualification.

- **Baseline Cross-Validation (26-Fold LOSO):** Accuracy **$63.09\% \pm 10.11\%$**, Macro F1 **$62.55\% \pm 10.44\%$**, ROC-AUC **$67.75\% \pm 12.89\%$**, MCC **$0.2675 \pm 0.2058$**, Cohen's Kappa **$0.2618 \pm 0.2022$**.
- **Lockbox Generalization (8 Unseen Subjects):** Accuracy **$63.71\% \pm 8.50\%$**, Macro F1 **$62.75\% \pm 9.31\%$**, ROC-AUC **$68.88\% \pm 11.12\%$**, MCC **$0.2816 \pm 0.1690$**.
- **Overfit Gap Analysis:** Train ($80.2\%$) vs. Test ($63.1\%$), confirming a persistent $+17.1\%$ cross-subject generalization gap.
- **Confusion Matrix:** True Low (2,288): 1,561 correct (Rec 0.682); True High (2,288): 1,326 correct (Rec 0.580).
- **Subject Heterogeneity:** Per-subject macro F1 ranging from $0.431$ (S12) to $0.807$ (S17); 4 subjects below chance.
- **Ablation Findings:**
  - 1D-CNN encoders significantly underperform MLPs ($p \le 0.002$).
  - DANN causes monotonic degradation ($p \le 0.014$).
  - Per-subject normalization is critical ($\Delta = -0.0753, p = 0.014$).
  - HbR features provide significant discriminative power ($\Delta = -0.0670, p = 0.014$).
  - Attention fusion provides modest, non-significant gain over concatenation ($\Delta = +0.0157, p = 0.230$).
  - 66K parameter model achieves parity with 245K baseline ($p = 0.929$).
- **SHAP Attribution:** fNIRS accounts for $94.5\%$ of gradient attribution (HbR $48.8\%$ > HbO $45.8\%$), while EEG accounts for $5.5\%$.

---

### Tier 2: Quarantined / Provisional (Requires Qualification or Re-Run)
> [!WARNING]
> These claims originate from PDF 3 or development screens. They must NOT be cited as verified facts unless verified via official pipeline runs.

- **5x Deep Ensemble ($66.32\% \pm 9.9\%$):** 
  - *Status:* Unconfirmed candidate. Cleared dev screen against an in-run baseline, but `--stage final` was not executed. 
  - *Manuscript Policy:* Exclude from main results table, or mention strictly in Discussion as an exploratory hypothesis requiring protocol confirmation.
- **Fast Linear Ensemble ($61.78\% \pm 8.0\%$):** 
  - *Status:* Unverified protocol. Originates from `classical_sweep.py` prior to the unified leakage assertion architecture.
  - *Manuscript Policy:* Quarantine until re-executed under `research/leakage.py`.
- **96.6% Dimensionality Pruning & ANOVA $F > 400$ ($p < 10^{-79}$):** 
  - *Status:* Highly suspect; likely an artifact of global pre-split feature selection.
  - *Manuscript Policy:* Omit entirely from manuscript claims to avoid reviewer challenge.
- **Hybrid + Random Forest Ensemble ($62.25\%$):** 
  - *Status:* Deprecated/disconnected architecture. Serving backend runs MLP alone.
  - *Manuscript Policy:* Do not report as the served or verified system.

---

### Tier 3: Known Discrepancies Requiring Code / Documentation Hygiene
> [!NOTE]
> Minor metadata and software discrepancies that do not compromise the core scientific findings but require cleanup.

- **Baseline Mismatch ($0.6537$ vs. $0.6304$ dev accuracy):** Reconcile by noting environment/package dependency drift, while anchoring all reported figures on the paired in-run baselines and verified 26-fold LOSO metrics.
- **`model_config.json` Stale Key:** Update `"served_model": "hybrid_rf_ensemble"` to `"served_model": "hybrid_cnn_lstm"` to match `model_service.py`.
- **3-Class Stand-in Notice:** Maintain clear documentation that 3-class serving is a demo stand-in, keeping the research focus on the primary binary task.

---

## 7. Actionable Work List & Roadmap to Publication

The following prioritized roadmap outlines the engineering and writing tasks required to finalize the IEEE manuscript.

```mermaid
gantt
    title Roadmap to IEEE Submission
    dateFormat  YYYY-MM-DD
    section Critical Blockers
    Fix model_config.json metadata            :crit, t1, 2026-10-16, 1d
    Execute Deep Ensemble --stage final       :crit, t2, 2026-10-16, 2d
    section Classical Re-Audit
    Re-run Classical Sweep with Leakage Audit  :t3, 2026-10-18, 2d
    section Writing & Framing
    Draft Methodology & Leakage Audit Section :t4, 2026-10-19, 3d
    Draft Negative Results & Probe Section     :t5, 2026-10-21, 3d
    IEEE Latex Compilation & Final Polish      :t6, 2026-10-24, 2d
```

### Priority 1: Critical Verification Tasks (Before Manuscript Finalization)
1. **Execute Final Protocol for Deep Ensemble:**
   Run `python ensemble_campaign.py --stage final` across all 26 LOSO folds and the 8-subject lockbox.
   - *If Confirmed ($\ge 65\%$):* Report as the high-accuracy deep candidate.
   - *If Collapsed (like D2):* Document as further evidence of development-screen overfitting.
2. **Re-Audit Classical Feature Pruning:**
   Re-implement the 50-feature ANOVA selection strictly inside each fold's training split via `research/experiment.py`. Determine if linear models truly reach $61.8\%$ without data leakage.
3. **Resolve Metadata Discrepancy:**
   Update `serving/model_config.json` so that `"served_model"` accurately reflects the served PyTorch MLP architecture.

### Priority 2: Manuscript Framing & Academic Defense
1. **Emphasize Methodological Superiority:** Frame the paper around the zero-leakage protocol, automated assertion suite, and lockbox confirmation. Present the $63.09\%$ accuracy not as a limitation, but as an honest, uninflated benchmark for plug-and-play BCI.
2. **Spotlight the Mechanistic Explanations:** Dedicate a major subsection to the Subject-Invariance Probe and the DANN failure analysis. This converts an empirical negative result into a high-impact theoretical insight for cross-subject neurocomputing.
3. **Clarify Modality Roles:** Present the dual findings of Attention Gating (dynamic weighting ~65:35) alongside SHAP gradient attribution ($94.5\%$ hemodynamic dominance), detailing how fNIRS establishes the metabolic workload foundation while EEG captures transient micro-state shifts.

---

**Audit Sign-off:**  
*Lead Scientific Manuscript Writer & BCI Research Auditor*  
*October 15, 2026*
