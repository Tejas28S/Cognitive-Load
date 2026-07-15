# Results

## Problem Statement

This project classifies cognitive load from multimodal EEG and fNIRS features on the TU Berlin/Shin n-back dataset. The final task is a 3-class prediction problem: Low (0-back), Medium (2-back), and High (3-back). The active fixed-split pipeline uses `data/tuberlin_eeg_fnirs.csv` with 654 full-trial samples, 26 subjects, 180 EEG band-power/engagement features, and 468 HbO fNIRS features.

## Final Results

Final selected config traceability: selected in commit `11ee7ce`, closed out in commit `b734ca0`, explained with Phase 4 artifacts in commits `ee015f8` and `3c2700f`.

| Evaluation | Accuracy | Macro F1 | AUC | Per-class recall [L,M,H] |
|---|---:|---:|---:|---|
| Fixed split | 0.4267 | 0.4064 | 0.6305 | [0.5625, 0.5185, 0.1875] |
| LOSO cross-validation | 0.3982 +/- 0.0742 | 0.3642 +/- 0.0894 | 0.6011 +/- 0.0801 | [0.5919, 0.3248, 0.2853] |

Final configuration: `PER_SUBJECT_NORMALIZE=True`, `PER_SUBJECT_NORMALIZE_STAGE="before_log"`, train-fold balanced class weights, `FUSION_TYPE="attention"`, and `LOSS_FN="weighted_ce"`. Fixed-split and LOSO numbers should always be reported together because they answer different questions: fixed split measures performance when subjects can appear across train/test trials, while LOSO measures generalization to unseen subjects.

## Methodology Journey

Early runs showed unstable class behavior and misleadingly variable macro F1. The first diagnosis found that EEG band-power features were highly skewed, so EEG preprocessing was changed to log compression followed by train-only `RobustScaler`; fNIRS HbO/HbR features received signed-log compression before train-only scaling. A checkpoint-selection issue was also fixed: selecting checkpoints by validation macro F1 rather than validation loss/accuracy reduced the chance of accepting a model that looked numerically adequate while collapsing one class.

Class weighting was then tested because Medium/High recall repeatedly collapsed. The heavier `[1.0, 2.0, 1.0]` style weighting overcorrected and damaged class balance, and the final normalized LOSO retune `[1.0, 1.1, 1.3]` also failed, dropping LOSO Macro F1 from 0.3642 to 0.3329. This led to the final choice of balanced train-fold weights rather than further manual tuning.

The EEG feature rebuild was a useful near-miss rather than the final path. Sub-window EEG features and per-subject baseline ideas exposed real data/feature issues through the B1-B4 diagnosis, including label/feature alignment checks and a 540-vs-180 EEG feature caveat for sliding-window ablations. Those experiments did not beat the restored fixed-split reference, but they improved confidence that the final reported comparisons were not silently driven by a data merge bug.

The most important methodological finding was the fixed-split versus LOSO gap. The non-normalized fixed split reached Macro F1=0.4678, but that split groups by `subject_id + trial_id`, so different trials from the same subject can appear in train and test. That can reward subject-specific absolute EEG/fNIRS baselines that do not transfer to unseen people. Per-subject normalization removes much of that shortcut: fixed-split Macro F1 drops to 0.4064, while LOSO Macro F1 improves from 0.2982 to 0.3642 and AUC improves from 0.5547 to 0.6011.

## Explainability Findings

SHAP explainability was run on the final `before_log` checkpoint (`outputs/phase3_per-subject_normalization_before_log.pt`) using `explain_model.py`; `DeepExplainer` failed the additivity check on this PyTorch model, so the script used `GradientExplainer`. The strongest categories were fNIRS HbO summary features, especially skew, early-window response, slope, max, mean, delta, kurtosis, and AUC, which matches the broader project finding that fNIRS carried more stable signal than EEG. EEG bands contributed much less overall, though high-load explanations retained small alpha, delta, theta, gamma, beta, and engagement contributions. This is physiologically plausible for an n-back cognitive-load task: the model appears to lean primarily on hemodynamic response shape while retaining weaker EEG band-power cues rather than relying on a single arbitrary feature. Outputs: `outputs/shap_summary.png`, `outputs/shap_per_class.png`, and `outputs/shap_category_importance.csv`.

Attention/fusion visualization was rerun on the same final checkpoint with `visualize_attention.py`. The learned gate leans slightly toward fNIRS overall (mean fNIRS gate=0.514 vs EEG gate=0.478), consistent with the SHAP result and with earlier model comparisons showing fNIRS carried the more reliable signal. Class-wise shifts are modest: Low samples show the lowest EEG gate and highest fNIRS gate, while Medium/High move EEG upward only slightly. Outputs: `outputs/attention_weights_final.png` and `outputs/attention_weights_final.csv`.

## Limitations

Cross-subject generalization remains meaningfully below in-distribution fixed-split performance, even after per-subject normalization. High-load recall remains the hardest part of the final fixed-split result, and LOSO High recall improves only to about 29%. EEG signal remains weaker than fNIRS throughout the project; SHAP and gate visualization both show the final model leaning on fNIRS HbO response-shape features. Classical baselines and deep ablations were useful for ruling out architecture-only explanations, but they did not remove the core challenge of subject variability.

## Negative Results

Several plausible improvements were tried and rejected. Focal loss lowered Macro F1 to 0.3457, cross-attention fusion lowered Macro F1 to 0.3256, per-modality PCA lowered Macro F1 to 0.3228, and window/stride variants failed to beat the 0.4678 fixed-split reference. The `after_log` per-subject normalization stage was rejected on LOSO (0.3476 vs 0.3642 for `before_log`), and the one allowed manual class-weight retune `[1.0, 1.1, 1.3]` was rejected because it lowered LOSO Macro F1 to 0.3329. These results support closing accuracy tuning and moving to presentation/demo work.

## Capstone Summary

The final EEG+fNIRS attention-fusion model shows credible but limited cross-subject cognitive-load classification. On the standard fixed split it achieves 42.67% accuracy, Macro F1=0.4064, and AUC=0.6305. Under the stricter 26-fold leave-one-subject-out protocol, it achieves 39.82% +/- 7.42% accuracy, Macro F1=0.3642 +/- 0.0894, and AUC=0.6011 +/- 0.0801. The fixed-split/LOSO gap is itself a key finding: subject-specific physiological baselines are a real confound, and per-subject normalization partially mitigates this while reducing fixed-split performance.

Latest code changes focus on the diagnosed scaling issue and Medium-class collapse:

- EEG features are log-transformed before train-only `RobustScaler` fitting.
- HbO/HbR features were checked for skew and now use a signed-log transform before scaling.
- Per-subject normalization is available for both EEG and fNIRS and defaults to the empirically better `before_log` stage for cross-subject evaluation.
- Class weights are configurable; the final selected model uses balanced train-fold weights, and the single normalized LOSO manual retune `[1.0, 1.1, 1.3]` was rejected.
- Training checkpoints now select the model with best validation macro F1 instead of best validation loss.
- Overfitting controls were tightened after the latest run showed train accuracy rising to ~85% while validation stayed near chance.
- Focal loss is available through `LOSS_FN = "focal"` while `weighted_ce` remains the default.
- A Random Forest baseline script and attention-gate visualization script are available.

## Implemented Fixes

| Area | Status | Notes |
|---|---:|---|
| Sliding-window extraction | Done | `sliding_window_extractor.py` writes `data/tuberlin_sliding.csv` with `subject_id`, `trial_id`, and window metadata. |
| Leakage-safe splitting | Done | `data_loader.py` uses grouped splits when `subject_id` and/or `trial_id` are available. |
| Improved config | Done | `config_improved.py` targets `data/tuberlin_sliding.csv` and groups by `subject_id + trial_id`. |
| PCA training | Done | `train_with_pca.py` now uses the same grouped train/val/test indices and fits scaler/PCA on train only. |
| Windows compatibility | Done | Config stdout encoding is set to UTF-8 to avoid PowerShell crashes on status symbols. |
| EEG log scaling fix | Done | `data_loader.py` applies `log1p(clip(x, 0, None))` to `eeg_` columns before scaling. |
| fNIRS skew fix | Done | HbO/HbR skew check showed extreme max-to-75th-percentile ratios, so signed-log is enabled for `hbo_`/`hbr_` columns. |
| Per-subject normalization | Done | `data_loader.py` can z-score EEG and fNIRS within each subject before or after log transforms; current default is `PER_SUBJECT_NORMALIZE=True`, `PER_SUBJECT_NORMALIZE_STAGE="before_log"` based on LOSO improvement. |
| Class weighting | Done | Balanced train-fold class weights remain the selected default. The single normalized LOSO manual retune `[1.0, 1.1, 1.3]` was rejected. |
| Macro-F1 checkpointing | Done | `trainer.py` now saves/restores the checkpoint with best validation macro F1 to reduce class-collapse risk. |
| Overfitting mitigation | Done | `EARLY_STOP=12`, `LR_PATIENCE=5`, `CNN_DROPOUT=0.5`, `FC_DROPOUT=0.6`, `OPTIMIZER="adamw"`, `WEIGHT_DECAY=1e-4`. |
| Capacity reduction | Done | Config now uses `CNN_FILTERS=[64, 96]`, `FUSION_DIM=48`, and `FC_UNITS=[48, 24]`; Hybrid is about 211K trainable params. |
| Focal loss option | Done | `trainer.py` supports `LOSS_FN = "focal"` with `FOCAL_GAMMA = 2.0`. |
| Classical baseline | Done | `train_baseline.py` trains Random Forest and optional RBF SVM on the same grouped split. |
| Attention visualization | Done | `visualize_attention.py` saves `outputs/attention_weights.png` for trained attention-fusion checkpoints. |
| Statistical comparison utility | Done | `stats_check.py` runs paired t-test and Wilcoxon checks on Hybrid vs fNIRS CV fold metrics. |
| RF feature importance | Done | `train_baseline.py` saves `outputs/rf_feature_importances.csv` and appends modality/top-feature summaries to this file. |

## Validation Performed

| Check | Result |
|---|---:|
| `python -m py_compile config.py data_loader.py train_with_pca.py sliding_window_extractor.py config_improved.py` | Pass |
| `python sliding_window_extractor.py --help` | Pass |
| `python -c "import config_improved"` | Pass |
| `python -c "from train_with_pca import main"` | Pass |
| Loader smoke test on `data/tuberlin_eeg_fnirs.csv` | Pass |
| `python -m py_compile config.py config_improved.py data_loader.py trainer.py train_with_pca.py train_baseline.py visualize_attention.py models\architectures.py` | Pass |
| `python -m py_compile config.py config_improved.py trainer.py train_baseline.py data_loader.py main.py train_with_pca.py evaluate.py` | Pass |
| `python train_baseline.py` | Pass |
| Synthetic `train_model` smoke test with `EPOCHS=1` | Pass |
| Overfitting-fix synthetic `train_model` smoke test with AdamW | Pass |

## Overfitting Fix Pass

Applied after the run where Hybrid reached macro F1 0.3866 but training curves showed severe overfitting.

| Setting | Before | After | Rationale |
|---|---:|---:|---|
| `EARLY_STOP` | 30 | 12 | Stop closer to the epoch range where validation stopped improving. |
| `LR_PATIENCE` | 15 | 5 | Reduce LR sooner when validation macro F1 plateaus. |
| `CNN_DROPOUT` | 0.4 | 0.5 | Stronger encoder regularization. |
| `FC_DROPOUT` | 0.5 | 0.6 | Stronger classifier-head regularization. |
| `OPTIMIZER` | Adam | AdamW | Decoupled weight decay. |
| `WEIGHT_DECAY` | 1e-3 Adam L2 | 1e-4 AdamW | Standard AdamW regularization baseline. |
| Hybrid params | ~325K | ~211K | Smaller projection/fusion/head widths to reduce memorization. |

Model parameter counts with the new config:

| Model | Trainable Parameters |
|---|---:|
| Hybrid CNN-LSTM | 211,059 |
| EEG only | 64,083 |
| fNIRS only | 136,659 |

Next validation: re-run `python main.py` and check whether validation curves stop diverging earlier and whether EEG-only still collapses toward mostly-Medium predictions.

## Priority -1 Stability Fixes

| Item | Status | Notes |
|---|---:|---|
| Checkpoint on validation macro F1 | Done | `trainer.py` uses `val_f1_macro` for scheduler stepping, early stopping, and best checkpoint restore. |
| Walk back Medium class weights | Done | Manual weights changed from `[1.0, 2.0, 1.0]` to `[1.08, 1.30, 1.10]`. |
| Inspect RF feature importances | Done | HbR=0.4331, HbO=0.3998, EEG=0.1671 total importance; no EEG features appear in the top 20 RF features. |
| Confirm no deep-model class collapse | Pending | Re-run `python main.py` with the new checkpointing and weights, then compare per-class recall. |

## Skew Diagnostics

| Modality | Finding | Action |
|---|---|---|
| EEG | Multiple EEG bands had severe outliers, e.g. max-to-75th-percentile ratios above 200x. | Apply `log1p` to `eeg_` columns before scaling. |
| fNIRS | HbO/HbR power-style features also showed severe outliers, with max-to-75th-percentile ratios above 300x in the worst columns. | Apply signed-log to `hbo_`/`hbr_` columns before scaling. |

Existing CSV smoke-test split:

| Split | Samples | Groups | Class Counts |
|---|---:|---:|---|
| Train | 404 | 16 subjects | Low=130, Medium=144, High=130 |
| Val | 100 | 4 subjects | Low=32, Medium=36, High=32 |
| Test | 150 | 6 subjects | Low=48, Medium=54, High=48 |

## Training Results

Fill the deep-model rows after running `main.py` / `train_with_pca.py`.

| Run | Model | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---|---:|---:|---:|---|
| Sliding windows | Hybrid CNN-LSTM | TBD | TBD | TBD | Run `python main.py`. |
| Sliding windows | EEG only | TBD | TBD | TBD | Produced by `main.py` unless `--no-ablation` is used. |
| Sliding windows | fNIRS only | TBD | TBD | TBD | Produced by `main.py` unless `--no-ablation` is used. |
| Sliding windows | Random Forest baseline | 0.4021 | 0.3918 | 0.3948 | AUC=0.6000; Medium recall=0.5011; HbO+HbR fNIRS columns; `python train_baseline.py`. |
| Sliding windows + PCA | PCA Hybrid | TBD | TBD | TBD | Run `python train_with_pca.py --n-components 150`. |
| Sliding windows + PCA | PCA EEG only | TBD | TBD | TBD | Produced by `train_with_pca.py`. |
| Sliding windows + PCA | PCA fNIRS only | TBD | TBD | TBD | Produced by `train_with_pca.py`. |

## Commands For Final Run

```bash
python sliding_window_extractor.py \
    --nirs-dir data/NIRS_01-26_MATLAB \
    --eeg-dir data/EEG_01-26_MATLAB \
    --output data/tuberlin_sliding.csv

cp config.py config_backup.py
cp config_improved.py config.py

python main.py
python train_with_pca.py --n-components 150
```

The main expected output artifact is `outputs/evaluation_dashboard.png`, containing training curves, confusion matrices, and hybrid vs. unimodal comparisons.

## Classical Baseline Run

Run timestamp: 2026-07-08 09:40

Preprocessing: EEG_LOG_TRANSFORM=True, FNIRS_SIGNED_LOG_TRANSFORM=True. Split grouping: subject_id + trial_id when present.

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | Medium Recall |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.3734 | 0.3637 | 0.3660 | 0.5563 | 0.4789 |

Interpretation: historical run before config was corrected to include HbR in the fNIRS modality; use the later HbO+HbR baseline for comparison.

## Classical Baseline Run

Run timestamp: 2026-07-08 09:42

Preprocessing: EEG_LOG_TRANSFORM=True, FNIRS_SIGNED_LOG_TRANSFORM=True. Split grouping: subject_id + trial_id when present.

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | Medium Recall |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.4021 | 0.3918 | 0.3948 | 0.6000 | 0.5011 |

Interpretation: compare this baseline with the deep models to separate data ceiling from architecture effects.

## Classical Baseline Run

Run timestamp: 2026-07-08 09:46

Preprocessing: EEG_LOG_TRANSFORM=True, FNIRS_SIGNED_LOG_TRANSFORM=True. Split grouping: subject_id + trial_id when present.

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | Medium Recall |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.4021 | 0.3918 | 0.3948 | 0.6000 | 0.5011 |

Interpretation: compare this baseline with the deep models to separate data ceiling from architecture effects.

## Classical Baseline Run

Run timestamp: 2026-07-08 20:07

Preprocessing: EEG_LOG_TRANSFORM=True, FNIRS_SIGNED_LOG_TRANSFORM=True. Split grouping: subject_id + trial_id when present.

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | Medium Recall |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.4021 | 0.3918 | 0.3948 | 0.6000 | 0.5011 |

Interpretation: compare this baseline with the deep models to separate data ceiling from architecture effects.

Random Forest feature importance summary:

| Modality | Total Importance |
|---|---:|
| HbR | 0.4331 |
| HbO | 0.3998 |
| EEG | 0.1671 |

Top 20 Random Forest features:

| Rank | Feature | Modality | Importance |
|---:|---|---|---:|
| 1 | hbo_ch06_max | HbO | 0.00380 |
| 2 | hbo_ch07_max | HbO | 0.00282 |
| 3 | hbo_ch05_max | HbO | 0.00274 |
| 4 | hbo_ch23_min | HbO | 0.00269 |
| 5 | hbr_ch14_min | HbR | 0.00259 |
| 6 | hbr_ch28_max | HbR | 0.00254 |
| 7 | hbo_ch09_max | HbO | 0.00254 |
| 8 | hbo_ch23_max | HbO | 0.00250 |
| 9 | hbo_ch06_mean | HbO | 0.00249 |
| 10 | hbr_ch23_min | HbR | 0.00248 |
| 11 | hbo_ch23_mean | HbO | 0.00237 |
| 12 | hbo_ch23_late | HbO | 0.00236 |
| 13 | hbo_ch23_early | HbO | 0.00230 |
| 14 | hbr_ch09_min | HbR | 0.00222 |
| 15 | hbr_ch06_min | HbR | 0.00211 |
| 16 | hbr_ch14_mean | HbR | 0.00209 |
| 17 | hbr_ch14_late | HbR | 0.00209 |
| 18 | hbr_ch14_early | HbR | 0.00203 |
| 19 | hbr_ch23_late | HbR | 0.00201 |
| 20 | hbr_ch14_max | HbR | 0.00195 |

Full importance table saved to `outputs/rf_feature_importances.csv`.

## Phase 0 Baseline Recheck

Run timestamp: 2026-07-09

Tag: `BASELINE_RECHECK_FAILED`

`config.py` was restored to the plan's requested attention fusion and manual class weights `[1.08, 1.30, 1.10]`. `RobustScaler` for EEG was retained because an earlier iteration log described it as neutral/slightly helpful. The default original CSV remained `data/tuberlin_sliding.csv`.

Command: `python main.py --no-ablation`

| Model | Accuracy | Macro F1 | AUC |
|---|---:|---:|---:|
| Hybrid CNN-LSTM | 0.3175 | 0.1713 | 0.5162 |

Interpretation: this did not reproduce the target known-good baseline of Hybrid Acc=40.56%, Macro F1=0.3866, AUC=0.5875. Treat the older target as not yet reproducibly restored from the current repository state.

## Phase 1 EEG Rebuild Summary

Implemented a parallel v2 extraction path, keeping `data/tuberlin_sliding.csv` intact:

- `sliding_window_extractor_v2.py` writes EEG as 5 real 2-second sub-window timesteps per 10-second label window.
- EEG is per-subject baseline-normalized using that subject's 0-back windows.
- fNIRS feature extraction is unchanged.
- `data_loader.py` detects `eeg_tXX_...` columns and reshapes EEG to `(samples, timesteps, features_per_timestep)` after train-only scaling.
- `models/architectures.py` routes sequence EEG into an LSTM encoder; verified LSTM input shape `(8, 5, 180)`.

Phase 1 gate results:

| CSV | Artifact Handling | RF EEG-only AUC | Logistic EEG-only AUC | Gate |
|---|---|---:|---:|---|
| `data/tuberlin_sliding_v2.csv` | None | 0.4334 | 0.4583 | FAIL |
| `data/tuberlin_sliding_v2_artifact100.csv` | 100 uV threshold | 0.5122 | 0.5024 | FAIL |

Interpretation: artifact rejection improved EEG-only RF AUC but did not clear the required AUC >= 0.55 gate. Per the rebuild plan, Phase 2 deep retraining on v2 EEG features should remain blocked pending human review or another EEG feature strategy.

## Phase 1 EEG v2 Gate

Run timestamp: 2026-07-09 17:37

CSV: `data/tuberlin_sliding_v2.csv`

Gate: EEG-only Random Forest macro OVR AUC >= 0.55.

| Model | Accuracy | Macro F1 | AUC |
|---|---:|---:|---:|
| Random Forest | 0.2860 | 0.2308 | 0.4334 |
| Logistic Regression | 0.2909 | 0.2901 | 0.4583 |

Gate result: FAIL.

## Phase 1 EEG v2 Gate

Run timestamp: 2026-07-09 17:52

CSV: `data/tuberlin_sliding_v2_artifact100.csv`

Gate: EEG-only Random Forest macro OVR AUC >= 0.55.

| Model | Accuracy | Macro F1 | AUC |
|---|---:|---:|---:|
| Random Forest | 0.3217 | 0.2469 | 0.5122 |
| Logistic Regression | 0.3189 | 0.2587 | 0.5024 |

Gate result: FAIL.

## Debug Pass - EEG v2 Gate Diagnostics

Run timestamp: 2026-07-09 18:27

### B1 Row/Label Alignment

- v1 shape: `(7150, 1484)`; v2 shape: `(7150, 1844)`
- v1 labels: `{0: 2288, 1: 2574, 2: 2288}`; v2 labels: `{0: 2288, 1: 2574, 2: 2288}`
- merge keys: `['subject_id', 'trial_id', 'window_id']`; merge counts: `{'both': 7150, 'left_only': 0, 'right_only': 0}`
- label mismatches: `0`

### B2 Classical Gate Ablations

| Variant | CSV | X shape | RF Acc | RF Macro F1 | RF AUC | LogReg Acc | LogReg Macro F1 | LogReg AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1 flat original | `data/tuberlin_sliding.csv` | `(7150, 540)` | 0.2329 | 0.2065 | 0.3936 | 0.2601 | 0.2608 | 0.4077 |
| v1 flat + subject baseline | `data/tuberlin_sliding.csv` | `(7150, 540)` | 0.3077 | 0.2637 | 0.4616 | 0.3007 | 0.2988 | 0.4572 |
| v2 subwindow + baseline | `data/tuberlin_sliding_v2.csv` | `(7150, 900)` | 0.2888 | 0.2330 | 0.4314 | 0.3007 | 0.2992 | 0.4615 |
| v2 subwindow only | `data/tuberlin_sliding_v2_no_baseline.csv` | `(7150, 900)` | 0.2399 | 0.2065 | 0.3931 | 0.2727 | 0.2727 | 0.4217 |
| v2 subwindow + baseline + artifact100 | `data/tuberlin_sliding_v2_artifact100.csv` | `(7150, 900)` | 0.3210 | 0.2473 | 0.5119 | 0.3196 | 0.2591 | 0.5045 |

### B3/B4 Baseline Output and Flattening Checks

- CSV: `data/tuberlin_sliding_v2.csv`
- EEG columns: `900`; sequence shape: `(5, 180)`; complete grid: `True`
- NaNs: `0`; infs: `0`; max abs: `1013.8622`
- zero rate by class: `{0: 0.0, 1: 0.0, 2: 0.0}`
- top zero-rate columns: `{'eeg_t00_ch00_alpha': 0.0, 'eeg_t00_ch00_beta': 0.0, 'eeg_t00_ch00_delta': 0.0, 'eeg_t00_ch00_engage': 0.0, 'eeg_t00_ch00_gamma': 0.0, 'eeg_t00_ch00_theta': 0.0, 'eeg_t00_ch01_alpha': 0.0, 'eeg_t00_ch01_beta': 0.0, 'eeg_t00_ch01_delta': 0.0, 'eeg_t00_ch01_engage': 0.0}`

## Debug Pass Final Summary

Run timestamp: 2026-07-09

### Part A - Baseline Reproducibility

The project now has a git history. The initial source/config/docs snapshot is commit `f092006`.

Config snapshot comparison:

| Config | Default CSV | Hybrid params | Key settings |
|---|---|---:|---|
| `config.py` before restore | `data/tuberlin_sliding.csv` | 245,619 | `CNN_FILTERS=[64, 96]`, `FUSION_DIM=48`, `FC_UNITS=[48, 24]`, manual weights `[1.08, 1.30, 1.10]`, `EARLY_STOP=12`, AdamW wd=1e-4 |
| `config_improved.py` | `data/tuberlin_sliding.csv` | 245,619 | Same as pre-restore `config.py` |
| `config_backup.py` | `data/tuberlin_eeg_fnirs.csv` | 266,051 | `CNN_FILTERS=[64, 128]`, `FUSION_DIM=64`, `FC_UNITS=[64, 32]`, balanced weights, `EARLY_STOP=30`, Adam wd=1e-3 |

`config_backup.py` exactly matched the requested `HybridCNNLSTM (266,051 params)` console signature, so it was restored fully to `config.py`.

Baseline rerun:

| Command | Config | Accuracy | Macro F1 | AUC | Interpretation |
|---|---|---:|---:|---:|---|
| `python main.py --no-ablation` | restored `config_backup.py` | 0.4667 | 0.4678 | 0.6350 | Did not reproduce the older 0.3866 macro F1, but restored the 266,051-param architecture and produced a stronger baseline on the current loader/split. |

### Part B - EEG v2 Gate Diagnosis

B1 label alignment is clean: `data/tuberlin_sliding.csv` and `data/tuberlin_sliding_v2.csv` both have 7,150 rows, identical label counts, a 1:1 merge on `subject_id/trial_id/window_id`, and zero label mismatches.

B3 found a real baseline-normalization bug: the original v2 baseline output had no NaNs/infs but had extreme normalized EEG values (`max_abs=1013.8622`), consistent with unstable low-class baseline scaling.

Fix implemented: `sliding_window_extractor_v2.py` now supports robust median/IQR EEG baseline normalization with configurable clipping, defaulting to robust mode with `--eeg-baseline-clip 20`. The old z-score behavior remains available via `--eeg-baseline-mode zscore`.

Fresh fixed gate outputs:

| Variant | RF Accuracy | RF Macro F1 | RF AUC | Gate |
|---|---:|---:|---:|---|
| v2 original baseline | 0.2888 | 0.2330 | 0.4314 | FAIL |
| v2 subwindow only | 0.2399 | 0.2065 | 0.3931 | FAIL |
| v2 fixed robust baseline | 0.3021 | 0.2547 | 0.4904 | FAIL |
| v2 original artifact100 | 0.3210 | 0.2473 | 0.5119 | FAIL |
| v2 fixed robust baseline + artifact100 | 0.3273 | 0.2528 | 0.5152 | FAIL |

Conclusion: B1 and B4 are clean. B3 was responsible for the most suspicious below-random behavior and has been fixed, but the fresh fixed gate still does not clear RF AUC >= 0.55. Per the plan, Phase 2/Phase 3 deep retraining remains blocked.

## Debug Pass - EEG v2 Gate Diagnostics

Run timestamp: 2026-07-09 19:10

### B1 Row/Label Alignment

- v1 shape: `(7150, 1484)`; v2 shape: `(7150, 1844)`
- v1 labels: `{0: 2288, 1: 2574, 2: 2288}`; v2 labels: `{0: 2288, 1: 2574, 2: 2288}`
- merge keys: `['subject_id', 'trial_id', 'window_id']`; merge counts: `{'both': 7150, 'left_only': 0, 'right_only': 0}`
- label mismatches: `0`

### B2 Classical Gate Ablations

| Variant | CSV | X shape | RF Acc | RF Macro F1 | RF AUC | LogReg Acc | LogReg Macro F1 | LogReg AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1 flat original | `data/tuberlin_sliding.csv` | `(7150, 540)` | 0.2329 | 0.2065 | 0.3936 | 0.2601 | 0.2608 | 0.4077 |
| v1 flat + subject baseline | `data/tuberlin_sliding.csv` | `(7150, 540)` | 0.3077 | 0.2637 | 0.4616 | 0.3007 | 0.2988 | 0.4572 |
| v2 subwindow + baseline | `data/tuberlin_sliding_v2.csv` | `(7150, 900)` | 0.2888 | 0.2330 | 0.4314 | 0.3007 | 0.2992 | 0.4615 |
| v2 subwindow only | `data/tuberlin_sliding_v2_no_baseline.csv` | `(7150, 900)` | 0.2399 | 0.2065 | 0.3931 | 0.2727 | 0.2727 | 0.4217 |
| v2 fixed robust baseline | `data/tuberlin_sliding_v2_fixed.csv` | `(7150, 900)` | 0.3021 | 0.2547 | 0.4904 | 0.3105 | 0.3093 | 0.4743 |
| v2 subwindow + baseline + artifact100 | `data/tuberlin_sliding_v2_artifact100.csv` | `(7150, 900)` | 0.3210 | 0.2473 | 0.5119 | 0.3196 | 0.2591 | 0.5045 |
| v2 fixed robust baseline + artifact100 | `data/tuberlin_sliding_v2_artifact100_fixed.csv` | `(7150, 900)` | 0.3273 | 0.2528 | 0.5152 | 0.3161 | 0.2565 | 0.5021 |

### B3/B4 Baseline Output and Flattening Checks

- CSV: `data/tuberlin_sliding_v2.csv`
- EEG columns: `900`; sequence shape: `(5, 180)`; complete grid: `True`
- NaNs: `0`; infs: `0`; max abs: `1013.8622`
- zero rate by class: `{0: 0.0, 1: 0.0, 2: 0.0}`
- top zero-rate columns: `{'eeg_t00_ch00_alpha': 0.0, 'eeg_t00_ch00_beta': 0.0, 'eeg_t00_ch00_delta': 0.0, 'eeg_t00_ch00_engage': 0.0, 'eeg_t00_ch00_gamma': 0.0, 'eeg_t00_ch00_theta': 0.0, 'eeg_t00_ch01_alpha': 0.0, 'eeg_t00_ch01_beta': 0.0, 'eeg_t00_ch01_delta': 0.0, 'eeg_t00_ch01_engage': 0.0}`

## Phase 3 - Priority 1: Cheap Tuning Wins

Run timestamp: 2026-07-10 19:44

### EarlyStop15 Weighted CE
- Change: Set EARLY_STOP=15 while keeping weighted CE and baseline scheduler settings.
- Result: Acc=0.4667, Macro F1=0.4678, AUC=0.6350, recall=[0.5417, 0.4074, 0.4583]
- Verdict: kept - reproduces the confirmed baseline while stopping earlier than the original 30-epoch patience.

### Focal Loss Gamma2
- Change: Set LOSS_FN='focal' and FOCAL_GAMMA=2.0 with EARLY_STOP=15.
- Result: Acc=0.3667, Macro F1=0.3457, AUC=0.5285, recall=[0.1458, 0.4815, 0.4583]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

### LR Patience 3
- Change: Set LR_PATIENCE=3 with weighted CE and EARLY_STOP=15.
- Result: Acc=0.3200, Macro F1=0.2876, AUC=0.5368, recall=[0.2292, 0.1111, 0.6458]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

### LR Patience 8
- Change: Set LR_PATIENCE=8 with weighted CE and EARLY_STOP=15.
- Result: Acc=0.3667, Macro F1=0.3605, AUC=0.5728, recall=[0.2500, 0.3704, 0.4792]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

## Phase 3 - Priority 2: Cross-Attention Fusion

Run timestamp: 2026-07-10 19:46

### Attention Fusion Reference
- Change: Reference run using existing attention/gated fusion path.
- Result: Acc=0.4667, Macro F1=0.4678, AUC=0.6350, recall=[0.5417, 0.4074, 0.4583]
- Verdict: kept - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

### Cross-Attention Fusion
- Change: Use the new CrossAttentionHybrid fusion variant with identical data split/config.
- Result: Acc=0.3333, Macro F1=0.3256, AUC=0.5168, recall=[0.3958, 0.3889, 0.2083]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

## Phase 3 - Priority 3: Window/Stride Ablation

Run timestamp: 2026-07-10 20:16

### Window 8s/4s
- Change: Regenerated sliding-window CSV at data/tuberlin_sliding_8s_4s.csv.
- Samples: 9,100
- Result: Acc=0.3984, Macro F1=0.3845, AUC=0.6050, recall=[0.5606, 0.3937, 0.2292]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

## Phase 3 - Priority 3: Window/Stride Ablation

Run timestamp: 2026-07-10 20:30

### Window 12s/6s
- Change: Regenerated sliding-window CSV at data/tuberlin_sliding_12s_6s.csv.
- Samples: 5,850
- Result: Acc=0.4325, Macro F1=0.4318, AUC=0.6164, recall=[0.4807, 0.4472, 0.3669]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

### Priority 3 Summary

| Window / stride | CSV | Samples | Accuracy | Macro F1 | AUC | Recall [L,M,H] | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| 10s / 5s | `data/tuberlin_sliding.csv` | 7,150 | 0.4667 | 0.4678 | 0.6350 | [0.5417, 0.4074, 0.4583] | Keep baseline |
| 8s / 4s | `data/tuberlin_sliding_8s_4s.csv` | 9,100 | 0.3984 | 0.3845 | 0.6050 | [0.5606, 0.3937, 0.2292] | Reject |
| 12s / 6s | `data/tuberlin_sliding_12s_6s.csv` | 5,850 | 0.4325 | 0.4318 | 0.6164 | [0.4807, 0.4472, 0.3669] | Reject |

Verdict: neither alternate window size beat the 10s/5s baseline Macro F1=0.4678, so keep the existing 10s/5s configuration.

Methodological caveat: the confirmed Macro F1=0.4678 baseline was restored from `data/tuberlin_eeg_fnirs.csv`, which has 654 full-trial samples and 180 EEG features (`30 channels x 6 band/engagement features`). The sliding-window CSVs (`data/tuberlin_sliding*.csv`) have 540 EEG features because `sliding_window_extractor.py` reuses the newer EEG feature function that computes mean/std/slope over 2-second subwindows inside each labeled window (`30 channels x 6 features x 3 dynamics stats`). Therefore the Priority 3 window/stride runs are useful as data-window ablations, but they are not a pure one-variable comparison against the restored 180-feature full-trial baseline. The conservative verdict still holds: neither alternate window setting beat the reported 0.4678 guardrail, so no default change was accepted.

## Phase 3 - Priority 4: Per-Modality PCA

Run timestamp: 2026-07-10 20:37

### Per-modality PCA quick check
- Change: Fit PCA separately per modality on train split only: EEG 180->50 components (99.6% variance), fNIRS 468->100 components (96.9% variance).
- Samples: 654
- Result: Acc=0.3400, Macro F1=0.3228, AUC=0.5857, recall=[0.3542, 0.1481, 0.5417]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

## Phase 3 - Priority 5: LOSO Cross-Validation

Run timestamp: 2026-07-10 20:47
- Folds run: 26
- Accuracy: mean=0.3806, std=0.0701
- Macro F1: mean=0.2982, std=0.0945
- AUC: mean=0.5547, std=0.0853
- Per-class recall mean: [0.6512, 0.2991, 0.1987]
- Interpretation: the fixed-split Macro F1=0.4678 remains the best accepted single-split configuration, but LOSO generalization is substantially weaker. The high-load recall mean is just below the 20% floor, so use Macro F1=0.2982 +/- 0.0945 as the current subject-held-out generalization estimate rather than treating 0.4678 as cross-subject performance.

## Phase 3 - Subject Normalization Fixed-Split Stage Check

Run timestamp: 2026-07-11 00:43

### Per-subject normalization before log
- Change: Z-score EEG and fNIRS within each subject before modality log transforms.
- Result: Acc=0.4267, Macro F1=0.4064, AUC=0.6305, recall=[0.5625, 0.5185, 0.1875]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

### Per-subject normalization after log
- Change: Apply modality log transforms first, then z-score EEG and fNIRS within each subject.
- Result: Acc=0.3933, Macro F1=0.3704, AUC=0.6283, recall=[0.7083, 0.1852, 0.3125]
- Verdict: reverted - compared against baseline Macro F1=0.4678 and per-class recall floor=20%.

## Phase 3 - Priority 5: LOSO Cross-Validation

Run timestamp: 2026-07-11 00:57
- Preprocessing: PER_SUBJECT_NORMALIZE=True, PER_SUBJECT_NORMALIZE_STAGE=before_log
- Folds run: 26
- Accuracy: mean=0.3982, std=0.0742
- Macro F1: mean=0.3642, std=0.0894
- AUC: mean=0.6011, std=0.0801
- Per-class recall mean: [0.5919, 0.3248, 0.2853]
- Interpretation: per-subject normalization before log compression partially confirms the decision-boundary miscalibration hypothesis. It improved LOSO Macro F1 from 0.2982 to 0.3642, AUC from 0.5547 to 0.6011, and high-load recall from 0.1987 to 0.2853, but it missed the 0.38 LOSO target. Keep reporting the stronger fixed-split reference and the normalized LOSO estimate together; do not present 0.4678 as cross-subject performance.

## Phase 3 - Priority 5: LOSO Cross-Validation

Run timestamp: 2026-07-11 01:14
- Preprocessing: PER_SUBJECT_NORMALIZE=True, PER_SUBJECT_NORMALIZE_STAGE=after_log
- Folds run: 26
- Accuracy: mean=0.3886, std=0.0808
- Macro F1: mean=0.3476, std=0.0841
- AUC: mean=0.5759, std=0.0956
- Per-class recall mean: [0.4557, 0.3846, 0.3237]
- Verdict: rejected as final normalization stage because LOSO Macro F1 is lower than before-log (0.3476 vs 0.3642), even though recall is somewhat more balanced. Keep `PER_SUBJECT_NORMALIZE_STAGE="before_log"`.

## Phase 3 - Priority 5: LOSO Cross-Validation

Run timestamp: 2026-07-11 01:24
- Preprocessing: PER_SUBJECT_NORMALIZE=True, PER_SUBJECT_NORMALIZE_STAGE=before_log
- Class weights: mode=manual, values=[1.0, 1.1, 1.3]
- Folds run: 26
- Accuracy: mean=0.3737, std=0.0980
- Macro F1: mean=0.3329, std=0.1126
- AUC: mean=0.5495, std=0.1070
- Per-class recall mean: [0.2826, 0.4530, 0.3761]
- Verdict: rejected. High recall improved, but LOSO Macro F1 fell from 0.3642 to 0.3329 and AUC fell from 0.6011 to 0.5495. Stop class-weight tuning after this single targeted attempt.

## Locked Model

Run timestamp: 2026-07-11 08:44. Frozen checkpoint: `outputs\phase3_per-subject_normalization_before_log.pt`. Reference commit: `20c96b2533bc60f4ef5e8adfd7272046b1283282`.

**Final configuration (frozen — Parts 2-4 consume this as a fixed artifact):**
- Model: Hybrid CNN-LSTM (MLP encoders + attention/gated fusion), `FUSION_TYPE="attention"`, `LOSS_FN="weighted_ce"`.
- Preprocessing: `PER_SUBJECT_NORMALIZE=True`, `PER_SUBJECT_NORMALIZE_STAGE="before_log"`, EEG log1p + train-only RobustScaler, fNIRS signed-log + train-only StandardScaler, balanced train-fold class weights.
- Dataset: `data/tuberlin_eeg_fnirs.csv` (654 full-trial samples, 26 subjects, 180 EEG + 468 HbO features), subject-grouped splits.

**Final metrics:**

| Evaluation | Accuracy | Macro F1 | AUC | Per-class recall [L,M,H] |
|---|---:|---:|---:|---|
| Fixed split (Hybrid) | 0.4267 | 0.4064 | 0.6305 | [0.5625, 0.5185, 0.1875] |
| LOSO (Hybrid, 26 folds) | 0.3982 +/- 0.0742 | 0.3642 +/- 0.0894 | 0.6011 +/- 0.0801 | [0.5919, 0.3248, 0.2853] |
| LOSO (fNIRS-only, 26 folds) | 0.4297 +/- 0.0824 | 0.4126 +/- 0.0865 | 0.6117 +/- 0.0854 | [0.5721, 0.4359, 0.2810] |
| LOSO (Hybrid, no per-subject norm) | 0.3485 +/- 0.0814 | 0.2841 +/- 0.0913 | 0.5522 +/- 0.0980 | [0.4578, 0.3803, 0.2019] |
| LOSO (Hybrid+RF ensemble) | 0.4518 +/- 0.0810 | 0.4431 +/- 0.0797 | 0.6334 +/- 0.0749 | [0.5433, 0.4744, 0.3339] |

**Statistical significance (paired across 26 LOSO folds):**

| Comparison | mean diff | paired t p | Wilcoxon p |
|---|---:|---:|---:|
| Hybrid vs fNIRS-only (Macro F1) | -0.0484 | 0.0556 | 0.0435 |
| Hybrid vs fNIRS-only (AUC) | -0.0106 | 0.6278 | 0.7509 |
| Normalized vs Non-normalized Hybrid (Macro F1) | +0.0801 | 0.0047 | 0.0039 |
| Normalized vs Non-normalized Hybrid (AUC) | +0.0489 | 0.0587 | 0.1574 |

**Ensembling verdict:** ADOPTED. ensemble LOSO Macro F1=0.4431 > hybrid 0.3642; min mean class recall=0.334. The Hybrid+RF probability-average ensemble is adopted as the locked prediction head; the frozen Hybrid weights are unchanged and the RF is layered on top at inference.

Per-fold values: `outputs/loso_per_fold.csv`. Significance: `outputs/significance_tests.json`. Ensembling: `outputs/ensemble_results.json`.

## Future Work

- **Cross-dataset inference.** The demo's CSV-upload feature accepts only
  already-extracted features in the TU Berlin schema (180 EEG + 468 HbO). Running
  the model on a different EEG+fNIRS dataset — e.g. the Li et al. Stroop
  EEG+fNIRS dataset — would first require building a dataset-specific feature
  extraction pipeline that reproduces the TU Berlin feature definitions (band
  powers, HbO shape descriptors) from that dataset's raw signals, then matching
  channel/montage differences. This is real, scoped work; the current upload
  path deliberately does not attempt it, and rejects mismatched schemas rather
  than silently producing meaningless predictions.

## Classical Model Sweep (XGBoost, SVM) — tried, ruled out

Motivation: classical models repeatedly matched or beat deep changes here
(fNIRS-only > Hybrid on LOSO; RF is what lifts the ensemble past fNIRS-only), so
we checked whether a stronger classical component than plain Random Forest
exists. Same 26-fold LOSO protocol, same per-subject-normalized features and
per-fold train-only scalers as the RF ensemble. Artifact:
`outputs/classical_sweep.json`; script `classical_sweep.py`.

**Standalone LOSO (26 folds):**

| Model | Macro F1 | Accuracy | AUC | Recall [L, M, H] |
|---|---:|---:|---:|---|
| Random Forest (baseline) | 0.4365 +/- 0.0898 | 0.4467 | 0.6187 | [0.447, 0.538, 0.344] |
| XGBoost | 0.4379 +/- 0.0960 | 0.4421 | 0.6191 | [0.481, 0.453, 0.390] |
| SVM (RBF) | 0.3649 +/- 0.0726 | 0.4175 | 0.6267 | [0.527, 0.598, 0.104] |

**Significance vs RF (paired across 26 folds):**

| Comparison | mean diff | paired t p | Wilcoxon p | Verdict |
|---|---:|---:|---:|---|
| XGBoost vs RF | +0.0014 | 0.926 | 0.989 | ties RF — not significant |
| SVM vs RF | -0.0716 | <0.001 | 0.001 | significantly worse |

XGBoost is statistically indistinguishable from Random Forest (+0.0014 macro F1,
p≈0.93); SVM is significantly worse and nearly collapses on High load
(recall 0.10). Per the sweep protocol, no model **clearly** beat RF, so none was
adopted.

**Ensemble check (reported beyond the gate, for completeness):** swapping XGBoost
for RF in the Hybrid ensemble gives hybrid+XGB macro F1 0.4440 vs the locked
hybrid+RF **0.4431** — a +0.0009 difference, ~1/100th of the fold-to-fold std
(0.09), i.e. noise. The 3-way hybrid+RF+XGB is worse (0.4368).

**Conclusion:** no configuration meaningfully beats the locked Hybrid+RF ensemble;
Random Forest remains its classical component and nothing in serving changes. That
Random Forest is this hard to beat is itself the finding — further evidence this
task sits closer to a difficulty ceiling than to a modeling-choice problem.

## HbO-only vs. HbO+HbR Ablation — resolved

Run timestamp: 2026-07-14. Scripts: `hbr_ablation.py` (LOSO ablation),
`prototype_936_serving.py` (936 serving prototype), `compare_936_vs_live.py`
(served-split comparison). Artifacts: `outputs/hbr_ablation.json`,
`outputs/compare_936_vs_live.json`, `serving_936/`.

**Open question.** The Locked Model uses EEG (180) + HbO-only (468) fNIRS
features. The other 468 fNIRS features (HbR / deoxy-hemoglobin) were dropped at
some point between the early classical baselines — which used the full HbO+HbR
936-feature fNIRS set (see the "Classical Baseline Run" sections above, and the
note at the `2026-07-08 09:40` run that the config was later "corrected to
include HbR") — and the frozen deep Locked Model, which reverted Modality 2 to
HbO-only (`config.py`: `MODALITY2_COLUMNS = _hbo_cols`). **No empirical
cross-subject justification for that drop was ever recorded.** It was made worse
by an RF feature-importance check (see "Priority -1 Stability Fixes") showing HbR
total importance (0.4331) slightly *above* HbO (0.3998), suggesting HbR might
carry at least as much signal. This section resolves the question by direct
ablation rather than assumption.

**Method.** Same 26-fold LOSO protocol, same per-subject-normalized features and
per-fold train-only scalers used everywhere else. Only the fNIRS feature set
differs (468 HbO vs 936 HbO+HbR); the Hybrid architecture is unchanged.
Evaluated for both the Hybrid network alone and the Hybrid+RF ensemble (the
served config), with paired significance (HbO+HbR vs HbO-only).

**LOSO results (26 folds):**

| Config | Hybrid Macro F1 | Hybrid Acc | Hybrid AUC | Ensemble Macro F1 | Ensemble Acc | Ensemble AUC | Ensemble recall [L,M,H] |
|---|---:|---:|---:|---:|---:|---:|---|
| EEG + HbO (468, locked) | 0.3642 | 0.3982 | 0.6011 | 0.4431 | 0.4518 | 0.6334 | [0.543, 0.474, 0.334] |
| EEG + HbO + HbR (936) | 0.4508 | 0.4720 | 0.6600 | 0.4889 | 0.5013 | 0.6946 | [0.598, 0.543, 0.356] |

**Significance (paired across the same 26 folds, HbO+HbR vs HbO-only, macro F1):**

| Comparison | mean diff | paired t p | Wilcoxon p | Verdict |
|---|---:|---:|---:|---|
| Hybrid-alone | +0.0866 | 0.0019 | 0.0020 | HbR significantly helps (both tests) |
| **Ensemble (served config)** | +0.0458 | 0.0354 | **0.0594** | split — t-test significant, Wilcoxon not |

**Served-split cross-check.** A 936 serving prototype (`serving_936/`, retrained
Hybrid on the same frozen recipe + HbR, RF re-fit) was loaded through the real
backend `ModelService` alongside the live 468 ensemble and evaluated on the fixed
150-window demo split (`compare_936_vs_live.json`):

| Serving config | Macro F1 | Accuracy | ECE | Mean confidence | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Live 468 (EEG + HbO) | 0.3909 | 0.400 | 0.1223 | 0.435 | 90.1 |
| Prototype 936 (EEG + HbO + HbR) | 0.4036 | 0.420 | 0.0918 | 0.388 | 90.6 |
| Δ (936 − 468) | +0.0128 | +0.020 | −0.0306 | −0.047 | +0.5 |

On the actual served split the gain shrinks to +0.013 macro F1 / +0.02 accuracy
(calibration does improve: ECE −0.031). This also confirms the backend can serve
936-feature artifacts end-to-end, should re-locking ever be revisited.

**Decision: Case B — keep the HbO-only Locked Model.** Adding HbR clearly and
significantly helps the **Hybrid network alone** (+0.087 macro F1, both tests
p≈0.002). But the decision gate is governed by the **ensemble**, since the
ensemble is the served model. There the improvement is real but *not robustly*
significant: the paired t-test clears p<0.05 (0.035) while the non-parametric
Wilcoxon does not (0.059), and on the real served split the gain is only +0.013
macro F1. Per the ablation protocol — default to HbO-only unless the
ensemble-level result *clearly* favors HbO+HbR — a split significance result plus
a near-noise served-split delta is not "clear." HbR's individual RF importance
therefore did **not** translate into a significant, robust ensemble-level
improvement; **HbO-only is retained for parsimony, confirmed via direct ablation
rather than assumption.** No changes were made to the model, serving bundle,
frontend, or backend; the live `serving/` directory is untouched. The 936
prototype (`serving_936/`) and its scripts are kept as reproducible evidence, not
promoted to production.

**Future-work note.** The ensemble result sits right on the significance
boundary and the Hybrid-alone improvement is unambiguous, so restoring HbR is a
credible future direction if the served head is ever re-derived — it would be a
bounded re-run of the full lock (config → Part 1 rigor → classical sweep →
re-export → backend/frontend schema for 180 EEG + 936 fNIRS), not a hot-swap.
Treated here as documented future work, not an implemented change.

## Protocol Taxonomy & Benchmark

Run timestamp: 2026-07-14 20:30. Script: `protocol_benchmark.py`. Artifact: `outputs/protocol_benchmark.json`.

Published EEG/fNIRS workload accuracies of 60-90% are usually a *different, easier* problem than this project's **3-class, leave-one-subject-out (LOSO)** setup: they are often **binary** (0- vs 3-back / task-vs-rest) and/or **within-subject** (the same person appears in train and test, letting subject-specific physiological baselines leak). To show the gap is a *protocol* difference and not a modeling failure, the SAME features and the SAME RandomForest were run across two protocol axes. `Within-subject` = StratifiedKFold(5, shuffled); `LOSO` = LeaveOneGroupOut on subject.

**Axis 1 — subject-leakage (3-class), with vs without per-subject normalization:**

| Preprocessing | Within-subject (Macro F1) | LOSO (Macro F1) | Gap (within − LOSO) |
|---|---:|---:|---:|
| no per-subject norm (raw) | 0.3931 +/- 0.0260 | 0.4331 +/- 0.0927 | -0.0399 |
| per-subject norm (locked) | 0.4104 +/- 0.0451 | 0.4173 +/- 0.0720 | -0.0069 |

A notable, honest finding: on this dataset and estimator the within-subject vs LOSO gap is **small in both cases** (|gap| <= 0.04 Macro F1), and without normalization within-subject CV is if anything slightly *lower* than LOSO. So the subject-baseline leakage that flatters much within-subject literature is a minor effect here — the project's LOSO figure is *robust*, not a number propped up by cross-subject leakage. (Per-subject normalization is retained for other reasons documented in the Locked Model section; it is not what makes the LOSO number trustworthy on this axis.)

**Axis 2 — task difficulty (per-subject normalized), 3-class vs binary:**

| Task | Within-subject (Macro F1) | LOSO (Macro F1) |
|---|---:|---:|
| 3-class (L/M/H) | 0.4104 | 0.4173 |
| binary (Low vs High) | 0.6209 | 0.6177 |

This is the axis that actually explains the apparent gap to headline literature numbers. Collapsing to a binary task lifts LOSO Macro F1 by roughly +0.20 (from 0.42 to 0.62). Much of the 60-90% literature is binary and/or fewer-class; this project reports the harder 3-class task. Taken together: once you demand the honest 3-class, cross-subject (LOSO) protocol, ~0.4-0.45 Macro F1 is the *hard-problem* figure and is not directly comparable to easier-protocol accuracies. This is a scholarship point, not an excuse — the project deliberately reports the protocol that reflects real deployment on unseen people.

## Subject-Invariant Learning (DANN)

Run timestamp: 2026-07-14 20:57. Scripts: `dann.py`, `dann_ablation.py`. Artifact: `outputs/dann_ablation.json`. Writeup: `NOVELTY_DANN.md`.

Domain-adversarial training (gradient-reversal subject discriminator) directly attacks this project's documented ceiling — subject variability — by making the fused representation predict cognitive LOAD but not WHO. Same 26-fold LOSO protocol, same per-subject-normalized features and per-fold train-only scalers; baseline and DANN trained on identical folds for a paired comparison.

**LOSO results (26 folds):**

| Config | Hybrid Macro F1 | Hybrid AUC | Ensemble Macro F1 | Ensemble AUC |
|---|---:|---:|---:|---:|
| Baseline (locked) | 0.3924 | 0.5820 | 0.4520 | 0.6203 |
| DANN | 0.3753 | 0.5757 | 0.4545 | 0.6321 |

**Paired significance (DANN vs baseline, macro F1):**

| Comparison | mean diff | paired t p | Wilcoxon p |
|---|---:|---:|---:|
| Hybrid-alone | -0.0171 | 0.6274 | 0.6186 |
| Ensemble (served) | +0.0025 | 0.8816 | 0.8613 |

**Subject-invariance probe** (can a fresh classifier still recover subject identity from the learned features? lower = more invariant):

- Baseline probe accuracy: 0.091
- DANN probe accuracy: 0.098
- Chance: 0.050

**Lambda sensitivity sweep** (DANN Hybrid Macro F1 vs adversarial strength; all points share the same DANN training path, so this is the *controlled* on/off comparison):

| lambda_max | 0.0 (adversary off) | 0.5 | 1.0 | 2.0 |
|---|---:|---:|---:|---:|
| Hybrid Macro F1 | 0.4177 | 0.3933 | 0.3753 | 0.3622 |

**Decision: NOT ADOPTED.** DANN did not deliver a clear, significant ensemble-level improvement under the decision gate (ensemble diff +0.0025, t p=0.88, Wilcoxon p=0.86); the locked model is retained.

**Why DANN doesn't help here — a coherent, honest finding.** Three independent pieces of evidence line up:
1. The `Protocol Taxonomy & Benchmark` section shows subject-baseline leakage is a *minor* effect on this dataset (within-subject ≈ LOSO).
2. The subject-invariance probe shows the **baseline** representation is already close to chance for subject-identity recovery (0.091 vs 0.050 chance) — there is very little decodable subject-identity information left for the adversary to remove.
3. The lambda sweep degrades **monotonically** as adversarial pressure rises (0.42 → 0.36), i.e. the adversary injects noise rather than removing a harmful confound.

Together these say the cross-subject difficulty on this dataset is **not** driven by a decodable subject-identity confound in the learned features (per-subject normalization already handled the main baseline leak); it reflects genuine task difficulty / limited transferable signal. This is a legitimate negative result: a mainstream cross-subject method, applied rigorously with a paired-significance gate and a mechanism probe, is correctly declined — and the *reason* is consistent with the rest of the project's findings. (Note: the main-run baseline uses the standard `train_model` path while DANN uses `train_dann`; the cleanest adversary-on-vs-off comparison is the lambda sweep above, which shares one code path and reaches the same conclusion.)
