# Binary Model Improvement — Measured Results

> Generated from run artifacts by `research/report.py`. Every number below
> is read from `outputs/runs/run_20260820_141126_19cb/`; none is transcribed by hand.
> Regenerate with `.venv/bin/python -m research.report outputs/runs/run_20260820_141126_19cb`.

**Run** `run_20260820_141126_19cb` · tag `resume-s9-s10-final` · created 2026-08-20T08:41:26.335633+00:00  
**Commit** `12b29b33c2ce` on `binary-class` (working tree dirty) · torch 2.13.0 · seed 42

Methodology: [`explanation/BINARY_MODEL_IMPROVEMENT.md`](explanation/BINARY_MODEL_IMPROVEMENT.md).

> **Campaign spans two runs.** The search was interrupted and resumed, so
> the experiment arms below are pooled from `run_20260819_184443_f0d1` and `run_20260820_141126_19cb`. Both used the same seed, the
> same development/lockbox split and the same fold generation, so the arms
> are directly comparable — the baseline arm was re-run in the second run
> and reproduced the first run's result exactly. All FINAL evaluations
> below come from the second run.

---

## 1. Data and protocol

- **Dataset**: `sliding_10s_5s` — 4,576 windows, 26 subjects
- **Task**: Low vs High (0-back vs 3-back); chance = 0.5000
- **Class balance**: Low = 2,288, High = 2,288
- **Preprocessing**: `nan_to_num -> per_subject_norm -> log -> scaler`; per-subject normalization `transductive` (before_log)
- **Selection protocol**: 18-fold LOSO on development subjects — subjects [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 24, 25]
- **Lockbox** (never used for any selection decision): [6, 8, 16, 18, 19, 21, 22, 26]
- **Headline protocol**: 26-fold LOSO over all subjects
- **Checkpoint metric**: validation macro F1

## 2. Leakage audit

Dataset-level checks (fold-level checks ran before every one of the 42 experiments' folds and are stored per experiment):

| Check | Result | Detail |
|---|---|---|
| `per_subject_norm_is_subject_local` | PASS | subject 1 recomputed in isolation: max|delta| eeg=0.00e+00 fnirs=0.00e+00 (tol 1e-05) |
| `per_subject_norm_is_subject_local` | PASS | subject 2 recomputed in isolation: max|delta| eeg=0.00e+00 fnirs=0.00e+00 (tol 1e-05) |
| `per_subject_norm_is_subject_local` | PASS | subject 3 recomputed in isolation: max|delta| eeg=0.00e+00 fnirs=0.00e+00 (tol 1e-05) |
| `per_subject_norm_is_subject_local` | PASS | subject 4 recomputed in isolation: max|delta| eeg=0.00e+00 fnirs=0.00e+00 (tol 1e-05) |
| `no_duplicate_rows_in_dataset` | PASS | all 4576 feature rows unique |
| `features_finite` | PASS | preprocessed features are finite |

Every fold additionally asserted: row-index disjointness, subject disjointness, trial disjointness, sliding-window identity disjointness, no byte-identical feature row across splits, scalers fitted on training rows only, the held-out subject absent from train and val, finite features, and valid labels. A failure raises `LeakageError` and aborts the run, so a leaked score is never produced.

The negative tests in `research/test_leakage.py` inject each leakage type deliberately and confirm the audit catches it — an audit that never fires would prove nothing.

## 3. Baseline

The served binary architecture (`serving/model_config.json`) with the training hyperparameters in `config.py`, recorded before anything was changed.

| Protocol | Macro F1 | Accuracy | Balanced acc | ROC-AUC | MCC | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| Development LOSO (selection) | 0.6451 ± 0.078 | 0.6537 ± 0.075 | 0.6537 ± 0.075 | 0.7174 ± 0.107 | 0.3238 ± 0.158 | 0.3074 ± 0.151 |
| LOSO, all subjects | 0.6255 ± 0.104 | 0.6309 ± 0.101 | 0.6309 ± 0.101 | 0.6775 ± 0.129 | 0.2675 ± 0.206 | 0.2618 ± 0.202 |
| Lockbox (unseen subjects) | 0.6275 ± 0.093 | 0.6371 ± 0.085 | 0.6371 ± 0.085 | 0.6888 ± 0.111 | 0.2816 ± 0.169 | 0.2741 ± 0.170 |

**Baseline generalization gap** (macro F1 at each fold's selected epoch):

| train | validation | held-out | train − held-out | mean best epoch |
|---:|---:|---:|---:|---:|
| 0.8001 | 0.6360 | 0.6255 | 0.1746 | 14.5 |

## 4. Experiments

42 arms were trained and evaluated on identical development folds. Each stage varied one axis from the incumbent; the winner was carried forward. Losing arms are kept — they are the negative results.

### Stage decisions

| Stage | Winner | Accepted? | Score before | Score after | Δ |
|---|---|:--:|---:|---:|---:|
| `S9_dann` | `A0_baseline` | no | 0.6451 | 0.6451 | 0.0000 |
| `S10_data` | `A0_baseline` | no | 0.6451 | 0.6451 | 0.0000 |

### Full leaderboard — development LOSO macro F1

Sorted best to worst. `gap` is train − held-out macro F1 (higher = more overfitting). `Δ vs base` and `p` are paired against the baseline on the same folds.

| Experiment | Stage | Macro F1 | Accuracy | ROC-AUC | MCC | Params | gap | Δ vs base | p | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `S5_proj192` | S5_capacity | 0.6611 ± 0.101 | 0.669 | 0.721 | 0.348 | 620,570 | 0.146 | 0.016 | 0.309 | projection width 192 |
| `S10_stride1s` | S10_data | 0.6566 ± 0.114 | 0.662 | 0.721 | 0.333 | 245,594 | 0.217 | 0.011 | 0.461 | dense windows: 10s / 1s stride (4.6x rows) |
| `S4_drop_low` | S4_dropout | 0.6537 ± 0.120 | 0.658 | 0.710 | 0.320 | 245,594 | 0.194 | 0.009 | 0.655 | dropout 0.3 / 0.3 |
| `S2_no_sched` | S2_schedule | 0.6474 ± 0.093 | 0.655 | 0.716 | 0.325 | 245,594 | 0.233 | 0.002 | 0.819 | no LR schedule (control) |
| `S5_proj32_enc32` | S5_capacity | 0.6466 ± 0.112 | 0.656 | 0.726 | 0.325 | 66,114 | 0.152 | 0.002 | 0.929 | narrow: proj 32 / enc 32 / head [32] |
| `A0_baseline` | S0_baseline | 0.6451 ± 0.078 | 0.654 | 0.717 | 0.324 | 245,594 | 0.163 | n/a | n/a | served binary architecture + config.py training hyperparameters |
| `S5_blocks1` | S5_capacity | 0.6433 ± 0.084 | 0.647 | 0.704 | 0.302 | 207,578 | 0.158 | -0.002 | 0.890 | 1 residual block |
| `S4_drop_mid` | S4_dropout | 0.6414 ± 0.104 | 0.645 | 0.697 | 0.298 | 245,594 | 0.245 | -0.004 | 0.804 | dropout 0.4 / 0.5 |
| `S7_md010` | S7_modality_dropout | 0.6396 ± 0.100 | 0.648 | 0.710 | 0.308 | 245,594 | 0.189 | -0.006 | 0.688 | modality dropout 0.1 / 0.1 |
| `S8_noise20` | S8_noise | 0.6380 ± 0.105 | 0.647 | 0.713 | 0.306 | 245,594 | 0.184 | -0.007 | 0.635 | input noise 0.20 |
| `S3_wd0` | S3_weight_decay | 0.6352 ± 0.110 | 0.651 | 0.712 | 0.321 | 245,594 | 0.144 | -0.010 | 0.494 | no weight decay |
| `S2_longer_p15` | S2_schedule | 0.6316 ± 0.100 | 0.642 | 0.702 | 0.299 | 245,594 | 0.214 | -0.014 | 0.107 | longer budget, patience 15 |
| `S2_longer_p25` | S2_schedule | 0.6316 ± 0.100 | 0.642 | 0.702 | 0.299 | 245,594 | 0.214 | -0.014 | 0.107 | longer budget, patience 25 |
| `S6_mlp_concat` | S6_architecture | 0.6294 ± 0.101 | 0.636 | 0.683 | 0.283 | 236,282 | 0.157 | -0.016 | 0.230 | MLP encoders, concat fusion (attention ablation) |
| `S2_cosine` | S2_schedule | 0.6294 ± 0.113 | 0.638 | 0.701 | 0.284 | 245,594 | 0.254 | -0.016 | 0.270 | cosine annealing |
| `S3_wd01` | S3_weight_decay | 0.6287 ± 0.107 | 0.640 | 0.709 | 0.294 | 245,594 | 0.207 | -0.016 | 0.140 | weight decay 0.1 |
| `S3_wd003` | S3_weight_decay | 0.6283 ± 0.098 | 0.637 | 0.701 | 0.287 | 245,594 | 0.166 | -0.017 | 0.296 | weight decay 0.003 |
| `S1_bs64_lr1e3` | S1_batch_lr | 0.6281 ± 0.122 | 0.634 | 0.694 | 0.278 | 245,594 | 0.236 | -0.017 | 0.292 | batch 64, lr 1e-3 |
| `S8_noise10` | S8_noise | 0.6187 ± 0.097 | 0.628 | 0.683 | 0.270 | 245,594 | 0.151 | -0.026 | 0.094 | input noise 0.10 |
| `S9_dann01` | S9_dann | 0.6182 ± 0.098 | 0.626 | 0.680 | 0.260 | 245,594 | 0.160 | -0.027 | 0.104 | DANN lambda 0.1 |
| `S10_win8s` | S10_data | 0.6160 ± 0.105 | 0.622 | 0.677 | 0.255 | 245,594 | 0.172 | -0.029 | 0.128 | 8s windows / 4s stride |
| `S7_md_eeg_only` | S7_modality_dropout | 0.6150 ± 0.091 | 0.625 | 0.676 | 0.259 | 245,594 | 0.175 | -0.030 | 0.172 | drop EEG more often (0.3 / 0.1) |
| `S10_norm_after_log` | S10_data | 0.6148 ± 0.099 | 0.624 | 0.684 | 0.260 | 245,594 | 0.173 | -0.030 | 0.050 | per-subject norm applied after the log |
| `S7_md020` | S7_modality_dropout | 0.6110 ± 0.117 | 0.628 | 0.697 | 0.264 | 245,594 | 0.157 | -0.034 | 0.209 | modality dropout 0.2 / 0.2 |
| `S5_proj48` | S5_capacity | 0.6061 ± 0.106 | 0.610 | 0.654 | 0.228 | 113,402 | 0.118 | -0.039 | 0.144 | projection width 48 |
| `S10_calibration_norm` | S10_data | 0.6049 ± 0.114 | 0.624 | 0.671 | 0.263 | 245,594 | 0.185 | -0.040 | 0.156 | per-subject norm from 30 calibration windows only |
| `S8_smooth` | S8_noise | 0.6048 ± 0.141 | 0.626 | 0.678 | 0.262 | 245,594 | 0.183 | -0.040 | 0.171 | label smoothing 0.1 |
| `S1_bs64` | S1_batch_lr | 0.5978 ± 0.120 | 0.616 | 0.677 | 0.252 | 245,594 | 0.115 | -0.047 | 0.019 | batch 64 |
| `S9_dann05` | S9_dann | 0.5968 ± 0.117 | 0.613 | 0.668 | 0.237 | 245,594 | 0.156 | -0.048 | 0.014 | DANN lambda 0.5 |
| `S1_bs128` | S1_batch_lr | 0.5864 ± 0.104 | 0.607 | 0.688 | 0.234 | 245,594 | 0.113 | -0.059 | 0.009 | batch 128 |
| `S3_wd005` | S3_weight_decay | 0.5846 ± 0.116 | 0.601 | 0.661 | 0.209 | 245,594 | 0.222 | -0.061 | 0.004 | weight decay 0.05 |
| `S5_blocks3` | S5_capacity | 0.5841 ± 0.100 | 0.606 | 0.652 | 0.235 | 283,610 | 0.135 | -0.061 | 0.011 | 3 residual blocks |
| `S8_noise0` | S8_noise | 0.5831 ± 0.094 | 0.601 | 0.652 | 0.217 | 245,594 | 0.134 | -0.062 | 0.025 | no input noise |
| `S10_no_hbr` | S10_data | 0.5781 ± 0.098 | 0.592 | 0.635 | 0.191 | 200,666 | 0.086 | -0.067 | 0.014 | HbO only (drop HbR) |
| `S9_dann10` | S9_dann | 0.5778 ± 0.110 | 0.598 | 0.663 | 0.213 | 245,594 | 0.125 | -0.067 | 0.012 | DANN lambda 1.0 |
| `S10_no_subject_norm` | S10_data | 0.5698 ± 0.106 | 0.589 | 0.635 | 0.190 | 245,594 | 0.125 | -0.075 | 0.014 | no per-subject normalization |
| `S6_cnn_res` | S6_architecture | 0.5550 ± 0.084 | 0.576 | 0.604 | 0.163 | 107,962 | 0.026 | -0.090 | 0.002 | CNN + residual blocks, concat fusion |
| `S1_lr1e4` | S1_batch_lr | 0.5547 ± 0.100 | 0.590 | 0.646 | 0.208 | 245,594 | 0.063 | -0.090 | 0.004 | lr 1e-4 |
| `S6_cnn_res_attn` | S6_architecture | 0.5467 ± 0.079 | 0.560 | 0.588 | 0.125 | 117,274 | 0.035 | -0.098 | 0.001 | CNN + residual + attention fusion |
| `S6_cnn_wide` | S6_architecture | 0.5467 ± 0.071 | 0.566 | 0.605 | 0.142 | 363,930 | 0.076 | -0.098 | 0.000 | CNN + residual + attention, 128 filters |
| `S4_drop_high` | S4_dropout | 0.5083 ± 0.127 | 0.565 | 0.602 | 0.150 | 245,594 | 0.040 | -0.137 | 0.000 | dropout 0.6 / 0.7 |
| `S6_cnn` | S6_architecture | 0.4786 ± 0.087 | 0.531 | 0.552 | 0.075 | 54,842 | 0.026 | -0.166 | 0.000 | 1D CNN over channel axis, concat fusion |

## 5. Selected model

`A0_baseline` — 245,594 parameters, mean best epoch 14.5.

```json
{
  "encoder": "mlp",
  "proj_dim": 96,
  "enc_dim": 48,
  "n_blocks": 2,
  "encoder_dropout": 0.5,
  "fusion": "attention",
  "fc_units": [
    48,
    24
  ],
  "fc_dropout": 0.6,
  "optimizer": "adamw",
  "lr": 0.0003,
  "weight_decay": 0.01,
  "batch_size": 16,
  "max_epochs": 35,
  "scheduler": "plateau",
  "lr_patience": 4,
  "lr_factor": 0.5,
  "early_stop_patience": 7,
  "augment_noise": 0.05,
  "class_weights": "balanced",
  "label_smoothing": 0.0,
  "modality_dropout_eeg": 0.0,
  "modality_dropout_fnirs": 0.0,
  "use_dann": false,
  "dann_lambda": 0.0
}
```

Data configuration:

```json
{
  "dataset": "sliding_10s_5s",
  "binary": true,
  "per_subject_norm": "transductive",
  "per_subject_norm_stage": "before_log",
  "drop_hbr": false
}
```

### Final results

| Protocol | Model | Macro F1 | Accuracy | Balanced acc | ROC-AUC | MCC | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| LOSO, all subjects | **selected** | 0.6255 ± 0.104 | 0.6309 ± 0.101 | 0.6309 ± 0.101 | 0.6775 ± 0.129 | 0.2675 ± 0.206 | 0.2618 ± 0.202 |
| LOSO, all subjects | baseline | 0.6255 ± 0.104 | 0.6309 ± 0.101 | 0.6309 ± 0.101 | 0.6775 ± 0.129 | 0.2675 ± 0.206 | 0.2618 ± 0.202 |
| Lockbox (unseen) | **selected** | 0.6275 ± 0.093 | 0.6371 ± 0.085 | 0.6371 ± 0.085 | 0.6888 ± 0.111 | 0.2816 ± 0.169 | 0.2741 ± 0.170 |
| Lockbox (unseen) | baseline | 0.6275 ± 0.093 | 0.6371 ± 0.085 | 0.6371 ± 0.085 | 0.6888 ± 0.111 | 0.2816 ± 0.169 | 0.2741 ± 0.170 |

**Paired significance, selected vs baseline on identical folds:**

- LOSO (all subjects): Δ 0.0000, paired t p=n/a, Wilcoxon p=n/a, 0W/0L
- Lockbox: Δ 0.0000, paired t p=n/a, Wilcoxon p=1.000, 0W/0L

### Generalization gap, before and after

| Model | train | validation | held-out | train − held-out | mean best epoch |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8001 | 0.6360 | 0.6255 | **0.1746** | 14.5 |
| selected | 0.8001 | 0.6360 | 0.6255 | **0.1746** | 14.5 |

### Full metric record (pooled held-out windows)

All 4,576 held-out predictions from the 26-fold LOSO over all subjects, pooled.

| Metric | Value |
|---|---:|
| accuracy | 0.6309 |
| balanced accuracy | 0.6309 |
| precision macro | 0.6323 |
| recall macro | 0.6309 |
| f1 macro | 0.6299 |
| precision weighted | 0.6323 |
| recall weighted | 0.6309 |
| f1 weighted | 0.6299 |
| precision micro | 0.6309 |
| recall micro | 0.6309 |
| f1 micro | 0.6309 |
| cohen kappa | 0.2618 |
| mcc | 0.2632 |
| roc auc macro ovr | 0.6707 |
| roc auc macro ovo | *one-vs-one is undefined for binary classification (identical to OVR)* |
| average precision macro | 0.6542 |
| log loss | 0.6792 |
| brier score | 0.4745 |
| top 3 accuracy | *Not applicable: K=3 is not smaller than the number of classes (2).* |
| top 5 accuracy | *Not applicable: K=5 is not smaller than the number of classes (2).* |

**Per class:**

| Class | Precision | Recall | F1 | Specificity | Support | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 0.6187 | 0.6823 | 0.6489 | 0.5795 | 2288 | 1561 | 962 | 727 | 1326 |
| High | 0.6459 | 0.5795 | 0.6109 | 0.6823 | 2288 | 1326 | 727 | 962 | 1561 |

**Confusion matrix** (rows = true, columns = predicted):

| | Low | High |
|---|---:|---:|
| **Low** | 1561 | 727 |
| **High** | 962 | 1326 |

### Per-subject held-out performance

| Subject | Windows | Accuracy | Macro F1 | Balanced acc | Kappa | MCC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| S17 | 176 | 0.807 | 0.807 | 0.807 | 0.614 | 0.614 | 0.891 |
| S3 | 176 | 0.807 | 0.803 | 0.807 | 0.614 | 0.638 | 0.909 |
| S7 | 176 | 0.756 | 0.756 | 0.756 | 0.511 | 0.512 | 0.804 |
| S25 | 176 | 0.756 | 0.753 | 0.756 | 0.511 | 0.521 | 0.849 |
| S16 | 176 | 0.722 | 0.718 | 0.722 | 0.443 | 0.456 | 0.798 |
| S10 | 176 | 0.705 | 0.705 | 0.705 | 0.409 | 0.409 | 0.748 |
| S9 | 176 | 0.688 | 0.686 | 0.688 | 0.375 | 0.378 | 0.774 |
| S22 | 176 | 0.676 | 0.675 | 0.676 | 0.352 | 0.355 | 0.722 |
| S11 | 176 | 0.676 | 0.674 | 0.676 | 0.352 | 0.358 | 0.747 |
| S24 | 176 | 0.676 | 0.671 | 0.676 | 0.352 | 0.365 | 0.752 |
| S1 | 176 | 0.670 | 0.670 | 0.670 | 0.341 | 0.341 | 0.751 |
| S15 | 176 | 0.676 | 0.667 | 0.676 | 0.352 | 0.373 | 0.711 |
| S6 | 176 | 0.659 | 0.649 | 0.659 | 0.318 | 0.338 | 0.707 |
| S18 | 176 | 0.648 | 0.645 | 0.648 | 0.295 | 0.299 | 0.678 |
| S13 | 176 | 0.648 | 0.643 | 0.648 | 0.295 | 0.303 | 0.771 |
| S20 | 176 | 0.636 | 0.632 | 0.636 | 0.273 | 0.280 | 0.679 |
| S23 | 176 | 0.608 | 0.599 | 0.608 | 0.216 | 0.227 | 0.586 |
| S8 | 176 | 0.574 | 0.566 | 0.574 | 0.148 | 0.153 | 0.561 |
| S4 | 176 | 0.562 | 0.561 | 0.562 | 0.125 | 0.126 | 0.595 |
| S26 | 176 | 0.523 | 0.522 | 0.523 | 0.045 | 0.046 | 0.526 |
| S21 | 176 | 0.517 | 0.499 | 0.517 | 0.034 | 0.037 | 0.546 |
| S2 | 176 | 0.494 | 0.491 | 0.494 | -0.011 | -0.012 | 0.494 |
| S19 | 176 | 0.511 | 0.485 | 0.511 | 0.023 | 0.026 | 0.556 |
| S5 | 176 | 0.483 | 0.482 | 0.483 | -0.034 | -0.034 | 0.483 |
| S14 | 176 | 0.477 | 0.475 | 0.477 | -0.045 | -0.046 | 0.519 |
| S12 | 176 | 0.449 | 0.431 | 0.449 | -0.102 | -0.109 | 0.455 |

Spread across subjects: macro F1 from 0.431 (S12) to 0.807 (S17) — the range across people is wider than any difference between the models compared above.

## 6. Feature importance (SHAP)

`shap.GradientExplainer` on this run's trained model, 160 held-out windows explained against 80 background samples.

| Modality | Share of total |SHAP| |
|---|---:|
| EEG | 0.0546 |
| fNIRS | 0.9454 |
| HbO | 0.4579 |
| HbR | 0.4875 |

**Top feature families:**

| Rank | Family | Modality | Features | Total mean \|SHAP\| |
|---:|---|---|---:|---:|
| 1 | HBR auc | fNIRS | 36 | 0.30160 |
| 2 | HBO max | fNIRS | 36 | 0.28242 |
| 3 | HBO min | fNIRS | 36 | 0.27163 |
| 4 | HBO mean | fNIRS | 36 | 0.26952 |
| 5 | HBR min | fNIRS | 36 | 0.25918 |
| 6 | HBR power | fNIRS | 36 | 0.25035 |
| 7 | HBO late | fNIRS | 36 | 0.24425 |
| 8 | HBR max | fNIRS | 36 | 0.24118 |
| 9 | HBO early | fNIRS | 36 | 0.24093 |
| 10 | HBR late | fNIRS | 36 | 0.23620 |
| 11 | HBR mean | fNIRS | 36 | 0.23597 |
| 12 | HBO auc | fNIRS | 36 | 0.20693 |
| 13 | HBR early | fNIRS | 36 | 0.19988 |
| 14 | HBO power | fNIRS | 36 | 0.19336 |
| 15 | HBR skew | fNIRS | 36 | 0.18659 |

**Top individual features:**

| Rank | Feature | Modality | Mean \|SHAP\| |
|---:|---|---|---:|
| 1 | `hbo_ch23_min` | fNIRS | 0.02237 |
| 2 | `hbr_ch24_max` | fNIRS | 0.02209 |
| 3 | `hbr_ch28_min` | fNIRS | 0.02174 |
| 4 | `hbo_ch24_power` | fNIRS | 0.01961 |
| 5 | `hbo_ch23_max` | fNIRS | 0.01936 |
| 6 | `hbr_ch06_skew` | fNIRS | 0.01888 |
| 7 | `hbr_ch13_auc` | fNIRS | 0.01881 |
| 8 | `hbr_ch28_mean` | fNIRS | 0.01860 |
| 9 | `hbr_ch12_min` | fNIRS | 0.01858 |
| 10 | `hbo_ch01_mean` | fNIRS | 0.01844 |
| 11 | `hbr_ch28_early` | fNIRS | 0.01834 |
| 12 | `hbr_ch21_min` | fNIRS | 0.01826 |
| 13 | `hbo_ch23_early` | fNIRS | 0.01818 |
| 14 | `hbr_ch28_late` | fNIRS | 0.01785 |
| 15 | `hbo_ch06_max` | fNIRS | 0.01779 |

## 7. Learned representations (t-SNE + subject-invariance probe)

All variants trained on the same development subjects and embedded on the same unseen subjects, so the comparison is controlled.

| Representation | Subject-probe accuracy | Chance | Reading |
|---|---:|---:|---|
| raw features | 0.013 ± 0.004 | 0.125 | near chance — subject identity largely absent |
| baseline model | 0.214 ± 0.019 | 0.125 | well above chance — subject identity still present |
| selected model | 0.214 ± 0.019 | 0.125 | well above chance — subject identity still present |
| dann model | 0.217 ± 0.014 | 0.125 | well above chance — subject identity still present |

Coordinates: `representations/tsne_*.csv`. Figures: `representations/*.png`. Browsable at `/evaluation` → *Learned representation*.

## 8. Artifacts

Everything above lives in `outputs/runs/run_20260820_141126_19cb/`:

| Path | Contents |
|---|---|
| `predictions/predictions.csv` | one row per held-out window: subject, trial, window, fold, true, predicted, probabilities, confidence |
| `metrics/` | metrics.json/.csv, per-class, LOSO, per-subject |
| `confusion_matrix/` | counts + figure |
| `curves/` | training/validation curves, ROC, PR (JSON + PNG) |
| `feature_importance/` | SHAP values, ranking, figures |
| `representations/` | t-SNE coordinates, embeddings, figures |
| `model/` | deployment model, per-fold checkpoints, scaler, schema, labels |
| `experiments/` | every arm's record, leaderboard.csv, stage_log.json |
| `logs/training.log` | full run log |

Reproduce the metrics from the predictions alone:

```bash
.venv/bin/python -m research.verify_run outputs/runs/run_20260820_141126_19cb
```

