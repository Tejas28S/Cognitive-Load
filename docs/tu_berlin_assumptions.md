# TU Berlin hard-coded assumptions — audit inventory

*Step 1 of the universal dataset parser work. Generated from a direct read of the
codebase on 2026-09-02 (branch `updated-frontend`).*

This is the inventory of every place the pipeline assumes it is looking at the
**TU Berlin / Shin n-back EEG+fNIRS dataset**. Each entry is what would have to
change (or be normalized away) for a different dataset to flow through.

---

## 0. Summary of the binding constraint

The single most important finding: **the served model's feature vector identifies
channels by position, not by electrode name.**

| Layer | Shape | Channel identity |
|---|---|---|
| Served feature vector | 540 EEG + 936 fNIRS = 1476 floats | `eeg_ch00..ch29`, `hbo_ch00..ch35`, `hbr_ch00..ch35` |
| Model input (`serving/model_config.json`) | `n_eeg_features: 540`, `n_fnirs_features: 936` | fixed, non-negotiable without retraining |

`ch00..ch29` is *not* an arbitrary index — it is the raw acquisition order of the
TU Berlin recording, recoverable from `clab` in the raw `.mat` files, because
`preprocess_tuberlin.load_eeg()` reads `cnt['x']` with **no channel selection or
reordering** ([preprocess_tuberlin.py:196-197](../preprocess_tuberlin.py#L196-L197)).
Verified against `data/EEG_01-26_MATLAB/VP001-EEG/cnt_nback.mat`:

```
ch00..ch29 = Fp1, AFF5h, AFz, F1, FC5, FC1, T7, C3, Cz, CP5, CP1, P7, P3, Pz,
             POz, O1, Fp2, AFF6h, F2, FC2, FC6, C4, T8, CP2, CP6, P4, P8, O2,
             HEOG, VEOG          (30 channels @ 200 Hz)
```

This is what makes cross-dataset normalization *possible at all* — the positional
indices can be given electrode identities, so another dataset's named electrodes
can be projected onto them. It is also what makes it **lossy**: a dataset without
a given electrode has no value to contribute to that position.

The fNIRS side has no equivalent rescue. `hbo_ch00..ch35` are optode-pair indices
with no recoverable montage metadata in the serving bundle, so cross-dataset fNIRS
channel identity is **positional guesswork** unless a per-dataset montage mapping
is supplied by hand.

---

## 1. Serving / inference path

### `serving/feature_schema.json`
| Assumption | Detail |
|---|---|
| `eeg_columns` | 540 exact names, `eeg_ch{00-29}_{delta,theta,alpha,beta,gamma,engage}_{mean,std,slope}` |
| `fnirs_columns` | 936 exact names, `{hbo,hbr}_ch{00-35}_{13 stats}` |
| `label_column` | `"Cognitive_Load"` |
| `subject_column` | `"subject_id"` |
| `class_names` | `["Low (0-back)", "High (3-back)"]` — n-back task semantics baked into the label text |
| `preprocessing` | order `nan_to_num → per_subject_zscore(before_log) → log → saved_scalers`; `eeg_log: log1p_clip_nonneg`, `fnirs_log: signed_log1p`, `per_subject_eps: 1e-08` |

### [backend/upload_service.py](../backend/upload_service.py)
| Line | Assumption |
|---|---|
| [L31-39](../backend/upload_service.py#L31-L39) | Requires `feature_schema.json` keys `eeg_columns`, `fnirs_columns`, `label_column`, `subject_column`, `preprocessing` |
| [L45](../backend/upload_service.py#L45) | `self.required = eeg_cols + fnirs_cols` — all 1476 columns are mandatory |
| [L61](../backend/upload_service.py#L61) | Exact string match on column names. No aliasing, no case-insensitivity, no unit-suffix tolerance |
| [L96-97](../backend/upload_service.py#L96-L97) | `df[self.eeg_cols]` — selects by exact name (order-tolerant, name-intolerant) |
| [L103-110](../backend/upload_service.py#L103-L110) | Falls back to a single `__ALL__` group when `subject_id` is absent; flagged as degraded |
| [L112-118](../backend/upload_service.py#L112-L118) | Per-subject z-score then log transforms — the exact TU Berlin final config |
| [L121-122](../backend/upload_service.py#L121-L122) | `eeg_scaler` / `fnirs_scaler` fitted on TU Berlin training data (`feature_scaler.pkl`) |

**Net effect:** an upload is accepted only if it is already in TU Berlin feature space.

### [backend/model_service.py](../backend/model_service.py)
| Line | Assumption |
|---|---|
| [L127](../backend/model_service.py#L127) | `self.n_eeg = cfg["n_eeg_features"]` = 540 |
| [L151-152](../backend/model_service.py#L151-L152) | Splits the flat vector at index 540 into EEG / fNIRS branches |
| [L24-25](../backend/model_service.py#L24-L25) | `ABSTAIN_CONFIDENCE`/`ABSTAIN_MARGIN` tuned to the 3-class LOSO operating point |

### [backend/mat_extractor.py](../backend/mat_extractor.py) — raw `.mat` upload path
| Line | Assumption |
|---|---|
| [L9-12](../backend/mat_extractor.py#L9-L12) | `FNIRS_FS = 10`, `EEG_FS = 200`, `WINDOW_SEC = 5.0`, `STRIDE_SEC = 1.0` (actual `fs` *is* read from the file at L110/L131, so these constants are mostly vestigial — but the window/stride are real) |
| [L99-106](../backend/mat_extractor.py#L99-L106) | Requires a `cnt_*`-prefixed struct key — BBCI/TU Berlin convention |
| [L108-110](../backend/mat_extractor.py#L108-L110) | `cnt_nirs['oxy'][0,0]['x'][0,0]` / `['deoxy']` — TU Berlin NIRS struct layout, **already in HbO/HbR space** |
| [L125-131](../backend/mat_extractor.py#L125-L131) | `cnt_eeg['x']`, `cnt_eeg['fs']` — TU Berlin EEG struct layout |
| [L111-112](../backend/mat_extractor.py#L111-L112) | fNIRS bandpass 0.01–0.1 Hz |
| [L132](../backend/mat_extractor.py#L132) | EEG bandpass 0.5–45 Hz |
| [L86-94](../backend/mat_extractor.py#L86-L94) | Column naming: `eeg_ch{NN}_{band}_{stat}`, `{hbo,hbr}_ch{NN}_{stat}` — **positional channel index, no electrode name** |
| [L179-180](../backend/mat_extractor.py#L179-L180) | `subject_id = 'LIVE_MAT'`, `trial_id = 'LIVE_MAT_001'` hard-coded |

**Note:** the feature *math* here (`eeg_features`, `fnirs_features`, `bandpass`,
`baseline_correct`) is channel-count agnostic — it loops over `trial.shape[1]`.
Only the *file parsing* and the *column naming* are TU Berlin-specific. This is
the reusable core.

### [backend/main.py](../backend/main.py)
| Line | Assumption |
|---|---|
| [L57](../backend/main.py#L57) | `_BUNDLES = {"binary": "serving", "3class": "serving_3class"}` — fixed bundle registry |
| [L303-376](../backend/main.py#L303-L376) | `/api/predict/upload` — CSV must already be TU Berlin feature space |
| [L378-445](../backend/main.py#L378-L445) | `/api/predict/upload_mat` — requires a TU Berlin raw `.mat` pair (EEG + NIRS) |

---

## 2. Training path

### [config.py](../config.py)
| Line | Assumption |
|---|---|
| [L30](../config.py#L30) | `_CSV = "data/tuberlin_sliding_augmented.csv"` |
| [L31-34](../config.py#L31-L34) | Column detection by literal prefix `eeg_`, `hbo_`, `hbr_` |
| [L52](../config.py#L52) | `LABEL_COLUMN = "Cognitive_Load"` |
| [L53-54](../config.py#L53-L54) | `SUBJECT_COLUMN = "subject_id"`, `SPLIT_GROUP_COLUMNS = ["subject_id"]` |
| [L59-63](../config.py#L59-L63) | `CLASS_NAMES` are n-back task labels (`"Low (0-back)"`, `"Medium (2-back)"`, `"High (3-back)"`) — task-specific, not transferable label text |
| [L68-69](../config.py#L68-L69) | `EEG_FEATURE_PREFIX`, `FNIRS_FEATURE_PREFIX` |

### [data_loader.py](../data_loader.py)
| Line | Assumption |
|---|---|
| [L96-104](../data_loader.py#L96-L104) | Prefix-based column detection, with a numeric-suffix sort key |
| [L167](../data_loader.py#L167) | Transform mask hard-codes `c.lower().startswith("eeg_")` |
| [L188](../data_loader.py#L188) | Transform mask hard-codes `c.lower().startswith(("hbo_", "hbr_"))` |
| [L303-307](../data_loader.py#L303-L307) | `LABEL_COLUMN` must exist |
| [L309](../data_loader.py#L309) | Subject column defaults to `"subject_id"` |
| [L335-351](../data_loader.py#L335-L351) | `BINARY_CLASSIFICATION` assumes labels are `{0,1,2}` with `1 = Medium = 2-back`; drops class 1 and remaps `2 → 1`. **Pure n-back semantics.** |

---

## 3. What a new dataset must supply

Derived from the above, to reach a prediction from the existing frozen model a
dataset must ultimately produce:

1. **EEG**: 30 channels, in TU Berlin `clab` order, band-power features
   (delta/theta/alpha/beta/gamma + engagement ratio) × (mean/std/slope) over
   2s sub-windows of a 5s window → 540 values.
2. **fNIRS**: 36 HbO + 36 HbR channels (i.e. already converted out of raw optical
   density into hemoglobin concentrations), 13 time-domain stats each → 936 values.
3. **Metadata**: `subject_id` (required for the per-subject z-score to be
   meaningful), optionally `Cognitive_Load` labels for scoring.

### Coverage reality check for the Stroop dataset

| | TU Berlin (model expects) | Stroop (supplies) | Direct overlap |
|---|---|---|---|
| EEG channels | 30 @ 200 Hz | 34 @ 1000 Hz | **16 exact** (Fp1, Fp2, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2, HEOG≈HEO, VEOG≈VEO); ~10 more reachable by documented nearest-neighbour substitution (F3→F1, FC3→FC1, CP3→CP1, and mirrors) |
| fNIRS channels | 36 HbO + 36 HbR @ 10 Hz | 20 channels × 2 wavelengths (785/850 nm) @ 100 Hz | **0 without a montage mapping** — Stroop ships raw optical intensity, needs modified Beer-Lambert conversion first, and covers frontal cortex only vs TU Berlin's whole-head layout |

**Consequence to carry into Steps 4–5:** the normalizer can legitimately map the
Stroop EEG side onto the canonical schema with a reported ~53–87% coverage
(exact-only vs. with neighbour substitution), but the fNIRS side cannot be mapped
onto TU Berlin optode positions in any defensible way. Per the project's own
"explicit errors over silent guessing" constraint, the pipeline must **report
coverage and refuse to emit a confident prediction below threshold** rather than
zero-filling the 936 missing fNIRS features and returning a confident-looking
number. Zero-filling would produce a prediction, but a meaningless one.

The scientifically valid route for Stroop end-to-end prediction is retraining on
Stroop's own feature space (a new serving bundle), which the canonical schema and
alias registry built in Steps 2–4 also enable — the ingestion layer is what makes
either route possible.
