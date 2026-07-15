# Cognitive Load Estimation from EEG + fNIRS

A hybrid deep-learning model that fuses **EEG** (electrical) and **fNIRS**
(hemodynamic) brain signals from the Tu Berlin open dataset to estimate mental
workload as **Low / Medium / High**, plus a demo web app (FastAPI + React) that
serves the final model (a frozen Hybrid network + Random Forest ensemble) and
its evaluation artifacts.

The headline result is as much methodological as it is a model: cross-subject
(leave-one-subject-out) performance is meaningfully lower than a fixed
train/test split suggests, because a fixed split leaks subject identity. Both
numbers are always reported together — see [`RESULTS.md`](RESULTS.md) and the
app's **Methodology** page.

## Repository structure

```
cognitive_load/
├── data/                     # raw + extracted feature CSVs (gitignored, large)
├── outputs/                  # training checkpoints + eval artifacts (gitignored)
├── models/architectures.py   # model definitions (training side)
├── config.py, data_loader.py, trainer.py, ...   # training pipeline
├── part1_rigor.py            # LOSO significance + ensembling (Part 1)
├── classical_sweep.py        # XGBoost/SVM vs RF sweep (ruled out; see RESULTS.md)
├── dann.py, dann_ablation.py # NOVEL: subject-invariant (DANN) training + LOSO ablation
├── protocol_benchmark.py     # protocol taxonomy (within-subject vs LOSO, binary vs 3-class)
├── NOVELTY_DANN.md           # writeup of the domain-adversarial contribution
├── generate_synthetic_test_csvs.py  # synthetic fixtures for the upload feature
├── export_for_serving.py     # packages the frozen model -> serving/ (Part 2)
├── serving/                  # COMMITTED portable demo bundle
│   ├── model_weights.pt, model_config.json, feature_scaler.pkl, label_map.json
│   ├── rf_model.pkl          # Random Forest — the ensemble's classical half
│   ├── demo_samples.json     # real held-out test windows for playback
│   ├── feature_schema.json, upload_template.csv  # CSV-upload schema + example
│   └── frontend_data/*.json  # confusion matrices, SHAP, attention, metrics
├── backend/                  # FastAPI serving API (Part 3); upload_service.py = CSV upload
├── frontend/                 # React + Vite demo UI (Part 4)
├── test_data/synthetic/      # synthetic upload fixtures (CSVs gitignored; README + generator tracked)
├── RESULTS.md                # full results + methodology narrative
└── README.md
```

The `serving/` bundle is committed so the demo runs without the training data or
checkpoints. To regenerate it you need the training data and the frozen
checkpoint (see below).

## Reproducing the model results

Requires Python with `torch`, `scikit-learn`, `scipy`, `pandas`, `joblib`
(and `xgboost` for the classical sweep only).

```bash
# Full 26-fold LOSO significance + ensembling; appends the "Locked Model"
# section to RESULTS.md and writes outputs/{loso_per_fold,significance_tests,
# ensemble_results,fixed_split_confusion}.*
python part1_rigor.py

# Classical-model sweep (XGBoost/SVM vs Random Forest, 26-fold LOSO).
# Ruled out — nothing beats the Hybrid+RF ensemble; see RESULTS.md.
python classical_sweep.py

# Protocol taxonomy: why literature's 60-90% is an easier problem than this
# project's 3-class LOSO. Appends "Protocol Taxonomy & Benchmark" to RESULTS.md.
python protocol_benchmark.py

# NOVEL contribution: subject-invariant (domain-adversarial) training. 26-fold
# LOSO comparing baseline Hybrid vs DANN (+RF ensembles), paired significance,
# and a subject-invariance probe. Writes outputs/dann_ablation.json. See
# NOVELTY_DANN.md. (Smoke test: --max-folds 2 --epochs 6)
python dann_ablation.py

# Re-package the frozen model (+ RF) + demo data + upload schema for serving/
python export_for_serving.py

# Consolidate the research-result artifacts (DANN, HbR, protocol, classical
# sweep) into serving/frontend_data/research.json for the app's Research page
python export_research_data.py

# (Optional) regenerate synthetic CSVs for manually testing the upload feature
python generate_synthetic_test_csvs.py
```

The full results and methodology narrative — including every rejected candidate
and the decision-gate reasoning — lives in [`RESULTS.md`](RESULTS.md); the novel
domain-adversarial contribution is written up in [`NOVELTY_DANN.md`](NOVELTY_DANN.md).

## Running the demo app

Two processes: the API on port 8000 and the Vite dev server on port 5173.

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>. Every value the API returns
traces to a committed artifact under `serving/` — nothing is hardcoded.

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The seven pages — Overview, **Research** (a
long-form dossier of the thesis, model, protocol taxonomy, the novel DANN
contribution, and the decision-gate results), Live Prediction, Session Trend,
Upload Data, Model Insights, Methodology — all read live data from the backend.

The Research page's charts read a consolidated `serving/frontend_data/research.json`
(the DANN / HbR / protocol / classical-sweep artifacts) via `GET /api/research/summary`.
Regenerate that bundle from `outputs/` with `python export_research_data.py` — like
everything else, its numbers trace to committed artifacts rather than being
hardcoded in the UI.

## CSV upload (batch prediction)

The **Upload Data** page (and `POST /api/predict/upload`) accepts a CSV of
**already-extracted TU Berlin features** and returns an ensemble prediction for
every row, plus a class-distribution summary.

- **Expected schema:** the 180 EEG + 468 HbO feature columns the model uses,
  plus `subject_id` (for per-subject normalization) and optionally
  `Cognitive_Load` (shown as ground truth). The standard `hbr_*` and `task`
  columns may be present — they are ignored. The exact schema lives in
  `serving/feature_schema.json`; download a ready-made example from the page or
  `GET /api/predict/upload/template`.
- **Strict validation:** a CSV missing any required feature column is rejected
  with the exact list of what's missing — it is never silently processed. If
  `subject_id` is absent, the response is flagged
  `"normalization": "global_fallback_no_subject_id"` (a degraded mode, since the
  model was trained with per-subject stats).
- **Scope:** this is upload-of-correctly-extracted-features only, for
  TU-Berlin-format data. It does **not** accept raw signals or data from another
  dataset/device — those would need a separate, dataset-specific feature
  extraction pipeline first (see Future Work in `RESULTS.md`).

The upload preprocessing is a self-contained reimplementation of the training
pipeline, verified byte-exact against `data_loader.py` (max abs diff 0.0), so
uploaded rows get identical predictions to the rest of the app.

## Honest limitations

- **Per-subject normalization is a demo-mode simplification.** The model
  normalizes each subject against that subject's own statistics. For a genuinely
  new person those don't exist yet, so the demo reuses each known subject's
  precomputed stats. This is documented in code, the API description, and the
  Methodology page.
- Only 26 subjects — small for cross-subject generalization claims.
- The "sustained-engagement" band in the trend view is literature-informed
  context (Yerkes–Dodson), not a clinical threshold.

See [`RESULTS.md`](RESULTS.md) for the complete results and methodology.
