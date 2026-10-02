# Cognitive Load Estimation from EEG + fNIRS

A hybrid deep-learning model that fuses **EEG** (electrical) and **fNIRS**
(hemodynamic) brain signals from the Tu Berlin open dataset to estimate mental
workload, plus a demo web app (FastAPI + React) that serves the frozen model and
its evaluation artifacts.

Two labelling regimes exist. The **binary** model (Low vs High) is what is
served today — a frozen Hybrid network alone; the Random Forest ensemble is
disabled for this bundle because the Hybrid outperforms it. The original
**3-class** model (Low / Medium / High) is kept in `serving_3class/` as a
historical comparison, and there the Hybrid + RF ensemble *was* accepted.

The headline result is as much methodological as it is a model: cross-subject
(leave-one-subject-out) performance is meaningfully lower than a fixed
train/test split suggests, because a fixed split leaks subject identity. Both
numbers are always reported together — see [`RESULTS.md`](RESULTS.md) and the
app's **Methodology** page.

For the current state of the work — headline numbers, the full 58-configuration
search history and what is still open — see
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

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
├── ingestion/                # universal dataset input layer (parser/normalizer/projection)
├── schema/                   # canonical schema + channel alias registry + dataset profiles
├── predict.py                # CLI: predict from any supported dataset file
├── tests/test_ingestion.py   # ingestion suite (incl. TU Berlin regression)
├── export_for_serving.py     # packages the frozen model -> serving/ (Part 2)
├── serving/                  # COMMITTED portable demo bundle
│   ├── model_weights.pt, model_config.json, feature_scaler.pkl, label_map.json
│   ├── rf_model.pkl          # Random Forest — the ensemble's classical half
│   ├── demo_samples.json     # real held-out test windows for playback
│   ├── feature_schema.json, upload_template.csv  # CSV-upload schema + example
│   └── frontend_data/*.json  # confusion matrices, SHAP, attention, metrics
├── research/                 # leakage-audited experiment framework (binary-model work)
│   ├── datamod.py            # preprocessing contract + LOSO folds + per-fold scalers
│   ├── leakage.py            # leakage assertions; aborts a run rather than leak a score
│   ├── archs.py              # model zoo: MLP + 1D-CNN encoders, fusion, DANN
│   ├── train.py, metrics.py, plots.py, explain.py, experiment.py, campaign.py
│   └── runctx.py             # the run archive (unique dirs, never overwritten)
├── run_campaign.py           # the original 42-experiment binary campaign (CLI)
├── target_campaign.py        # 15 follow-up candidates probing axes the 42 never touched
├── finalize_winner.py        # promote ONE dev winner to 26-fold LOSO + lockbox, paired vs a same-run baseline
├── ensemble_campaign.py      # 5-member deep-ensemble screen (--stage dev | final)
├── export_binary_reliability.py  # parse RESULTS_BINARY_IMPROVEMENT.md -> extra serving metrics keys
├── outputs/runs/             # per-run archive: metrics, predictions, figures, models
│   └── target_campaign_log.md    # one row per attempt, win or lose
├── backend/                  # FastAPI serving API (Part 3); upload_service.py = CSV upload
│   └── runs_service.py       # read-only access to outputs/runs/ for the API
├── frontend/                 # React + Vite demo UI (Part 4)
├── test_data/synthetic/      # synthetic upload fixtures (CSVs gitignored; README + generator tracked)
├── RESULTS.md                # full results + methodology narrative
├── RESULTS_BINARY_IMPROVEMENT.md    # measured results of the binary-model campaign
├── PROJECT_STATUS.md         # current state: headline numbers, search history, what's open
├── explanation/              # project, API and methodology write-ups
│   ├── PROJECT_RECORD.md     # the complete record: architecture, every experiment, all results
│   └── *.pdf                 # every document as PDF (research/make_pdf.py)
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

## Improving the binary model (`research/`)

A separate, additive framework for improving the **binary (Low vs High)** model
without letting the evaluation flatter it. It does not touch `serving/`,
`serving_3class/` or the legacy training scripts, so the existing demo is
unaffected.

```bash
# Full staged campaign: baseline, then one controlled experiment axis per stage,
# then 26-fold LOSO + a lockbox evaluation for the selected configuration.
.venv/bin/python run_campaign.py --tag my-run

# A single stage (e.g. the CNN/architecture comparison)
.venv/bin/python run_campaign.py --stages S6

# Smoke test: reduced cohort + epoch budget, exercises every code path in ~2 min
.venv/bin/python run_campaign.py --quick --limit-subjects 7 --n-lockbox 3 --max-epochs 6
```

Two later campaigns extend the search using the same framework and the same
acceptance rule:

```bash
# 15 follow-up candidates on axes the original 42 never touched (joint changes,
# intermediate capacity points, batch 32, shorter early-stop patience).
.venv/bin/python target_campaign.py

# Promote ONE dev-screen winner to the headline protocol (26-fold LOSO +
# lockbox). Re-runs the untouched baseline through the same two protocols in the
# same environment, so the comparison is paired and free of cross-run noise.
.venv/bin/python finalize_winner.py --candidate D2_narrow_patience6

# Deep-ensemble screen: 5 independently-seeded copies of the served architecture
# per fold, softmax-averaged. Mechanistically distinct from every prior attempt.
.venv/bin/python ensemble_campaign.py --stage dev
.venv/bin/python ensemble_campaign.py --stage final
```

**Where the search stands: 58 configurations, no accepted improvement.** The 42
original experiments and the 15 target candidates were all rejected. The best
target candidate (`D2_narrow_patience6`) was promoted to the full protocol and
*lost* to the baseline on both 26-fold LOSO and the lockbox, with no result
significant — a direct demonstration that a development-screen win can evaporate
on the headline protocol. The 5-member deep ensemble is the only candidate to
clear the one-standard-error bar on the development screen (accuracy 0.6632 vs
the in-run baseline's 0.6304); it has **not** been promoted or served, and it
carries real caveats — thin margins, a worse per-class balance, and a wider
train−heldout gap. Full accounting in [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
and `outputs/runs/target_campaign_log.md`.

Key properties:

- **Development / lockbox split.** 18 subjects drive every experiment and every
  comparison; 8 subjects are held back and evaluated once, after the
  configuration is frozen. The 26-fold LOSO number is also reported and labelled
  as selection-informed.
- **Leakage assertions run before every fold trains** — subject, trial, window,
  duplicate-row, scaler-fit and held-out-subject checks. A failure aborts the run,
  so a leaked score is never produced.
- **Every run gets its own directory** under `outputs/runs/`, created before any
  data is read. Runs are never overwritten and failed runs are never deleted.
- **Metrics are computed from saved predictions**, so anything the UI shows can be
  recomputed from that run's `predictions.csv` — `research.verify_run` does exactly
  that and reports zero mismatches.

Results are browsable at `/evaluation` and `/runs` in the app.

- **Start here**: [`explanation/Cognitive_Load_Project_Report.pdf`](explanation/Cognitive_Load_Project_Report.pdf) —
  an 18-page illustrated report with every figure, a table of contents and page
  numbers. Every document in `explanation/` is also available as a PDF; rebuild
  them all with `.venv/bin/python -m research.make_pdf --all` (needs the
  `pandoc` binary and the `weasyprint` package).
- Full detail in text: [`explanation/PROJECT_RECORD.md`](explanation/PROJECT_RECORD.md) —
  architecture, protocol, leakage audit, all 42 experiments of the original
  campaign, final results and limitations. The later target and ensemble
  campaigns are recorded in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
- Methodology only: [`explanation/BINARY_MODEL_IMPROVEMENT.md`](explanation/BINARY_MODEL_IMPROVEMENT.md)
- Machine-generated numbers: [`RESULTS_BINARY_IMPROVEMENT.md`](RESULTS_BINARY_IMPROVEMENT.md)

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

Open <http://localhost:5173> (use `localhost`, not `127.0.0.1` — Vite binds the
hostname, which resolves to IPv6). Every page reads live data from the backend:

| Route | Page | What it is |
|---|---|---|
| `/` | Landing | Full-viewport opening + project abstract |
| `/overview` | Overview | Headline result and model comparison |
| `/workflow` | **Workflow** | The research pipeline in six stages, with a scroll-tracked progress rail |
| `/methodology` | Methodology | Protocol, calibration and honest limitations |
| `/research` | **Research** | Long-form dossier: thesis, protocol taxonomy, the novel DANN contribution, decision gates |
| `/live`, `/session`, `/upload` | Assess | Single prediction, session playback, CSV batch prediction |
| `/insights`, `/evaluation`, `/runs` | Evidence | Confusion/SHAP/attention, full run metrics, the run archive |
| `/about`, `/bios`, `/faq` | About | Motivation and principles, team, plain-language answers |

> `/bios` ships **placeholder** team entries and shows a warning banner until
> they are filled in at `frontend/src/lib/team.ts` — names and affiliations are
> the one thing on the site that cannot be derived from the repository.

### 3. Smoke tests (optional)

With both processes running:

```bash
cd frontend
npx playwright install chromium   # first time only
npm test
```

Mounts every route in a real browser and fails on uncaught exceptions, React
errors, unexpected failed requests, or any page that scrolls sideways at 375px.

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
- **Scope:** this endpoint is upload-of-correctly-extracted-features only, for
  TU-Berlin-format data. For raw signals or another dataset entirely, see
  **Universal dataset ingestion** below.

The upload preprocessing is a self-contained reimplementation of the training
pipeline, verified byte-exact against `data_loader.py` (max abs diff 0.0), so
uploaded rows get identical predictions to the rest of the app.

## Universal dataset ingestion

A dataset-agnostic input layer sits in front of the model, so a differently
structured dataset can be read without hand-editing its columns. It is
**additive** — the endpoints above are unchanged, and the TU Berlin flow is
covered by a regression test asserting bit-identical predictions.

```bash
python predict.py --list-datasets
python predict.py --input any_dataset.csv          # extracted features
python predict.py --input eeg.mat --input nirs.mat # raw signals, one per modality
```

```
raw file -> DatasetParser -> SchemaNormalizer -> ChannelProjector -> [existing extraction] -> [frozen model]
            reads it        names -> canonical  channels -> bundle       UNCHANGED             UNCHANGED
```

- **Formats:** `.csv .tsv .txt .mat .npz .npy .edf .bdf`, wide or long layout,
  with the dataset profile auto-detected from the file's own structure.
- **Config, not code:** channel spellings live in
  `schema/channel_aliases.yaml` and file layouts in `schema/datasets/*.yaml`.
  Onboarding a dataset whose naming varies on known aliases needs no code change.
- **Every prediction carries a mapping report** saying how each input column was
  interpreted, which were ignored, and what coverage of the model's expected
  channels the dataset actually achieved.
- **It refuses rather than guesses.** Below 70% channel coverage the pipeline
  returns a structured error naming the missing channels instead of zero-filling
  them. After the per-subject z-score a zero is a legitimate value, so a
  zero-filled channel would be indistinguishable from real data — a
  confident-looking prediction built on fabricated inputs.

Worked example — the Stroop EEG+fNIRS dataset parses and normalizes with no
manual renaming, projects its EEG at 0.767 coverage (16 exact electrode matches
+ 14 documented neighbour substitutions), and then correctly **refuses** at the
fNIRS stage: the model expects 36 optode positions and Stroop supplies 20
frontal ones, with no montage metadata in either to justify a correspondence.
Predicting for such a dataset requires a bundle trained on its own feature
space, which this layer also makes possible.

See [`docs/adding_a_new_dataset.md`](docs/adding_a_new_dataset.md) for the
onboarding guide and [`docs/tu_berlin_assumptions.md`](docs/tu_berlin_assumptions.md)
for the audit of what was TU Berlin-specific and why.

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
