"""
main.py — FastAPI app for the cognitive-load demo.

Serves the FINAL locked model (via ModelService) and precomputed evaluation
artifacts (via DataService). Every number returned traces to a committed
artifact under serving/ — nothing here is hardcoded.

Run:
    uvicorn main:app --reload --port 8000
(from inside the backend/ directory)
# Triggering reload
"""

import io
import json

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from data_service import DataService
from model_service import ModelService
from upload_service import SERVING_DIR, UploadService
from schemas import (
    ConfusionMatrix,
    Health,
    PredictRequest,
    PredictResponse,
    SessionRequest,
)

app = FastAPI(
    title="Cognitive Load — Demo API",
    description=(
        "Serves the frozen Hybrid EEG+fNIRS cognitive-load model and its "
        "committed evaluation artifacts. Predictions run on real "
        "subject-held-out demo windows. Per-subject normalization is a "
        "demo-mode simplification (known subjects), not production behaviour "
        "for genuinely unseen subjects."
    ),
    version="1.0.0",
)

# CORS for the Vite dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model + data once at startup (not per-request).
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLES = {"binary": "serving", "3class": "serving_3class"}

model_services, data_services, upload_services = {}, {}, {}
bundle_errors = {}

# A bundle is only served if it is complete. serving_3class/ ships its JSON
# artifacts but not model_weights.pt / feature_scaler.pkl (gitignored), so a
# hard load here would take the whole API down with it.
for _version, _dirname in _BUNDLES.items():
    _dir = os.path.join(_ROOT, _dirname)
    try:
        model_services[_version] = ModelService(serving_dir=_dir)
        data_services[_version] = DataService(serving_dir=_dir)
        upload_services[_version] = UploadService(serving_dir=_dir)
    except Exception as exc:
        bundle_errors[_version] = f"{type(exc).__name__}: {exc}"
        model_services.pop(_version, None)
        data_services.pop(_version, None)
        upload_services.pop(_version, None)
        print(f"[startup] '{_version}' bundle unavailable ({_dirname}/): {exc}")

if not model_services:
    raise RuntimeError(f"No serving bundle could be loaded: {bundle_errors}")


def get_services(version: str):
    if version in bundle_errors:
        raise HTTPException(
            status_code=503,
            detail=f"The '{version}' model bundle is not available: {bundle_errors[version]}",
        )
    if version not in model_services:
        raise HTTPException(status_code=400, detail="Invalid version. Use 'binary' or '3class'")
    return model_services[version], data_services[version], upload_services[version]


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=Health)
def health():
    return {"status": "ok"}


# ── Which model bundles can actually be served ──────────────────────────────

@app.get("/api/bundles")
def bundles():
    """Describe every known bundle and whether it can be served.

    The UI's model switch reads this so it can offer 2-class and 3-class as real
    options, label each with its true class names, and disable (with a reason)
    any bundle whose artifacts are missing — rather than letting the user pick a
    mode that only fails at request time.
    """
    out = []
    for version, dirname in _BUNDLES.items():
        service = model_services.get(version)
        if service is not None:
            names = list(service.class_names)
            out.append({
                "version": version,
                "available": True,
                "num_classes": len(names),
                "class_names": names,
                "reason": None,
                # Present when a bundle is not the original frozen research
                # checkpoint; the UI labels those so their numbers are never
                # mistaken for the published ones.
                "provenance": service.cfg.get("provenance"),
            })
            continue

        # The config JSON usually survives even when the weights do not, so the
        # switch can still name the option it is greying out.
        names, n_classes = [], None
        cfg_path = os.path.join(_ROOT, dirname, "model_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                names = list(cfg.get("class_names", []))
                n_classes = cfg.get("num_classes") or (len(names) or None)
            except (OSError, ValueError):
                pass
        out.append({
            "version": version,
            "available": False,
            "num_classes": n_classes,
            "class_names": names,
            "reason": bundle_errors.get(version, "bundle not loaded"),
            "provenance": None,
        })

    return {
        "bundles": out,
        "default": next((b["version"] for b in out if b["available"]), None),
    }


# ── Model metrics / confusion matrices ──────────────────────────────────────

@app.get("/api/model/metrics")
def model_metrics(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    return data_service.metrics()


@app.get("/api/model/confusion-matrix", response_model=ConfusionMatrix)
def confusion_matrix(model: str = Query("hybrid", pattern="^(hybrid|eeg|fnirs)$"), version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    cm = data_service.confusion_matrix(model)
    if cm is None:
        raise HTTPException(status_code=404, detail=f"No confusion matrix for '{model}'")
    return cm


# ── Explainability ──────────────────────────────────────────────────────────

@app.get("/api/explainability/shap")
def explain_shap(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    return data_service.shap()


@app.get("/api/explainability/attention")
def explain_attention(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    return data_service.attention()


# ── Model diagnostics / config ──────────────────────────────────────────────

@app.get("/api/model/diagnostics")
def model_diagnostics(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    diag = data_service.diagnostics()
    if not diag:
        raise HTTPException(
            status_code=404,
            detail="No diagnostics.json in the serving bundle. Run export_diagnostics.py.",
        )
    return diag


@app.get("/api/model/config")
def model_config(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    cfg = data_service.model_config()
    if not cfg:
        raise HTTPException(status_code=404, detail="No model_config.json in the serving bundle.")
    return cfg


# ── Research artifacts (narrative page) ─────────────────────────────────────

@app.get("/api/research/summary")
def research_summary(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    research = data_service.research()
    if not research:
        raise HTTPException(
            status_code=404,
            detail="No research.json in the serving bundle. Run export_research_data.py.",
        )
    return research


# ── Demo sessions ───────────────────────────────────────────────────────────

@app.get("/api/demo/subjects")
def demo_subjects(version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    return data_service.list_subjects()


@app.get("/api/demo/session/{subject_id}/{trial_id}")
def demo_session(subject_id: int, trial_id: int = 0, version: str = Query("binary")):
    _, data_service, _ = get_services(version)
    session = data_service.get_session(subject_id, trial_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"No demo session for subject {subject_id}")
    return session


# ── Prediction ──────────────────────────────────────────────────────────────

def _resolve_features(req: PredictRequest, data_service):
    """Return (features, true_label, true_label_name, window_index)."""
    if req.features is not None:
        return req.features, None, None, req.window_index
    if req.subject_id is not None and req.window_index is not None:
        w = data_service.get_window(req.subject_id, req.window_index)
        if w is None:
            raise HTTPException(
                status_code=404,
                detail=f"No window {req.window_index} for subject {req.subject_id}",
            )
        return w["features"], w["true_label"], w["true_label_name"], w["window_index"]
    raise HTTPException(
        status_code=422,
        detail="Provide either 'features' or ('subject_id' and 'window_index').",
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest, version: str = Query("binary")):
    model_service, data_service, _ = get_services(version)
    features, true_label, true_label_name, window_index = _resolve_features(req, data_service)
    out = model_service.predict_single(features)
    out["true_label"] = true_label
    out["true_label_name"] = true_label_name
    out["window_index"] = window_index
    return out


@app.post("/api/predict/session")
def predict_session(req: SessionRequest, version: str = Query("binary")):
    model_service, data_service, _ = get_services(version)
    session = data_service.get_session(req.subject_id, req.trial_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"No demo session for subject {req.subject_id}")
    results = model_service.predict_batch([w["features"] for w in session])
    out = []
    for w, r in zip(session, results):
        out.append({
            "window_index": w["window_index"],
            "timestamp_s": w["timestamp_s"],
            "predicted_label": r["predicted_label"],
            "predicted_index": r["predicted_index"],
            "probabilities": r["probabilities"],
            "attention_weights": r["attention_weights"],
            "true_label": w["true_label"],
            "true_label_name": w["true_label_name"],
        })
    return out


# ── CSV upload (batch prediction on already-extracted TU Berlin features) ────

@app.get("/api/predict/upload/template")
def upload_template(version: str = Query("binary")):
    _, _, upload_service = get_services(version)
    path = f"{upload_service.serving_dir}/upload_template.csv"
    return FileResponse(path, media_type="text/csv", filename="upload_template.csv")


@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...), version: str = Query("binary")):
    model_service, _, upload_service = get_services(version)
    SHORT = upload_service.class_names_short
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="The uploaded CSV has no rows.")

    ok, validation = upload_service.validate_columns(df.columns)
    if not ok:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"CSV rejected: {validation['n_missing_required']} required "
                    "feature column(s) missing. This endpoint expects "
                    "already-extracted TU Berlin features "
                    f"({validation['expected_eeg']} EEG + {validation['expected_fnirs']} "
                    "HbO), not raw signals or a different dataset."
                ),
                **validation,
            },
        )

    features, subjects, labels, norm_mode = upload_service.preprocess(df)
    results = model_service.predict_batch(features.tolist())

    rows, dist = [], {name: 0 for name in SHORT}
    for i, r in enumerate(results):
        dist[r["predicted_label"]] += 1
        true_idx = int(labels[i]) if labels is not None else None
        rows.append({
            "row_index": i,
            "subject_id": str(subjects[i]) if norm_mode == "per_subject" else None,
            "predicted_label": r["predicted_label"],
            "predicted_index": r["predicted_index"],
            "probabilities": r["probabilities"],
            "attention_weights": r["attention_weights"],
            "true_label": true_idx,
            "true_label_name": SHORT[true_idx] if true_idx is not None and 0 <= true_idx < len(SHORT) else None,
        })

    accuracy = None
    if labels is not None:
        correct = sum(1 for row in rows if row["true_label"] == row["predicted_index"])
        accuracy = correct / len(rows) if rows else None

    return {
        "served_model": "hybrid_rf_ensemble",
        "normalization": norm_mode,
        "normalization_note": (
            "Per-subject normalization applied using each subject's own uploaded "
            "rows." if norm_mode == "per_subject" else
            "No subject_id column: per-subject normalization could not be applied; "
            "the whole file was normalized as one group (degraded — the model was "
            "trained with per-subject stats)."
        ),
        "summary": {
            "rows_processed": len(rows),
            "rows_rejected": 0,
            "class_distribution": dist,
            "accuracy_vs_uploaded_labels": accuracy,
            "unexpected_columns": validation["unexpected_columns"],
        },
        "predictions": rows,
    }


@app.post("/api/predict/upload_mat")
async def predict_upload_mat(
    eeg_file: UploadFile = File(...),
    nirs_file: UploadFile = File(...),
    version: str = Query("binary")
):
    from mat_extractor import process_raw_mats
    model_service, _, upload_service = get_services(version)
    SHORT = upload_service.class_names_short
    
    eeg_bytes = await eeg_file.read()
    nirs_bytes = await nirs_file.read()
    
    try:
        df = process_raw_mats(eeg_bytes, nirs_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract features from .mat files: {exc}")

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="No windows could be extracted. Files might be too short.")

    ok, validation = upload_service.validate_columns(df.columns)
    if not ok:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Extracted features do not match expected schema.",
                **validation,
            },
        )

    features, subjects, labels, norm_mode = upload_service.preprocess(df)
    results = model_service.predict_batch(features.tolist())

    rows, dist = [], {name: 0 for name in SHORT}
    for i, r in enumerate(results):
        dist[r["predicted_label"]] += 1
        rows.append({
            "row_index": i,
            "subject_id": str(subjects[i]) if norm_mode == "per_subject" else None,
            "predicted_label": r["predicted_label"],
            "predicted_index": r["predicted_index"],
            "probabilities": r["probabilities"],
            "attention_weights": r["attention_weights"],
            "true_label": None,
            "true_label_name": None,
        })

    return {
        "served_model": "hybrid_rf_ensemble",
        "normalization": norm_mode,
        "normalization_note": "Extracted directly from .mat files. Per-subject normalization applied.",
        "summary": {
            "rows_processed": len(rows),
            "rows_rejected": 0,
            "class_distribution": dist,
            "accuracy_vs_uploaded_labels": None,
            "unexpected_columns": validation["unexpected_columns"],
        },
        "predictions": rows,
    }



# ════════════════════════════════════════════════════════════════════════════
#  Evaluation runs — the experiment archive under outputs/runs/
#
#  These endpoints serve artifacts a training run wrote to disk. Nothing is
#  computed here: every metric, matrix, curve and figure below was produced by
#  research/ during that run and can be recomputed from its predictions.csv.
#  Each run is self-describing (its dataset_info.json carries class names and
#  classification mode), so binary and 3-class runs are handled by the same
#  code path with no per-model branching.
# ════════════════════════════════════════════════════════════════════════════

from fastapi.responses import Response

from runs_service import ArtifactNotFound, RunNotFound, RunsService

runs_service = RunsService(repo_root=_ROOT)


def _run_or_404(run_id):
    try:
        return runs_service.run_dir(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail=f"No run '{run_id}' in the archive")


def _artifact_or_404(run_id, relpath, what):
    _run_or_404(run_id)
    data = runs_service.try_read(run_id, relpath)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' has no {what} (expected {relpath}). "
                   "It may have failed before that stage.",
        )
    return data


@app.get("/api/runs")
def list_runs(
    mode: str = Query(None, description="filter by classification_mode, e.g. binary"),
    status: str = Query(None, description="filter by run status"),
    limit: int = Query(100, ge=1, le=500),
):
    """Run history. Newest first, generated by scanning the archive."""
    rows = runs_service.list_runs()
    if mode:
        rows = [r for r in rows if r.get("classification_mode") == mode]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return {
        "runs": rows[:limit],
        "total": len(rows),
        "runs_root": runs_service.runs_root,
        "modes": sorted({r.get("classification_mode") for r in runs_service.list_runs()
                         if r.get("classification_mode")}),
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    _run_or_404(run_id)
    return runs_service.get_run(run_id)


@app.get("/api/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    """Pooled held-out metrics plus the plain-English metric explanations."""
    metrics = _artifact_or_404(run_id, "metrics/metrics.json", "metrics")
    return {
        "metrics": metrics,
        "explanations": runs_service.try_read(
            run_id, "metrics/metric_explanations.json", {}),
        "loso": runs_service.try_read(run_id, "metrics/loso_metrics.json"),
        "per_class": runs_service.try_read(run_id, "metrics/per_class_metrics.json"),
    }


@app.get("/api/runs/{run_id}/confusion-matrix")
def run_confusion_matrix(run_id: str):
    return _artifact_or_404(
        run_id, "confusion_matrix/confusion_matrix.json", "confusion matrix")


@app.get("/api/runs/{run_id}/curves")
def run_curves(run_id: str):
    """ROC and precision-recall curve points for the held-out predictions."""
    return _artifact_or_404(run_id, "curves/curves.json", "ROC/PR curves")


@app.get("/api/runs/{run_id}/training-history")
def run_training_history(run_id: str):
    return _artifact_or_404(
        run_id, "curves/training_history.json", "training history")


@app.get("/api/runs/{run_id}/per-subject")
def run_per_subject(run_id: str):
    """Per-subject held-out performance, read from the LOSO metrics artifact."""
    loso = _artifact_or_404(run_id, "metrics/loso_metrics.json", "LOSO metrics")
    summary = runs_service.try_read(run_id, "run_summary.json", {}) or {}
    exp_id = (summary.get("model") or {}).get("selected_experiment")
    rows = runs_service.try_read(run_id, "experiments/FINAL_loso26/per_subject_metrics.json")
    return {
        "per_subject": rows or [],
        "fold_aggregate": loso.get("fold_aggregate"),
        "protocol": loso.get("protocol"),
        "lockbox": loso.get("lockbox"),
        "selected_experiment": exp_id,
    }


@app.get("/api/runs/{run_id}/feature-importance")
def run_feature_importance(run_id: str):
    data = _artifact_or_404(
        run_id, "feature_importance/shap_values.json", "feature importance")
    return data


@app.get("/api/runs/{run_id}/representations")
def run_representations(run_id: str):
    return _artifact_or_404(
        run_id, "representations/representations.json", "t-SNE representations")


@app.get("/api/runs/{run_id}/experiments")
def run_experiments(run_id: str):
    """The experiment leaderboard for this run, built from its own records."""
    _run_or_404(run_id)
    return runs_service.experiments(run_id)


@app.get("/api/runs/{run_id}/experiments/{exp_id}")
def run_experiment_detail(run_id: str, exp_id: str):
    _run_or_404(run_id)
    try:
        return runs_service.experiment_detail(run_id, exp_id)
    except ArtifactNotFound:
        raise HTTPException(
            status_code=404, detail=f"Run '{run_id}' has no experiment '{exp_id}'")


@app.get("/api/runs/{run_id}/predictions")
def run_predictions(run_id: str, limit: int = Query(500, ge=1, le=5000),
                    offset: int = Query(0, ge=0), subject: int = Query(None)):
    """Raw saved predictions, so any metric on the page can be reproduced."""
    _run_or_404(run_id)
    try:
        return runs_service.predictions(run_id, limit=limit, offset=offset,
                                        subject=subject)
    except ArtifactNotFound:
        raise HTTPException(status_code=404,
                            detail=f"Run '{run_id}' saved no predictions")


@app.get("/api/runs/{run_id}/summary")
def run_summary(run_id: str):
    return _artifact_or_404(run_id, "run_summary.json", "summary")


@app.get("/api/runs/{run_id}/figure")
def run_figure(run_id: str, path: str = Query(..., description="e.g. curves/roc_curve.png")):
    """Serve a PNG this run produced. Path-guarded to the run's own directory."""
    _run_or_404(run_id)
    if not path.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Only .png artifacts are served")
    try:
        return FileResponse(runs_service.artifact_path(run_id, path),
                            media_type="image/png")
    except ArtifactNotFound:
        raise HTTPException(status_code=404,
                            detail=f"Run '{run_id}' has no figure '{path}'")


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str):
    """The run's markdown summary, as text."""
    _run_or_404(run_id)
    try:
        with open(runs_service.artifact_path(run_id, "run_summary.md"),
                  encoding="utf-8") as fh:
            return Response(content=fh.read(), media_type="text/markdown")
    except ArtifactNotFound:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' has no report")
