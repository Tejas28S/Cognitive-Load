"""
main.py — FastAPI app for the cognitive-load demo.

Serves the FINAL locked model (via ModelService) and precomputed evaluation
artifacts (via DataService). Every number returned traces to a committed
artifact under serving/ — nothing here is hardcoded.

Run:
    uvicorn main:app --reload --port 8000
(from inside the backend/ directory)
"""

import io

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
model_service = ModelService()
data_service = DataService()
upload_service = UploadService()


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=Health)
def health():
    return {"status": "ok"}


# ── Model metrics / confusion matrices ──────────────────────────────────────

@app.get("/api/model/metrics")
def model_metrics():
    """Final fixed-split + LOSO metrics, ensembling, and significance tests."""
    return data_service.metrics()


@app.get("/api/model/confusion-matrix", response_model=ConfusionMatrix)
def confusion_matrix(model: str = Query("hybrid", pattern="^(hybrid|eeg|fnirs)$")):
    cm = data_service.confusion_matrix(model)
    if cm is None:
        raise HTTPException(status_code=404, detail=f"No confusion matrix for '{model}'")
    return cm


# ── Explainability ──────────────────────────────────────────────────────────

@app.get("/api/explainability/shap")
def explain_shap():
    return data_service.shap()


@app.get("/api/explainability/attention")
def explain_attention():
    return data_service.attention()


# ── Research artifacts (narrative page) ─────────────────────────────────────

@app.get("/api/research/summary")
def research_summary():
    """Consolidated results behind the /research narrative: DANN ablation +
    lambda sweep, protocol taxonomy, HbR ablation (+ served-split), classical
    sweep, and locked-model significance. Every block traces to an artifact
    under outputs/, copied into serving/frontend_data/research.json by
    export_research_data.py."""
    research = data_service.research()
    if not research:
        raise HTTPException(
            status_code=404,
            detail="No research.json in the serving bundle. Run export_research_data.py.",
        )
    return research


# ── Demo sessions ───────────────────────────────────────────────────────────

@app.get("/api/demo/subjects")
def demo_subjects():
    return data_service.list_subjects()


@app.get("/api/demo/session/{subject_id}/{trial_id}")
def demo_session(subject_id: int, trial_id: int = 0):
    session = data_service.get_session(subject_id, trial_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"No demo session for subject {subject_id}")
    return session


# ── Prediction ──────────────────────────────────────────────────────────────

def _resolve_features(req: PredictRequest):
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
def predict(req: PredictRequest):
    features, true_label, true_label_name, window_index = _resolve_features(req)
    out = model_service.predict_single(features)
    out["true_label"] = true_label
    out["true_label_name"] = true_label_name
    out["window_index"] = window_index
    return out


@app.post("/api/predict/session")
def predict_session(req: SessionRequest):
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

SHORT = upload_service.class_names_short  # ["Low", "Medium", "High"]


@app.get("/api/predict/upload/template")
def upload_template():
    """A small RAW-feature example CSV showing the exact expected upload format."""
    path = f"{SERVING_DIR}/upload_template.csv"
    return FileResponse(path, media_type="text/csv", filename="upload_template.csv")


@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Validate + preprocess a CSV of already-extracted features and return
    ensemble predictions for every valid row, plus a summary. Rejects on any
    missing required feature column — never silently processes mismatched data."""
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
