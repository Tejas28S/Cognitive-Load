"""
Part 2 — Export the FINAL locked model + demo data for serving.

Everything the FastAPI backend needs, with no dependency on the training
pipeline at inference time:

  serving/model_weights.pt        frozen Hybrid state_dict
  serving/model_config.json       architecture description to rebuild it
  serving/feature_scaler.pkl      fitted EEG RobustScaler + fNIRS StandardScaler
  serving/label_map.json          {0:Low,1:Medium,2:High}
  serving/demo_samples.json       real held-out test windows, fully preprocessed
  serving/frontend_data/*.json    confusion matrices, SHAP, attention, metrics

The demo windows are the genuine subject-held-out fixed-split TEST rows, fully
preprocessed exactly as the model saw them at evaluation time (per-subject
normalization + log transforms + train-fitted scalers already applied), so the
backend just runs a forward pass. Per-subject normalization uses each demo
subject's own stats — a DEMO-MODE simplification (known subjects), not
production behaviour for genuinely unseen subjects. See Task 2.3 in the plan.
"""

import argparse
import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import Config
from data_loader import CognitiveLoadDataLoader
from models.architectures import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="outputs/hybrid_best.pt")
parser.add_argument("--serving_dir", type=str, default="serving")
args = parser.parse_args()

FROZEN_HYBRID = args.model
SERVING = args.serving_dir
FRONTEND = os.path.join(SERVING, "frontend_data")
LABELS = ["Low", "High"] if "binary" in FROZEN_HYBRID or "best_model.pt" in FROZEN_HYBRID else ["Low", "Medium", "High"]

# Ensemble spec — must match part1_rigor.py exactly (fixed-split Hybrid+RF).
RF_ESTIMATORS = 400
RF_SEED = 42
ENSEMBLE_WEIGHTS = [0.5, 0.5]  # simple probability average: 0.5*hybrid + 0.5*rf


def final_config():
    cfg = copy.copy(Config())
    cfg.PER_SUBJECT_NORMALIZE = True
    cfg.PER_SUBJECT_NORMALIZE_STAGE = "before_log"
    cfg.FUSION_TYPE = "attention"
    cfg.LOSS_FN = "weighted_ce"
    return cfg


def build_fixed_split(cfg):
    """Deterministically reproduce data_loader.prepare()'s split + scaling."""
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, groups = dl.load_raw(return_groups=True)
    subjects = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])[
        cfg.SUBJECT_COLUMN
    ].astype(str).to_numpy()
    tr, va, te = dl.split_indices(labels, groups)

    eeg_scaler = RobustScaler().fit(eeg_raw[tr])
    fnirs_scaler = StandardScaler().fit(fnirs_raw[tr])
    return {
        "n_eeg": eeg_raw.shape[1],
        "n_fnirs": fnirs_raw.shape[1],
        "test_idx": te,
        "subjects": subjects,
        "labels": labels,
        "eeg_test": eeg_scaler.transform(eeg_raw[te]),
        "fnirs_test": fnirs_scaler.transform(fnirs_raw[te]),
        # Scaled TRAIN features, kept to fit the ensemble's Random Forest exactly
        # as part1_rigor.py did (RobustScaler EEG + StandardScaler fNIRS, hstacked).
        "eeg_train": eeg_scaler.transform(eeg_raw[tr]),
        "fnirs_train": fnirs_scaler.transform(fnirs_raw[tr]),
        "labels_train": labels[tr],
        "eeg_scaler": eeg_scaler,
        "fnirs_scaler": fnirs_scaler,
        # Resolved feature-column ORDER actually used by the model (authoritative
        # for the CSV-upload schema). dl.load_raw set these.
        "eeg_cols": list(dl.eeg_cols),
        "fnirs_cols": list(dl.fnirs_cols),
    }


def export_model_and_scaler(cfg, fs, device):
    os.makedirs(SERVING, exist_ok=True)
    model = build_model("hybrid_cnn_lstm", fs["n_eeg"], fs["n_fnirs"], cfg).to(device)
    state = torch.load(FROZEN_HYBRID, map_location=device)
    model.load_state_dict(state, strict=False)
    # Save the serving weights without the domain head
    torch.save(model.state_dict(), os.path.join(SERVING, "model_weights.pt"))

    model_config = {
        "model_type": "hybrid_cnn_lstm",
        "n_eeg_features": int(fs["n_eeg"]),
        "n_fnirs_features": int(fs["n_fnirs"]),
        "proj_dim": int(cfg.CNN_FILTERS[-1]),
        "enc_dim": int(cfg.FUSION_DIM),
        "n_blocks": int(len(cfg.LSTM_UNITS)),
        "encoder_dropout": float(cfg.CNN_DROPOUT),
        "fusion_type": cfg.FUSION_TYPE,
        "fc_units": list(cfg.FC_UNITS),
        "fc_dropout": float(cfg.FC_DROPOUT),
        "num_classes": int(cfg.NUM_CLASSES),
        "class_names": LABELS,
        "frozen_checkpoint": FROZEN_HYBRID,
        # The served model is the Hybrid+RF ensemble (the LOSO-best config in
        # RESULTS.md's Locked Model section). Probabilities are combined as a
        # simple average of the frozen Hybrid softmax and the RF predict_proba.
        "served_model": "hybrid_rf_ensemble",
        "ensemble": {
            "type": "probability_average",
            "members": ["hybrid_cnn_lstm", "random_forest"],
            "weights": ENSEMBLE_WEIGHTS,
            "rf_file": "rf_model.pkl",
            "rf_estimators": RF_ESTIMATORS,
            "rf_seed": RF_SEED,
            "rf_feature_order": "hstack([eeg_scaled, fnirs_scaled])",
        },
    }
    with open(os.path.join(SERVING, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    joblib.dump(
        {"eeg_scaler": fs["eeg_scaler"], "fnirs_scaler": fs["fnirs_scaler"],
         "n_eeg": int(fs["n_eeg"]), "n_fnirs": int(fs["n_fnirs"])},
        os.path.join(SERVING, "feature_scaler.pkl"),
    )
    with open(os.path.join(SERVING, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"0": "Low", "1": "High"}, f, indent=2)
    print("Exported model_weights.pt, model_config.json, feature_scaler.pkl, label_map.json")
    return model, model_config


def export_rf(fs):
    """Fit + save the ensemble's Random Forest, identical to part1_rigor.py's
    fixed-split ensemble (n_estimators=400, random_state=42, class_weight=
    'balanced', on hstack([eeg_scaled, fnirs_scaled]))."""
    X_tr = np.hstack([fs["eeg_train"], fs["fnirs_train"]])
    y_tr = fs["labels_train"]
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, random_state=RF_SEED,
        class_weight="balanced", n_jobs=-1,
    ).fit(X_tr, y_tr)
    joblib.dump(rf, os.path.join(SERVING, "rf_model.pkl"))
    print(f"Exported rf_model.pkl (fit on {X_tr.shape[0]} train rows x {X_tr.shape[1]} features)")
    return rf


def export_demo_samples(fs):
    """Real held-out test windows, grouped into per-subject playback sessions."""
    te = fs["test_idx"]
    subj = fs["subjects"][te]
    y = fs["labels"][te]
    eeg = fs["eeg_test"]
    fnirs = fs["fnirs_test"]

    samples = []
    for s in sorted(set(subj), key=lambda x: int(x)):
        mask = subj == s
        rows = np.where(mask)[0]
        for window_index, r in enumerate(rows):
            samples.append({
                "subject_id": int(s),
                "trial_id": window_index,          # each full-trial row = one trial
                "window_index": window_index,
                "timestamp_s": window_index * 10,  # synthetic 10s cadence for playback
                "true_label": int(y[r]),
                "true_label_name": LABELS[int(y[r])],
                "features": (eeg[r].tolist() + fnirs[r].tolist()),
            })
    with open(os.path.join(SERVING, "demo_samples.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_eeg": int(fs["n_eeg"]),
            "n_fnirs": int(fs["n_fnirs"]),
            "note": ("Real subject-held-out fixed-split test windows, fully "
                     "preprocessed (per-subject normalization is a demo-mode "
                     "simplification using each known subject's own stats)."),
            "samples": samples,
        }, f)
    n_subj = len(set(subj))
    print(f"Exported demo_samples.json: {len(samples)} windows across {n_subj} held-out subjects")


def export_frontend_data(sig_metrics=True):
    os.makedirs(FRONTEND, exist_ok=True)

    # Confusion matrices (produced by part1_rigor.py)
    conf_path = "outputs/fixed_split_confusion.json"
    if os.path.exists(conf_path):
        with open(conf_path, encoding="utf-8") as f:
            conf = json.load(f)
        with open(os.path.join(FRONTEND, "confusion_matrices.json"), "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=2)
        print("Exported confusion_matrices.json")

    # SHAP category importance -> JSON
    shap_csv = "outputs/shap_category_importance.csv"
    if os.path.exists(shap_csv):
        df = pd.read_csv(shap_csv)
        overall = (
            df.groupby("category", as_index=False)["mean_abs_shap"].mean()
            .sort_values("mean_abs_shap", ascending=False)
        )
        per_class = {}
        for cls in df["class"].unique():
            sub = df[df["class"] == cls].sort_values("mean_abs_shap", ascending=False)
            per_class[cls] = [
                {"category": r.category, "mean_abs_shap": float(r.mean_abs_shap),
                 "share": float(r.share) if not pd.isna(r.share) else None}
                for r in sub.itertuples()
            ]
        with open(os.path.join(FRONTEND, "shap.json"), "w", encoding="utf-8") as f:
            json.dump({
                "categories": [
                    {"category": r.category, "mean_abs_shap": float(r.mean_abs_shap)}
                    for r in overall.itertuples()
                ],
                "per_class": per_class,
            }, f, indent=2)
        print("Exported shap.json")

    # Attention gate weights -> JSON
    att_csv = "outputs/attention_weights_final.csv"
    if os.path.exists(att_csv):
        df = pd.read_csv(att_csv)
        overall_row = df[df["class"] == "All"].iloc[0]
        per_class = []
        for _, r in df[df["class"] != "All"].iterrows():
            total = r["eeg_mean"] + r["fnirs_mean"]
            per_class.append({
                "class": r["class"],
                "eeg": float(r["eeg_mean"]), "fnirs": float(r["fnirs_mean"]),
                "eeg_share": float(r["eeg_mean"] / total) if total else 0.5,
                "fnirs_share": float(r["fnirs_mean"] / total) if total else 0.5,
            })
        total = overall_row["eeg_mean"] + overall_row["fnirs_mean"]
        with open(os.path.join(FRONTEND, "attention.json"), "w", encoding="utf-8") as f:
            json.dump({
                "overall": {
                    "eeg": float(overall_row["eeg_mean"]),
                    "fnirs": float(overall_row["fnirs_mean"]),
                    "eeg_share": float(overall_row["eeg_mean"] / total) if total else 0.5,
                    "fnirs_share": float(overall_row["fnirs_mean"] / total) if total else 0.5,
                },
                "per_class": per_class,
            }, f, indent=2)
        print("Exported attention.json")

    # Metrics + significance -> JSON
    metrics = {}
    if os.path.exists("outputs/ensemble_results.json"):
        with open("outputs/ensemble_results.json", encoding="utf-8") as f:
            ens = json.load(f)
        metrics["fixed_split"] = ens.get("fixed_split", {})
        metrics["loso"] = ens.get("loso", {})
        metrics["ensemble_accepted"] = ens.get("ensemble_accepted")
        metrics["acceptance_reason"] = ens.get("acceptance_reason")
    if os.path.exists("outputs/significance_tests.json"):
        with open("outputs/significance_tests.json", encoding="utf-8") as f:
            metrics["significance"] = json.load(f)
    with open(os.path.join(FRONTEND, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Exported metrics.json")


def export_upload_schema(cfg, fs):
    """Record the exact feature schema + preprocessing params the CSV-upload
    endpoint needs to reproduce the served pipeline without importing the
    training code."""
    schema = {
        "eeg_columns": fs["eeg_cols"],
        "fnirs_columns": fs["fnirs_cols"],
        "n_eeg": int(fs["n_eeg"]),
        "n_fnirs": int(fs["n_fnirs"]),
        "label_column": cfg.LABEL_COLUMN,
        "subject_column": cfg.SUBJECT_COLUMN,
        # Standard TU Berlin columns that may appear but are NOT used by the
        # model (HbR features, task name). Tolerated on upload, never required.
        "optional_ignored_prefixes": ["hbr_"],
        "optional_ignored_columns": ["task", "trial_id"],
        "class_names": cfg.CLASS_NAMES,
        "class_names_short": LABELS,
        "preprocessing": {
            "order": ["nan_to_num", "per_subject_zscore", "log", "saved_scalers"],
            "per_subject_normalize": bool(cfg.PER_SUBJECT_NORMALIZE),
            "per_subject_stage": cfg.PER_SUBJECT_NORMALIZE_STAGE,
            "per_subject_eps": float(getattr(cfg, "PER_SUBJECT_NORMALIZE_EPS", 1e-8)),
            "eeg_log": "log1p_clip_nonneg",   # EEG_LOG_MODE="clip"
            "fnirs_log": "signed_log1p",
        },
    }
    with open(os.path.join(SERVING, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Exported feature_schema.json ({len(fs['eeg_cols'])} EEG + {len(fs['fnirs_cols'])} HbO cols)")


def export_upload_template(cfg, n_rows=3):
    """A small RAW-feature example CSV (full TU Berlin header + a few real rows)
    so users can see the exact expected upload format."""
    df = pd.read_csv(cfg.MERGED_CSV)
    # A few rows spanning different subjects/classes for a representative sample.
    picks = []
    for cls in sorted(df[cfg.LABEL_COLUMN].unique())[:n_rows]:
        picks.append(df[df[cfg.LABEL_COLUMN] == cls].index[0])
    sample = df.loc[picks]
    sample.to_csv(os.path.join(SERVING, "upload_template.csv"), index=False)
    print(f"Exported upload_template.csv ({len(sample)} rows x {sample.shape[1]} cols)")


def verify_ensemble(cfg, model, rf, fs, device):
    """Confirm the exported Hybrid+RF ensemble reproduces the documented
    fixed-split ensemble numbers (macro F1 ~0.3909, acc ~0.40)."""
    from sklearn.metrics import accuracy_score, f1_score

    X_te = np.hstack([fs["eeg_test"], fs["fnirs_test"]]).astype(np.float32)
    y_te = fs["labels"][fs["test_idx"]]
    eeg_t = torch.tensor(fs["eeg_test"], dtype=torch.float32, device=device)
    fn_t = torch.tensor(fs["fnirs_test"], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        h_probs = torch.softmax(model(eeg_t, fn_t), dim=-1).cpu().numpy()
    rf_probs = rf.predict_proba(X_te)
    ens_probs = ENSEMBLE_WEIGHTS[0] * h_probs + ENSEMBLE_WEIGHTS[1] * rf_probs
    ens_preds = ens_probs.argmax(axis=1)
    acc = accuracy_score(y_te, ens_preds)
    f1 = f1_score(y_te, ens_preds, average="macro")
    print(f"Ensemble check on fixed-split test: acc={acc:.4f} macro_f1={f1:.4f} "
          f"(documented: acc=0.4000 f1=0.3909)")


def main():
    device = torch.device("cpu")
    cfg = final_config()
    fs = build_fixed_split(cfg)
    model, _ = export_model_and_scaler(cfg, fs, device)
    rf = export_rf(fs)
    export_demo_samples(fs)
    export_frontend_data()
    export_upload_schema(cfg, fs)
    export_upload_template(cfg)
    verify_ensemble(cfg, model, rf, fs, device)
    print("\nPart 2 export complete -> serving/")


if __name__ == "__main__":
    main()
