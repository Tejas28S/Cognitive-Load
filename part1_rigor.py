"""
Part 1 remaining rigor work for the Master Build Plan.

Produces, all traceable to committed artifacts:
  1. Per-fold LOSO metrics (26 folds) for three configs, saved to
     outputs/loso_per_fold.csv:
        - hybrid   + per-subject normalization (the FINAL locked config)
        - hybrid   without per-subject normalization
        - fnirs-only + per-subject normalization
  2. Paired significance tests (paired t-test + Wilcoxon) across the 26
     folds:
        - Hybrid vs fNIRS-only         (does multimodality help?)
        - normalized vs non-normalized (does per-subject norm help LOSO?)
  3. Ensembling (Hybrid + RandomForest, probability averaging) on both the
     fixed split and LOSO, accepted only if it beats Hybrid-alone on LOSO
     without class collapse.
  4. Fixed-split confusion matrices (hybrid / eeg-only / fnirs-only) saved
     as raw counts for the frontend.

Outputs:
  outputs/loso_per_fold.csv
  outputs/significance_tests.json
  outputs/ensemble_results.json
  outputs/fixed_split_confusion.json
  appends a "Locked Model", significance and ensembling summary to RESULTS.md

Usage:
  python part1_rigor.py            # full 26-fold run (slow, ~30-45 min CPU)
  python part1_rigor.py --max-folds 3   # smoke test
"""

import argparse
import copy
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader

from config import Config
from data_loader import (
    CognitiveLoadDataLoader,
    CognitiveLoadDataset,
    configured_class_weights,
)
from evaluate import predict
from models.architectures import build_model
from trainer import train_model

FROZEN_HYBRID = os.path.join("outputs", "phase3_per-subject_normalization_before_log.pt")
RF_ESTIMATORS = 400
RF_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_config(overrides=None):
    cfg = copy.copy(Config())
    for key, value in (overrides or {}).items():
        setattr(cfg, key, value)
    return cfg


def make_loader(eeg, fnirs, labels, cfg, shuffle=False, drop_last=False):
    return DataLoader(
        CognitiveLoadDataset(
            eeg.astype(np.float32), fnirs.astype(np.float32), labels.astype(np.int64)
        ),
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def metrics_from(labels, preds, probs, num_classes=3):
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "recall": recall_score(
            labels, preds, labels=list(range(num_classes)), average=None, zero_division=0
        ).tolist(),
    }
    try:
        out["auc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        out["auc"] = float("nan")
    return out


# ── Fixed split (mirrors data_loader.prepare, deterministic) ──────────────

def build_fixed_split(cfg):
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, groups = dl.load_raw(return_groups=True)
    subjects = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])[
        cfg.SUBJECT_COLUMN
    ].astype(str).to_numpy()
    tr, va, te = dl.split_indices(labels, groups)

    eeg_scaler = RobustScaler()
    fnirs_scaler = StandardScaler()
    eeg_tr = eeg_scaler.fit_transform(eeg_raw[tr])
    eeg_va = eeg_scaler.transform(eeg_raw[va])
    eeg_te = eeg_scaler.transform(eeg_raw[te])
    fn_tr = fnirs_scaler.fit_transform(fnirs_raw[tr])
    fn_va = fnirs_scaler.transform(fnirs_raw[va])
    fn_te = fnirs_scaler.transform(fnirs_raw[te])

    return {
        "dl": dl,
        "n_eeg": eeg_raw.shape[1],
        "n_fnirs": fnirs_raw.shape[1],
        "idx": (tr, va, te),
        "subjects": subjects,
        "eeg": (eeg_tr, eeg_va, eeg_te),
        "fnirs": (fn_tr, fn_va, fn_te),
        "labels": (labels[tr], labels[va], labels[te]),
        "scalers": (eeg_scaler, fnirs_scaler),
    }


def fixed_split_work(cfg, device):
    print("\n" + "=" * 72)
    print("FIXED SPLIT: confusion matrices + Hybrid+RF ensemble")
    print("=" * 72)
    fs = build_fixed_split(cfg)
    (eeg_tr, eeg_va, eeg_te) = fs["eeg"]
    (fn_tr, fn_va, fn_te) = fs["fnirs"]
    (y_tr, y_va, y_te) = fs["labels"]
    n_eeg, n_fnirs = fs["n_eeg"], fs["n_fnirs"]

    tr_loader = make_loader(eeg_tr, fn_tr, y_tr, cfg, shuffle=True, drop_last=True)
    va_loader = make_loader(eeg_va, fn_va, y_va, cfg)
    te_loader = make_loader(eeg_te, fn_te, y_te, cfg)
    meta = {
        "n_eeg_features": n_eeg,
        "n_fnirs_features": n_fnirs,
        "class_weights": configured_class_weights(y_tr, cfg),
    }

    # Hybrid: load the FROZEN locked checkpoint (no retraining).
    hybrid = build_model("hybrid_cnn_lstm", n_eeg, n_fnirs, cfg).to(device)
    hybrid.load_state_dict(torch.load(FROZEN_HYBRID, map_location=device))
    h_preds, h_probs, _ = predict(hybrid, te_loader, device)

    # EEG-only + fNIRS-only ablation models (for confusion matrices only).
    set_seed(cfg.RANDOM_SEED)
    eeg_model = build_model("unimodal_eeg", n_eeg, n_fnirs, cfg)
    eeg_model, _ = train_model(
        eeg_model, tr_loader, va_loader, meta, cfg, verbose=False,
        is_unimodal=True, modality="eeg",
    )
    e_preds, _, _ = predict(eeg_model, te_loader, device, is_unimodal=True, modality="eeg")

    set_seed(cfg.RANDOM_SEED)
    fnirs_model = build_model("unimodal_fnirs", n_eeg, n_fnirs, cfg)
    fnirs_model, _ = train_model(
        fnirs_model, tr_loader, va_loader, meta, cfg, verbose=False,
        is_unimodal=True, modality="fnirs",
    )
    f_preds, _, _ = predict(fnirs_model, te_loader, device, is_unimodal=True, modality="fnirs")

    # Random Forest on flattened [eeg|fnirs] for ensembling.
    X_tr = np.hstack([eeg_tr, fn_tr])
    X_te = np.hstack([eeg_te, fn_te])
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, random_state=RF_SEED, class_weight="balanced", n_jobs=-1
    ).fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_te)

    ens_probs = 0.5 * (h_probs + rf_probs)
    ens_preds = ens_probs.argmax(axis=1)

    labels3 = list(range(cfg.NUM_CLASSES))
    conf = {
        "labels": ["Low", "Medium", "High"],
        "hybrid": confusion_matrix(y_te, h_preds, labels=labels3).tolist(),
        "eeg": confusion_matrix(y_te, e_preds, labels=labels3).tolist(),
        "fnirs": confusion_matrix(y_te, f_preds, labels=labels3).tolist(),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/fixed_split_confusion.json", "w", encoding="utf-8") as fh:
        json.dump(conf, fh, indent=2)

    hybrid_m = metrics_from(y_te, h_preds, h_probs, cfg.NUM_CLASSES)
    ens_m = metrics_from(y_te, ens_preds, ens_probs, cfg.NUM_CLASSES)
    print(f"  Fixed-split Hybrid  : {hybrid_m}")
    print(f"  Fixed-split Ensemble: {ens_m}")
    return {"hybrid": hybrid_m, "ensemble": ens_m}


# ── LOSO ──────────────────────────────────────────────────────────────────

def loso_run(cfg, model_kind, device, max_folds=None, want_rf=False):
    """Return list of per-fold dicts with labels/preds/probs (+rf probs)."""
    set_seed(cfg.RANDOM_SEED)
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, _ = dl.load_raw(return_groups=True)
    subjects = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])[
        cfg.SUBJECT_COLUMN
    ].astype(str).to_numpy()

    folds = []
    for fold, (fit_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(eeg_raw, labels, subjects), 1
    ):
        if max_folds is not None and fold > max_folds:
            break
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=cfg.VAL_SIZE, random_state=cfg.RANDOM_SEED + fold
        )
        train_rel, val_rel = next(
            val_splitter.split(fit_idx, labels[fit_idx], subjects[fit_idx])
        )
        train_idx = fit_idx[train_rel]
        val_idx = fit_idx[val_rel]

        es = RobustScaler()
        fsc = StandardScaler()
        eeg_tr = es.fit_transform(eeg_raw[train_idx])
        eeg_va = es.transform(eeg_raw[val_idx])
        eeg_te = es.transform(eeg_raw[test_idx])
        fn_tr = fsc.fit_transform(fnirs_raw[train_idx])
        fn_va = fsc.transform(fnirs_raw[val_idx])
        fn_te = fsc.transform(fnirs_raw[test_idx])

        y_tr, y_va, y_te = labels[train_idx], labels[val_idx], labels[test_idx]
        held = subjects[test_idx][0]

        meta = {
            "n_eeg_features": eeg_raw.shape[1],
            "n_fnirs_features": fnirs_raw.shape[1],
            "class_weights": configured_class_weights(y_tr, cfg),
        }
        tr_loader = make_loader(eeg_tr, fn_tr, y_tr, cfg, shuffle=True, drop_last=True)
        va_loader = make_loader(eeg_va, fn_va, y_va, cfg)
        te_loader = make_loader(eeg_te, fn_te, y_te, cfg)

        if model_kind == "hybrid":
            model = build_model("hybrid_cnn_lstm", eeg_raw.shape[1], fnirs_raw.shape[1], cfg)
            model, _ = train_model(model, tr_loader, va_loader, meta, cfg, verbose=False)
            preds, probs, y = predict(model.to(device), te_loader, device)
        elif model_kind == "fnirs":
            model = build_model("unimodal_fnirs", eeg_raw.shape[1], fnirs_raw.shape[1], cfg)
            model, _ = train_model(
                model, tr_loader, va_loader, meta, cfg, verbose=False,
                is_unimodal=True, modality="fnirs",
            )
            preds, probs, y = predict(
                model.to(device), te_loader, device, is_unimodal=True, modality="fnirs"
            )
        else:
            raise ValueError(model_kind)

        rec = {"subject": held, "labels": y, "preds": preds, "probs": probs}
        if want_rf:
            X_tr = np.hstack([eeg_tr, fn_tr])
            X_te = np.hstack([eeg_te, fn_te])
            rf = RandomForestClassifier(
                n_estimators=RF_ESTIMATORS, random_state=RF_SEED,
                class_weight="balanced", n_jobs=-1,
            ).fit(X_tr, y_tr)
            rec["rf_probs"] = rf.predict_proba(X_te)
        folds.append(rec)
        m = metrics_from(y, preds, probs, cfg.NUM_CLASSES)
        print(f"  [{model_kind:6s}] fold {fold:2d} subj {held:>2}: "
              f"acc={m['acc']:.3f} f1={m['f1_macro']:.3f} auc={m['auc']:.3f}")
    return folds


def summarize(folds, num_classes=3):
    per_fold = [metrics_from(f["labels"], f["preds"], f["probs"], num_classes) for f in folds]
    f1 = np.array([m["f1_macro"] for m in per_fold])
    acc = np.array([m["acc"] for m in per_fold])
    auc = np.array([m["auc"] for m in per_fold])
    rec = np.array([m["recall"] for m in per_fold])
    return {
        "f1": f1, "acc": acc, "auc": auc, "recall": rec, "per_fold": per_fold,
        "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)),
        "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
        "auc_mean": float(np.nanmean(auc)), "auc_std": float(np.nanstd(auc)),
        "recall_mean": np.mean(rec, axis=0).tolist(),
    }


def paired_tests(a, b, name):
    a = np.asarray(a, float); b = np.asarray(b, float)
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:  # all-zero differences
        w_stat, w_p = float("nan"), 1.0
    return {
        "comparison": name,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a - b)),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
        "n": int(len(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_norm = clone_config(
        {"PER_SUBJECT_NORMALIZE": True, "PER_SUBJECT_NORMALIZE_STAGE": "before_log"}
    )
    cfg_nonorm = clone_config({"PER_SUBJECT_NORMALIZE": False})

    fixed = fixed_split_work(cfg_norm, device)

    print("\n" + "=" * 72)
    print("LOSO Run A: normalized hybrid + fnirs-only + RF")
    print("=" * 72)
    hybrid_norm = loso_run(cfg_norm, "hybrid", device, args.max_folds, want_rf=True)
    fnirs_norm = loso_run(cfg_norm, "fnirs", device, args.max_folds)
    print("\n" + "=" * 72)
    print("LOSO Run B: non-normalized hybrid")
    print("=" * 72)
    hybrid_nonorm = loso_run(cfg_nonorm, "hybrid", device, args.max_folds)

    s_hn = summarize(hybrid_norm)
    s_fn = summarize(fnirs_norm)
    s_nn = summarize(hybrid_nonorm)

    # Ensemble (hybrid_norm + RF), per fold, LOSO
    ens_folds = []
    for f in hybrid_norm:
        ens_probs = 0.5 * (f["probs"] + f["rf_probs"])
        ens_folds.append({
            "subject": f["subject"], "labels": f["labels"],
            "preds": ens_probs.argmax(axis=1), "probs": ens_probs,
        })
    s_ens = summarize(ens_folds)

    # ── per-fold CSV ──────────────────────────────────────────────────────
    rows = []
    for tag, folds in [
        ("hybrid_normalized", hybrid_norm),
        ("fnirs_only_normalized", fnirs_norm),
        ("hybrid_non_normalized", hybrid_nonorm),
        ("ensemble_hybrid_rf_normalized", ens_folds),
    ]:
        for f in folds:
            m = metrics_from(f["labels"], f["preds"], f["probs"])
            rows.append({
                "config": tag, "subject": f["subject"],
                "acc": m["acc"], "f1_macro": m["f1_macro"], "auc": m["auc"],
                "recall_low": m["recall"][0], "recall_medium": m["recall"][1],
                "recall_high": m["recall"][2],
            })
    pd.DataFrame(rows).to_csv("outputs/loso_per_fold.csv", index=False)
    print("\nSaved outputs/loso_per_fold.csv")

    # ── significance tests (paired, by subject/fold) ─────────────────────
    sig = {
        "hybrid_vs_fnirs_f1": paired_tests(s_hn["f1"], s_fn["f1"], "Hybrid vs fNIRS-only (Macro F1)"),
        "hybrid_vs_fnirs_auc": paired_tests(s_hn["auc"], s_fn["auc"], "Hybrid vs fNIRS-only (AUC)"),
        "normalized_vs_nonnormalized_f1": paired_tests(
            s_hn["f1"], s_nn["f1"], "Normalized vs Non-normalized Hybrid (Macro F1)"
        ),
        "normalized_vs_nonnormalized_auc": paired_tests(
            s_hn["auc"], s_nn["auc"], "Normalized vs Non-normalized Hybrid (AUC)"
        ),
    }
    with open("outputs/significance_tests.json", "w", encoding="utf-8") as fh:
        json.dump(sig, fh, indent=2)
    print("Saved outputs/significance_tests.json")

    # ── ensembling results + acceptance ──────────────────────────────────
    recall_floor = float(np.min(s_ens["recall_mean"]))
    ensemble_beats = s_ens["f1_mean"] > s_hn["f1_mean"]
    no_collapse = recall_floor >= 0.10  # no class fully collapsed
    ensemble_accepted = bool(ensemble_beats and no_collapse)
    ensemble = {
        "loso": {
            "hybrid": {k: s_hn[k] for k in ["f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean"]},
            "fnirs_only": {k: s_fn[k] for k in ["f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean"]},
            "hybrid_non_normalized": {k: s_nn[k] for k in ["f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean"]},
            "ensemble_hybrid_rf": {k: s_ens[k] for k in ["f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean"]},
        },
        "fixed_split": fixed,
        "ensemble_accepted": ensemble_accepted,
        "acceptance_reason": (
            f"ensemble LOSO Macro F1={s_ens['f1_mean']:.4f} "
            f"{'>' if ensemble_beats else '<='} hybrid {s_hn['f1_mean']:.4f}; "
            f"min mean class recall={recall_floor:.3f}"
        ),
    }
    with open("outputs/ensemble_results.json", "w", encoding="utf-8") as fh:
        json.dump(ensemble, fh, indent=2)
    print("Saved outputs/ensemble_results.json")

    write_results_md(cfg_norm, s_hn, s_fn, s_nn, s_ens, sig, ensemble, fixed)
    print("\nAppended Locked Model + significance + ensembling to RESULTS.md")


def write_results_md(cfg, s_hn, s_fn, s_nn, s_ens, sig, ensemble, fixed):
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit = "(uncommitted)"

    def rec(v):
        return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"

    lines = [
        "",
        "## Locked Model",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Frozen checkpoint: `{FROZEN_HYBRID}`. Reference commit: `{commit}`.",
        "",
        "**Final configuration (frozen — Parts 2-4 consume this as a fixed artifact):**",
        "- Model: Hybrid CNN-LSTM (MLP encoders + attention/gated fusion), "
        "`FUSION_TYPE=\"attention\"`, `LOSS_FN=\"weighted_ce\"`.",
        "- Preprocessing: `PER_SUBJECT_NORMALIZE=True`, "
        "`PER_SUBJECT_NORMALIZE_STAGE=\"before_log\"`, EEG log1p + train-only "
        "RobustScaler, fNIRS signed-log + train-only StandardScaler, balanced "
        "train-fold class weights.",
        "- Dataset: `data/tuberlin_eeg_fnirs.csv` (654 full-trial samples, 26 "
        "subjects, 180 EEG + 468 HbO features), subject-grouped splits.",
        "",
        "**Final metrics:**",
        "",
        "| Evaluation | Accuracy | Macro F1 | AUC | Per-class recall [L,M,H] |",
        "|---|---:|---:|---:|---|",
        f"| Fixed split (Hybrid) | {fixed['hybrid']['acc']:.4f} | "
        f"{fixed['hybrid']['f1_macro']:.4f} | {fixed['hybrid']['auc']:.4f} | "
        f"{rec(fixed['hybrid']['recall'])} |",
        f"| LOSO (Hybrid, 26 folds) | {s_hn['acc_mean']:.4f} +/- {s_hn['acc_std']:.4f} | "
        f"{s_hn['f1_mean']:.4f} +/- {s_hn['f1_std']:.4f} | "
        f"{s_hn['auc_mean']:.4f} +/- {s_hn['auc_std']:.4f} | {rec(s_hn['recall_mean'])} |",
        f"| LOSO (fNIRS-only, 26 folds) | {s_fn['acc_mean']:.4f} +/- {s_fn['acc_std']:.4f} | "
        f"{s_fn['f1_mean']:.4f} +/- {s_fn['f1_std']:.4f} | "
        f"{s_fn['auc_mean']:.4f} +/- {s_fn['auc_std']:.4f} | {rec(s_fn['recall_mean'])} |",
        f"| LOSO (Hybrid, no per-subject norm) | {s_nn['acc_mean']:.4f} +/- {s_nn['acc_std']:.4f} | "
        f"{s_nn['f1_mean']:.4f} +/- {s_nn['f1_std']:.4f} | "
        f"{s_nn['auc_mean']:.4f} +/- {s_nn['auc_std']:.4f} | {rec(s_nn['recall_mean'])} |",
        f"| LOSO (Hybrid+RF ensemble) | {s_ens['acc_mean']:.4f} +/- {s_ens['acc_std']:.4f} | "
        f"{s_ens['f1_mean']:.4f} +/- {s_ens['f1_std']:.4f} | "
        f"{s_ens['auc_mean']:.4f} +/- {s_ens['auc_std']:.4f} | {rec(s_ens['recall_mean'])} |",
        "",
        "**Statistical significance (paired across 26 LOSO folds):**",
        "",
        "| Comparison | mean diff | paired t p | Wilcoxon p |",
        "|---|---:|---:|---:|",
    ]
    for key in ["hybrid_vs_fnirs_f1", "hybrid_vs_fnirs_auc",
                "normalized_vs_nonnormalized_f1", "normalized_vs_nonnormalized_auc"]:
        s = sig[key]
        lines.append(
            f"| {s['comparison']} | {s['mean_diff']:+.4f} | {s['t_p']:.4f} | {s['wilcoxon_p']:.4f} |"
        )
    lines += [
        "",
        f"**Ensembling verdict:** {'ADOPTED' if ensemble['ensemble_accepted'] else 'REJECTED'}. "
        f"{ensemble['acceptance_reason']}. "
        + ("The ensemble is reported as an auxiliary result; the frozen single "
           "Hybrid remains the locked model unless the ensemble is adopted above."
           if not ensemble["ensemble_accepted"] else
           "The Hybrid+RF probability-average ensemble is adopted as the locked "
           "prediction head; the frozen Hybrid weights are unchanged and the RF is "
           "layered on top at inference."),
        "",
        "Per-fold values: `outputs/loso_per_fold.csv`. Significance: "
        "`outputs/significance_tests.json`. Ensembling: "
        "`outputs/ensemble_results.json`.",
        "",
    ]
    with open("RESULTS.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
