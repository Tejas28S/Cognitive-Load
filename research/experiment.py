"""
experiment.py — one experiment = one model spec evaluated over one fold set.

An experiment never sees the lockbox subjects.  The only function that does is
`evaluate_lockbox`, which is called once, at the end of a campaign, for the
single spec that the development folds selected.

Everything an experiment reports is derived from `predictions.csv`, which
holds one row per evaluated window with its subject, trial, window id, fold,
true label, predicted label and full probability vector.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import leakage, metrics as M
from .datamod import loso_folds, scale_fold
from .runctx import write_json
from .train import modality_gates, predict, train_fold


def _prob_cols(class_names):
    return [f"prob_{c}" for c in class_names]


def run_experiment(ds, spec, fold_specs, exp_id, out_dir, seed=42, device="cpu",
                   log=print, keep_models=False, use_domains=False):
    """Train + evaluate one spec across `fold_specs`. Returns the record."""
    os.makedirs(out_dir, exist_ok=True)
    class_names = ds.class_names
    pcols = _prob_cols(class_names)
    t0 = time.time()

    rows, histories, fold_audits, fold_records = [], [], [], []
    gates_accum = []
    models = []

    for fold in fold_specs:
        scaled = scale_fold(ds, fold)
        audit = leakage.audit_fold(ds, fold, scaled, strict=True)
        fold_audits.append(audit)

        domains = None
        if use_domains:
            # Domain labels are the TRAIN subjects only, densely re-indexed.
            tr_subj = ds.subjects[fold["train_idx"]]
            uniq = np.unique(tr_subj)
            mapping = {s: i for i, s in enumerate(uniq)}
            domains = {
                "train": np.array([mapping[s] for s in tr_subj], dtype=np.int64),
                "n_domains": len(uniq),
            }

        res = train_fold(
            spec, ds.schema, ds.n_classes, scaled, domains=domains,
            seed=seed + fold["fold"], device=device, log=log,
        )
        eeg_te, fn_te, y_te = scaled["test"]
        probs = predict(res["model"], eeg_te, fn_te, device=device)
        preds = probs.argmax(axis=1)

        g = modality_gates(res["model"], eeg_te, fn_te, device=device)
        if g is not None:
            gates_accum.append((g[0].mean(), g[1].mean()))

        te = fold["test_idx"]
        block = pd.DataFrame({
            "experiment_id": exp_id,
            "fold": fold["fold"],
            "held_out_subject": (
                fold["held_out_subject"] if fold["held_out_subject"] is not None else -1
            ),
            "subject_id": ds.subjects[te],
            "trial_id": ds.trials[te],
            "window_id": ds.window_ids[te],
            "window_start_sec": ds.window_start[te],
            "row_index": te,
            "y_true": y_te,
            "y_pred": preds,
            "y_true_name": [class_names[i] for i in y_te],
            "y_pred_name": [class_names[i] for i in preds],
            "confidence": probs.max(axis=1),
            "correct": (preds == y_te).astype(int),
        })
        for i, c in enumerate(pcols):
            block[c] = probs[:, i]
        rows.append(block)

        h = dict(res["history"])
        h["fold"] = fold["fold"]
        h["held_out_subject"] = fold["held_out_subject"]
        h["best_epoch"] = res["best_epoch"]
        histories.append(h)

        fm = M.compute_metrics(y_te, preds, probs, ds.n_classes, class_names)
        fold_records.append({
            "fold": fold["fold"],
            "held_out_subject": fold["held_out_subject"],
            "best_epoch": res["best_epoch"],
            "epochs_run": res["epochs_run"],
            "best_val_f1_macro": res["best_val_f1_macro"],
            "train_seconds": res["train_seconds"],
            "final_train_f1_macro": h["train_f1_macro"][res["best_epoch"] - 1],
            "final_val_f1_macro": h["val_f1_macro"][res["best_epoch"] - 1],
            "final_train_loss": h["train_loss"][res["best_epoch"] - 1],
            "final_val_loss": h["val_loss"][res["best_epoch"] - 1],
            "test_accuracy": fm["accuracy"],
            "test_f1_macro": fm["f1_macro"],
        })
        log(f"      fold {fold['fold']:>2} subj {fold['held_out_subject']}: "
            f"val_f1={res['best_val_f1_macro']:.4f} "
            f"test_acc={fm['accuracy']:.4f} test_f1={fm['f1_macro']:.4f} "
            f"ep={res['best_epoch']}/{res['epochs_run']}")

        if keep_models:
            models.append({"fold": fold["fold"], "model": res["model"],
                           "scalers": scaled["scalers"]})
        else:
            del res["model"]

    preds_df = pd.concat(rows, ignore_index=True)
    preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    preds_df[
        ["experiment_id", "fold", "subject_id", "trial_id", "window_id", "y_true"]
        + pcols
    ].to_csv(os.path.join(out_dir, "probabilities.csv"), index=False)

    # ── pooled metrics: every held-out window, one confusion matrix ─────
    y_true = preds_df["y_true"].to_numpy(int)
    y_pred = preds_df["y_pred"].to_numpy(int)
    probs = preds_df[pcols].to_numpy(float)
    pooled = M.compute_metrics(y_true, y_pred, probs, ds.n_classes, class_names)
    curves = M.curve_points(y_true, probs, ds.n_classes, class_names)

    # ── per-subject + fold aggregate: the cross-subject summary ─────────
    per_subject = M.per_subject_metrics(preds_df, ds.n_classes, class_names, pcols)
    fold_agg = M.aggregate_folds(per_subject)

    # ── generalization gap, measured at each fold's selected epoch ──────
    tr_f1 = np.array([r["final_train_f1_macro"] for r in fold_records], float)
    va_f1 = np.array([r["final_val_f1_macro"] for r in fold_records], float)
    te_f1 = np.array([r["test_f1_macro"] for r in fold_records], float)
    gap = {
        "train_f1_macro_mean": float(tr_f1.mean()),
        "val_f1_macro_mean": float(va_f1.mean()),
        "heldout_f1_macro_mean": float(te_f1.mean()),
        "train_minus_val": float((tr_f1 - va_f1).mean()),
        "train_minus_heldout": float((tr_f1 - te_f1).mean()),
        "val_minus_heldout": float((va_f1 - te_f1).mean()),
        "mean_best_epoch": float(np.mean([r["best_epoch"] for r in fold_records])),
        "mean_epochs_run": float(np.mean([r["epochs_run"] for r in fold_records])),
    }

    record = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "spec": spec,
        "seed": seed,
        "device": device,
        "n_folds": len(fold_specs),
        "fold_subjects": [f["held_out_subject"] for f in fold_specs],
        "n_params": None,   # filled in below from a freshly built model
        "runtime_seconds": float(time.time() - t0),
        "pooled": pooled,
        "fold_aggregate": fold_agg,
        "per_subject": per_subject,
        "per_fold": fold_records,
        "generalization_gap": gap,
        "leakage_audit_passed": all(a["passed"] for a in fold_audits),
        "modality_gates": (
            {
                "eeg_mean": float(np.mean([g[0] for g in gates_accum])),
                "fnirs_mean": float(np.mean([g[1] for g in gates_accum])),
            }
            if gates_accum else None
        ),
        "status": "completed",
        "warnings": [],
    }
    # parameter count from a freshly built model (folds all share the spec)
    from .archs import build_model, count_params
    record["n_params"] = count_params(build_model(spec, ds.schema, ds.n_classes))

    write_json(os.path.join(out_dir, "metrics.json"), pooled)
    write_json(os.path.join(out_dir, "curves.json"), curves)
    write_json(os.path.join(out_dir, "per_subject_metrics.json"), per_subject)
    write_json(os.path.join(out_dir, "fold_aggregate.json"), fold_agg)
    write_json(os.path.join(out_dir, "history.json"), histories)
    write_json(os.path.join(out_dir, "leakage_audit.json"), fold_audits)
    write_json(os.path.join(out_dir, "record.json"),
               {k: v for k, v in record.items() if k != "per_subject"})
    pd.DataFrame(per_subject).to_csv(
        os.path.join(out_dir, "per_subject_metrics.csv"), index=False
    )
    pd.DataFrame(M.metrics_to_rows(pooled)).to_csv(
        os.path.join(out_dir, "metrics.csv"), index=False
    )

    return record, preds_df, histories, curves, models


def mean_history(histories):
    """Average the per-fold learning curves onto a common epoch axis.

    Folds stop at different epochs, so each point is the mean over the folds
    that were still running; `n_folds` records how many that was.
    """
    max_len = max(len(h["epoch"]) for h in histories)
    keys = [k for k in histories[0]
            if k.startswith(("train_", "val_")) or k == "lr"]
    out = {"epoch": list(range(1, max_len + 1)), "n_folds": []}
    for k in keys:
        out[k] = []
    for i in range(max_len):
        contributing = 0
        for k in keys:
            vals = [h[k][i] for h in histories
                    if i < len(h[k]) and h[k][i] is not None]
            out[k].append(float(np.mean(vals)) if vals else None)
        contributing = sum(1 for h in histories if i < len(h["epoch"]))
        out["n_folds"].append(contributing)
    return out
