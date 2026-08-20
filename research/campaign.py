"""
campaign.py — the controlled experiment loop.

Protocol
--------
The 26 subjects are split once, deterministically, into

    development pool (18 subjects)   -> every experiment, every comparison
    lockbox          ( 8 subjects)   -> touched exactly once, at the very end

Experiments are compared by mean macro F1 over 18-fold development LOSO.  The
lockbox is never used to choose an architecture, a hyperparameter, a scheduler,
an epoch or a threshold.  That separation is what makes the final number an
estimate rather than a selection artifact.

The search is staged and greedy: each stage varies ONE axis, the winner is
carried into the next stage, and every arm is recorded whether it won or lost.
Losing arms are kept in the leaderboard as negative results.

Selection rule (fixed in advance, applied mechanically):
  1. rank by mean development-LOSO macro F1;
  2. among arms within one standard error of the best, take the one with the
     fewest parameters (parsimony tie-break);
  3. a challenger only replaces the incumbent if it is strictly better by rule
     1-2, so ties keep the simpler incumbent.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import explain, leakage, plots
from research import metrics as M
from research.archs import build_model, count_params
from research.datamod import (
    Dataset, holdout_fold, loso_folds, scale_fold, split_dev_lockbox,
)
from research.experiment import mean_history, run_experiment
from research.runctx import RunContext, write_json
from research.train import embeddings, predict, train_fold

SELECT_METRIC = "f1_macro"

DATASETS = {
    "sliding_10s_5s": "data/tuberlin_sliding.csv",
    "sliding_10s_1s": "data/tuberlin_sliding_augmented.csv",
    "sliding_8s_4s": "data/tuberlin_sliding_8s_4s.csv",
}

# The served binary model's architecture (serving/model_config.json) plus the
# training hyperparameters in config.py. This is the baseline; it is recorded
# before anything is changed and never overwritten.
BASELINE_SPEC = {
    "encoder": "mlp",
    "proj_dim": 96,
    "enc_dim": 48,
    "n_blocks": 2,
    "encoder_dropout": 0.5,
    "fusion": "attention",
    "fc_units": [48, 24],
    "fc_dropout": 0.6,
    "optimizer": "adamw",
    "lr": 3e-4,
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
    "use_dann": False,
    "dann_lambda": 0.0,
}

BASELINE_DATA = {
    "dataset": "sliding_10s_5s",
    "binary": True,          # Low (0-back) vs High (3-back); False keeps all three
    "per_subject_norm": "transductive",
    "per_subject_norm_stage": "before_log",
    "drop_hbr": False,
}


def data_key(d):
    return json.dumps(d, sort_keys=True)


class DataCache:
    def __init__(self, root, log=print):
        self.root = root
        self.log = log
        self._cache = {}

    def get(self, dcfg):
        key = data_key(dcfg)
        if key not in self._cache:
            path = os.path.join(self.root, DATASETS[dcfg["dataset"]])
            self.log(f"   loading dataset {dcfg['dataset']} ({dcfg})")
            self._cache[key] = Dataset(
                path,
                binary=dcfg.get("binary", True),
                per_subject_norm=dcfg["per_subject_norm"],
                per_subject_norm_stage=dcfg["per_subject_norm_stage"],
                calibration_windows=dcfg.get("calibration_windows", 30),
                drop_hbr=dcfg.get("drop_hbr", False),
            )
        return self._cache[key]


def paired_vs(record, baseline_record):
    """Paired significance across the folds the two arms share."""
    a = {r["subject_id"]: r[SELECT_METRIC] for r in record["per_subject"]}
    b = {r["subject_id"]: r[SELECT_METRIC] for r in baseline_record["per_subject"]}
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    x = np.array([a[s] for s in common], float)
    y = np.array([b[s] for s in common], float)
    t_stat, t_p = stats.ttest_rel(x, y)
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0
    diff = x - y
    return {
        "n_folds": len(common),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "significant_at_0.05": bool(t_p < 0.05),
    }


def score_of(record):
    agg = record["fold_aggregate"][SELECT_METRIC]
    return agg["mean"] if agg["mean"] is not None else -np.inf


def stderr_of(record):
    agg = record["fold_aggregate"][SELECT_METRIC]
    if agg["std"] is None or not agg["n"]:
        return 0.0
    return agg["std"] / np.sqrt(agg["n"])


def select_best(records, incumbent=None):
    """One-standard-error rule, with the incumbent winning every tie.

    Ranking by mean alone would chase noise: with 18 folds and a fold-to-fold
    standard deviation around 0.08, the standard error on mean macro F1 is
    roughly 0.02, so a +0.002 "win" is indistinguishable from zero. Drifting on
    differences that size across ten stages would accumulate into a selection
    artifact rather than a better model.

    So: take every arm within one standard error of the leader, and inside that
    set prefer (1) the incumbent, then (2) the smallest model. A challenger has
    to clear the noise floor to be adopted.
    """
    ranked = sorted(records, key=score_of, reverse=True)
    leader = ranked[0]
    tol = stderr_of(leader)
    tied = [r for r in ranked if score_of(r) >= score_of(leader) - tol]

    if incumbent is not None:
        for r in tied:
            if r["experiment_id"] == incumbent["experiment_id"]:
                return r
    return min(tied, key=lambda r: r["n_params"])


class Campaign:
    def __init__(self, run, root, device="cpu", seed=42, n_lockbox=8,
                 val_fraction=0.2, quick=False, limit_subjects=None):
        self.run = run
        self.root = root
        self.device = device
        self.seed = seed
        self.val_fraction = val_fraction
        self.quick = quick
        self.cache = DataCache(root, log=run.log)
        self.records = []
        self.exp_log_path = run.path("experiments", "experiment_log.txt")

        base_ds = self.cache.get(BASELINE_DATA)
        all_subjects = base_ds.unique_subjects
        if limit_subjects:
            # smoke-test only: shrink the whole cohort so every code path runs
            all_subjects = all_subjects[:limit_subjects]
            run.warn(
                f"limit_subjects={limit_subjects}: this run uses a REDUCED cohort "
                "and its metrics are not a valid performance estimate"
            )
        self.all_subjects = all_subjects
        self.dev_subjects, self.lockbox_subjects = split_dev_lockbox(
            all_subjects, n_lockbox=n_lockbox, seed=seed
        )
        if quick:
            self.dev_subjects = self.dev_subjects[:4]
        run.log(f"   development subjects ({len(self.dev_subjects)}): "
                f"{self.dev_subjects.tolist()}")
        run.log(f"   LOCKBOX subjects ({len(self.lockbox_subjects)}): "
                f"{self.lockbox_subjects.tolist()}  [never used for selection]")

    # ── logging ────────────────────────────────────────────────────────
    def elog(self, msg):
        with open(self.exp_log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        self.run.log(msg)

    # ── one arm ────────────────────────────────────────────────────────
    def evaluate_spec(self, exp_id, spec, dcfg, stage, note=""):
        ds = self.cache.get(dcfg)
        folds = list(loso_folds(ds, self.dev_subjects,
                                val_fraction=self.val_fraction, seed=self.seed))
        out_dir = self.run.path("experiments", exp_id)
        self.elog(f"\n[{stage}] {exp_id} — {note}")
        t0 = time.time()
        record, preds, hists, curves, _ = run_experiment(
            ds, spec, folds, exp_id, out_dir, seed=self.seed,
            device=self.device, log=self.run.log, use_domains=bool(spec.get("use_dann")),
        )
        record.update({
            "stage": stage, "note": note, "data": dcfg,
            "dataset_name": dcfg["dataset"],
            "n_dev_subjects": int(len(self.dev_subjects)),
            "classification_mode": ds.info()["classification_mode"],
            "class_names": ds.class_names,
        })
        record["mean_history"] = mean_history(hists)
        if self.records:
            record["vs_baseline"] = paired_vs(record, self.records[0])
        write_json(os.path.join(out_dir, "record.json"),
                   {k: v for k, v in record.items() if k != "per_subject"})
        self.records.append(record)
        agg = record["fold_aggregate"][SELECT_METRIC]
        self.elog(
            f"    -> dev-LOSO macro F1 {agg['mean']:.4f} +/- {agg['std']:.4f}  "
            f"acc {record['fold_aggregate']['accuracy']['mean']:.4f}  "
            f"params {record['n_params']:,}  ({time.time() - t0:.0f}s)"
        )
        self.save_leaderboard()
        return record

    # ── leaderboard ────────────────────────────────────────────────────
    def save_leaderboard(self):
        rows = []
        for r in self.records:
            agg = r["fold_aggregate"]
            rows.append({
                "experiment_id": r["experiment_id"],
                "stage": r["stage"],
                "note": r["note"],
                "dataset": r["dataset_name"],
                "encoder": r["spec"]["encoder"],
                "fusion": r["spec"]["fusion"],
                "n_params": r["n_params"],
                "dev_loso_f1_macro_mean": agg[SELECT_METRIC]["mean"],
                "dev_loso_f1_macro_std": agg[SELECT_METRIC]["std"],
                "dev_loso_accuracy_mean": agg["accuracy"]["mean"],
                "dev_loso_balanced_acc_mean": agg["balanced_accuracy"]["mean"],
                "dev_loso_auc_mean": agg["roc_auc"]["mean"],
                "dev_loso_mcc_mean": agg["mcc"]["mean"],
                "dev_loso_kappa_mean": agg["cohen_kappa"]["mean"],
                "pooled_f1_macro": r["pooled"]["f1_macro"],
                "pooled_accuracy": r["pooled"]["accuracy"],
                "train_minus_heldout_f1": r["generalization_gap"]["train_minus_heldout"],
                "mean_best_epoch": r["generalization_gap"]["mean_best_epoch"],
                "runtime_seconds": r["runtime_seconds"],
                "vs_baseline_mean_diff": (r.get("vs_baseline") or {}).get("mean_diff"),
                "vs_baseline_t_p": (r.get("vs_baseline") or {}).get("t_p"),
                "vs_baseline_wilcoxon_p": (r.get("vs_baseline") or {}).get("wilcoxon_p"),
                "leakage_audit_passed": r["leakage_audit_passed"],
            })
        df = pd.DataFrame(rows).sort_values(
            "dev_loso_f1_macro_mean", ascending=False
        )
        df.to_csv(self.run.path("experiments", "leaderboard.csv"), index=False)
        write_json(
            self.run.path("experiments", "experiment_results.json"),
            [{k: v for k, v in r.items() if k != "mean_history"} for r in self.records],
        )
        return df

    # ── lockbox ────────────────────────────────────────────────────────
    def evaluate_lockbox(self, spec, dcfg, exp_id="FINAL_lockbox"):
        """One shot, on subjects no experiment ever saw."""
        ds = self.cache.get(dcfg)
        folds = list(loso_folds(ds, self.lockbox_subjects,
                                val_fraction=self.val_fraction, seed=self.seed))
        # Each lockbox fold trains on all development subjects plus the other
        # lockbox subjects' data is excluded entirely: restrict the pool.
        adjusted = []
        for f in folds:
            held = f["held_out_subject"]
            train_pool_subjects = self.dev_subjects
            g = holdout_fold(ds, train_pool_subjects, [held],
                             val_fraction=self.val_fraction,
                             seed=self.seed + int(held))
            g["fold"] = f["fold"]
            g["held_out_subject"] = held
            adjusted.append(g)
        out_dir = self.run.path("experiments", exp_id)
        self.elog(f"\n[FINAL] lockbox evaluation on subjects "
                  f"{self.lockbox_subjects.tolist()} (never used for selection)")
        record, preds, hists, curves, models = run_experiment(
            ds, spec, adjusted, exp_id, out_dir, seed=self.seed,
            device=self.device, log=self.run.log,
            use_domains=bool(spec.get("use_dann")), keep_models=True,
        )
        record.update({"stage": "final", "note": "lockbox (held-out subjects)",
                       "data": dcfg, "dataset_name": dcfg["dataset"],
                       "class_names": ds.class_names})
        record["mean_history"] = mean_history(hists)
        return record, preds, hists, curves, models, ds

    def evaluate_full_loso(self, spec, dcfg, exp_id="FINAL_loso26"):
        """The project's headline protocol: LOSO over all subjects."""
        ds = self.cache.get(dcfg)
        allsub = self.all_subjects
        folds = list(loso_folds(ds, allsub, val_fraction=self.val_fraction,
                                seed=self.seed))
        out_dir = self.run.path("experiments", exp_id)
        self.elog(f"\n[FINAL] {len(folds)}-fold LOSO over all subjects")
        record, preds, hists, curves, models = run_experiment(
            ds, spec, folds, exp_id, out_dir, seed=self.seed,
            device=self.device, log=self.run.log,
            use_domains=bool(spec.get("use_dann")), keep_models=True,
        )
        record.update({"stage": "final", "note": f"{len(folds)}-fold LOSO (all subjects)",
                       "data": dcfg, "dataset_name": dcfg["dataset"],
                       "class_names": ds.class_names})
        record["mean_history"] = mean_history(hists)
        return record, preds, hists, curves, models, ds
