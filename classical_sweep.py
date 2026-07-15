"""
Bounded classical-model sweep (CLASSICAL_MODEL_SWEEP.md).

Task 1: standalone 26-fold LOSO for RandomForest (baseline), XGBoost, SVM on the
        same per-subject-normalized features + per-fold train-only scalers used
        by the current RF ensemble.
Task 2: for any classical model that clearly beats standalone RF, test it in the
        Hybrid ensemble (0.5/0.5 replacement, and 3-way) and accept only if it
        beats the current best Hybrid+RF (LOSO macro F1 0.4431) with no class
        recall < 0.20.

The Hybrid network is NOT modified — it is trained per fold exactly as in
part1_rigor.py only when Task 2 is actually reached.

Outputs: outputs/classical_sweep.json  (+ appends a section to RESULTS.md)
Usage:   python classical_sweep.py [--max-folds N]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import Config
from data_loader import CognitiveLoadDataLoader, configured_class_weights
from models.architectures import build_model
from evaluate import predict as nn_predict
from trainer import train_model
from part1_rigor import (
    FROZEN_HYBRID, RF_ESTIMATORS, RF_SEED,
    make_loader, metrics_from, paired_tests, set_seed, summarize,
)

CURRENT_BEST_F1 = 0.4431  # Hybrid+RF, RESULTS.md Locked Model


def make_rf():
    return RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, random_state=RF_SEED,
        class_weight="balanced", n_jobs=-1,
    )


def make_xgb():
    # Doc config + balanced sample weights at fit time (RF and SVM both use
    # class_weight="balanced", so weight XGB too for an apples-to-apples compare).
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        objective="multi:softprob", num_class=3, random_state=42,
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
    )


def make_svm():
    return SVC(kernel="rbf", C=1.0, class_weight="balanced",
              probability=True, random_state=42)


def build_folds(cfg, max_folds=None):
    """Yield per-fold scaled train/test features + labels (identical protocol to
    part1_rigor.loso_run)."""
    set_seed(cfg.RANDOM_SEED)
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, _ = dl.load_raw(return_groups=True)
    subjects = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])[
        cfg.SUBJECT_COLUMN
    ].astype(str).to_numpy()

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
        train_idx, val_idx = fit_idx[train_rel], fit_idx[val_rel]

        es, fsc = RobustScaler(), StandardScaler()
        eeg_tr = es.fit_transform(eeg_raw[train_idx])
        eeg_va = es.transform(eeg_raw[val_idx])
        eeg_te = es.transform(eeg_raw[test_idx])
        fn_tr = fsc.fit_transform(fnirs_raw[train_idx])
        fn_va = fsc.transform(fnirs_raw[val_idx])
        fn_te = fsc.transform(fnirs_raw[test_idx])

        yield {
            "fold": fold,
            "subject": subjects[test_idx][0],
            "eeg": (eeg_tr, eeg_va, eeg_te),
            "fnirs": (fn_tr, fn_va, fn_te),
            "y": (labels[train_idx], labels[val_idx], labels[test_idx]),
            "n_eeg": eeg_raw.shape[1], "n_fnirs": fnirs_raw.shape[1],
        }


def classical_probs(fold):
    (eeg_tr, _, eeg_te) = fold["eeg"]
    (fn_tr, _, fn_te) = fold["fnirs"]
    y_tr = fold["y"][0]
    X_tr = np.hstack([eeg_tr, fn_tr])
    X_te = np.hstack([eeg_te, fn_te])
    sw = compute_sample_weight("balanced", y_tr)

    rf = make_rf().fit(X_tr, y_tr)
    xgb = make_xgb().fit(X_tr, y_tr, sample_weight=sw)
    svm = make_svm().fit(X_tr, y_tr)
    return {
        "rf": rf.predict_proba(X_te),
        "xgb": xgb.predict_proba(X_te),
        "svm": svm.predict_proba(X_te),
    }


def hybrid_probs(fold, cfg, device):
    (eeg_tr, eeg_va, eeg_te) = fold["eeg"]
    (fn_tr, fn_va, fn_te) = fold["fnirs"]
    y_tr, y_va, y_te = fold["y"]
    meta = {
        "n_eeg_features": fold["n_eeg"], "n_fnirs_features": fold["n_fnirs"],
        "class_weights": configured_class_weights(y_tr, cfg),
    }
    tr = make_loader(eeg_tr, fn_tr, y_tr, cfg, shuffle=True, drop_last=True)
    va = make_loader(eeg_va, fn_va, y_va, cfg)
    te = make_loader(eeg_te, fn_te, y_te, cfg)
    model = build_model("hybrid_cnn_lstm", fold["n_eeg"], fold["n_fnirs"], cfg)
    model, _ = train_model(model, tr, va, meta, cfg, verbose=False)
    _, probs, y = nn_predict(model.to(device), te, device)
    return probs, y


def loso_metrics(fold_labels, fold_probs, num_classes=3):
    """Summarize LOSO from per-fold (labels, probs)."""
    folds = [{"labels": lb, "preds": pr.argmax(1), "probs": pr}
             for lb, pr in zip(fold_labels, fold_probs)]
    return summarize(folds, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("cpu")

    # ── Task 1: standalone classical LOSO ────────────────────────────────
    print("=" * 72)
    print("TASK 1 — standalone classical LOSO (RF baseline vs XGBoost, SVM)")
    print("=" * 72)
    fold_labels, probs = {"rf": [], "xgb": [], "svm": []}, {"rf": [], "xgb": [], "svm": []}
    labels_list = []
    fold_cache = []
    for fold in build_folds(cfg, args.max_folds):
        cp = classical_probs(fold)
        labels_list.append(fold["y"][2])
        for k in ("rf", "xgb", "svm"):
            probs[k].append(cp[k])
        fold_cache.append(fold)
        m = {k: metrics_from(fold["y"][2], cp[k].argmax(1), cp[k], cfg.NUM_CLASSES)["f1_macro"]
             for k in ("rf", "xgb", "svm")}
        print(f"  fold {fold['fold']:2d} subj {fold['subject']:>2}: "
              f"RF={m['rf']:.3f} XGB={m['xgb']:.3f} SVM={m['svm']:.3f}")

    stand = {k: loso_metrics(labels_list, probs[k], cfg.NUM_CLASSES) for k in ("rf", "xgb", "svm")}
    print("\nStandalone LOSO (26-fold):")
    for k in ("rf", "xgb", "svm"):
        s = stand[k]
        print(f"  {k.upper():4s}  macroF1={s['f1_mean']:.4f}+/-{s['f1_std']:.4f}  "
              f"acc={s['acc_mean']:.4f}  auc={s['auc_mean']:.4f}  "
              f"recall={[round(x,3) for x in s['recall_mean']]}")

    rf_f1 = stand["rf"]["f1_mean"]
    winners = []            # CLEARLY beat RF (numeric + significant) -> proceed to Task 2
    numeric_over_rf = []    # numerically ahead, reported but not gate-passing
    sig_vs_rf = {}
    for k in ("xgb", "svm"):
        sig = paired_tests(stand[k]["f1"], stand["rf"]["f1"], f"{k} vs RF (macro F1)")
        sig_vs_rf[k] = sig
        numeric = stand[k]["f1_mean"] > rf_f1
        # "Clear beat" (doc Task 1): a REAL individual improvement, not noise.
        # Require both a numeric win and a significant paired difference.
        clear = numeric and sig["wilcoxon_p"] < 0.05
        if numeric:
            numeric_over_rf.append(k)
        if clear:
            winners.append(k)
        verdict = ("CLEARLY beats RF" if clear else
                   "ties RF (not significant)" if numeric else "worse than RF")
        print(f"  {k.upper()} vs RF standalone: diff={sig['mean_diff']:+.4f} "
              f"(t_p={sig['t_p']:.3f}, wilcoxon_p={sig['wilcoxon_p']:.3f}) -> {verdict}")

    results = {
        "standalone_loso": {k: {kk: stand[k][kk] for kk in
                               ("f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean")}
                            for k in ("rf", "xgb", "svm")},
        "rf_standalone_f1": rf_f1,
        "significance_vs_rf": sig_vs_rf,
        "task1_clear_winners_over_rf": winners,
        "task1_numeric_over_rf": numeric_over_rf,
        "current_best_ensemble_f1": CURRENT_BEST_F1,
    }

    # ── Task 2: ensemble tests ───────────────────────────────────────────
    # Doc gate: only a model that CLEARLY (significantly) beats standalone RF
    # warrants adopting into the ensemble. We still REPORT ensembles for any
    # model that is merely numerically ahead (informational, to show even the
    # ensemble doesn't help), but new_best requires a significant Task-1 win.
    report_for = numeric_over_rf
    if not report_for:
        print("\nTASK 2 — SKIPPED: no classical model even numerically beat RF.")
        results["task2"] = {"run": False, "reason": "no classical model beat standalone RF"}
        results["new_best"] = False
    else:
        note = ("clear (significant) winners" if winners
                else "numerically-ahead-only models — reported for the record, "
                     "NOT gate-passing")
        print("\n" + "=" * 72)
        print(f"TASK 2 — ensemble tests for {report_for} ({note}); training Hybrid per fold")
        print("=" * 72)
        hy_probs, hy_labels = [], []
        for fold in fold_cache:
            hp, hy = hybrid_probs(fold, cfg, device)
            hy_probs.append(hp); hy_labels.append(hy)
            print(f"  hybrid fold {fold['fold']:2d} subj {fold['subject']:>2} done")

        def ens(members):
            out = []
            for i in range(len(hy_probs)):
                stack = [hy_probs[i]] + [probs[m][i] for m in members]
                out.append(np.mean(stack, axis=0))
            return loso_metrics(hy_labels, out, cfg.NUM_CLASSES)

        combos = {"hybrid+rf": ["rf"]}
        for w in report_for:
            combos[f"hybrid+{w}"] = [w]
            combos[f"hybrid+rf+{w}"] = ["rf", w]
        ens_res = {}
        print("\nEnsemble LOSO (26-fold):")
        for name, members in combos.items():
            s = ens(members)
            floor = float(min(s["recall_mean"]))
            ens_res[name] = {k: s[k] for k in ("f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean")}
            ens_res[name]["min_recall"] = floor
            print(f"  {name:16s} macroF1={s['f1_mean']:.4f}  acc={s['acc_mean']:.4f}  "
                  f"auc={s['auc_mean']:.4f}  min_recall={floor:.3f}")
        results["task2"] = {"run": True, "gate_passing": bool(winners), "ensembles": ens_res}

        best_name = max(ens_res, key=lambda n: ens_res[n]["f1_mean"])
        best = ens_res[best_name]
        # Adopt ONLY if a classical model clearly (significantly) beat RF AND the
        # resulting ensemble beats the current best without class collapse.
        accept = (bool(winners) and best["f1_mean"] > CURRENT_BEST_F1
                  and best["min_recall"] >= 0.20 and best_name != "hybrid+rf")
        results["new_best"] = bool(accept)
        results["best_config"] = best_name
        results["best_f1"] = best["f1_mean"]
        print(f"\nBest ensemble: {best_name} macroF1={best['f1_mean']:.4f} "
              f"(current best hybrid+rf {CURRENT_BEST_F1}); "
              f"{'ACCEPT — new best' if accept else 'no improvement over current best'}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classical_sweep.json", "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2)
    print("\nSaved outputs/classical_sweep.json")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


if __name__ == "__main__":
    main()
