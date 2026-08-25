#!/usr/bin/env python
"""
ensemble_campaign.py — deep-ensemble screen for the binary Hybrid model.

Not a repeat of anything in the 57-config history (42 original + 15 in
target_campaign.py): every prior attempt varied ONE model's capacity,
regularization or data. This trains N=5 INDEPENDENTLY SEEDED copies of the
served architecture per fold and averages their softmax probabilities
(a standard deep ensemble, Lakshminarayanan et al. 2017). The mechanism is
variance reduction across independently-initialized models, not a capacity
or regularization change to any single model -- genuinely untried here.

Protocol (unchanged from the rest of the campaign):
  - dev-LOSO (18 folds) is the cheap screening signal, compared against the
    baseline dev-LOSO score with the SAME one-SE / paired-significance rule
    used everywhere else in this project.
  - Every fold still runs the full leakage audit before any model in the
    ensemble trains. The audit runs once per fold (the data split, not the
    model, is what's being audited), not once per ensemble member.
  - Only promoted to the full 26-fold LOSO + lockbox if it clears baseline
    on the development screen -- same discipline as everything else here.

    .venv/Scripts/python ensemble_campaign.py --stage dev
    .venv/Scripts/python ensemble_campaign.py --stage final
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))

from research import leakage
from research import metrics as M
from research.campaign import BASELINE_DATA, BASELINE_SPEC, Campaign, paired_vs
from research.datamod import holdout_fold, loso_folds, scale_fold
from research.runctx import RunContext, write_json
from research.train import predict, train_fold

N_MEMBERS = 5


def _prob_cols(class_names):
    return [f"prob_{c}" for c in class_names]


def run_ensemble_experiment(ds, spec, fold_specs, exp_id, out_dir, base_seed,
                            n_members, device, log, keep_models=False):
    """Same shape as research.experiment.run_experiment, but each fold trains
    N independently-seeded models and averages their softmax probabilities.
    """
    os.makedirs(out_dir, exist_ok=True)
    class_names = ds.class_names
    pcols = _prob_cols(class_names)
    t0 = time.time()

    rows, fold_records, fold_audits, models = [], [], [], []
    member_solo_accs = []  # diagnostic: how much does averaging actually buy us

    for fold in fold_specs:
        scaled = scale_fold(ds, fold)
        audit = leakage.audit_fold(ds, fold, scaled, strict=True)
        fold_audits.append(audit)

        eeg_tr, fn_tr, y_tr = scaled["train"]
        eeg_te, fn_te, y_te = scaled["test"]
        member_probs_test, member_probs_train = [], []
        member_final_train_f1, member_final_train_acc = [], []
        fold_models = []

        for m in range(n_members):
            seed = base_seed + fold["fold"] * 1000 + m
            res = train_fold(spec, ds.schema, ds.n_classes, scaled, domains=None,
                             seed=seed, device=device, log=log)
            p_test = predict(res["model"], eeg_te, fn_te, device=device)
            p_train = predict(res["model"], eeg_tr, fn_tr, device=device)
            member_probs_test.append(p_test)
            member_probs_train.append(p_train)
            train_pred = p_train.argmax(axis=1)
            member_final_train_acc.append(float((train_pred == y_tr).mean()))
            member_final_train_f1.append(
                M.compute_metrics(y_tr, train_pred, p_train, ds.n_classes, class_names)["f1_macro"]
            )
            fold_models.append(res["model"])
            if not keep_models:
                del res["model"]

        ens_probs_test = np.mean(member_probs_test, axis=0)
        ens_probs_train = np.mean(member_probs_train, axis=0)
        preds = ens_probs_test.argmax(axis=1)
        train_preds = ens_probs_train.argmax(axis=1)

        member_solo_accs.append(float(np.mean([
            (mp.argmax(axis=1) == y_te).mean() for mp in member_probs_test
        ])))

        te = fold["test_idx"]
        block = pd.DataFrame({
            "experiment_id": exp_id, "fold": fold["fold"],
            "held_out_subject": fold["held_out_subject"] if fold["held_out_subject"] is not None else -1,
            "subject_id": ds.subjects[te], "trial_id": ds.trials[te], "window_id": ds.window_ids[te],
            "row_index": te, "y_true": y_te, "y_pred": preds,
            "y_true_name": [class_names[i] for i in y_te],
            "y_pred_name": [class_names[i] for i in preds],
            "confidence": ens_probs_test.max(axis=1),
            "correct": (preds == y_te).astype(int),
        })
        for i, c in enumerate(pcols):
            block[c] = ens_probs_test[:, i]
        rows.append(block)

        ens_test_f1 = M.compute_metrics(y_te, preds, ens_probs_test, ds.n_classes, class_names)["f1_macro"]
        ens_test_acc = float((preds == y_te).mean())
        ens_train_f1 = M.compute_metrics(y_tr, train_preds, ens_probs_train, ds.n_classes, class_names)["f1_macro"]
        fold_records.append({
            "fold": fold["fold"], "held_out_subject": fold["held_out_subject"],
            "ensemble_test_accuracy": ens_test_acc, "ensemble_test_f1_macro": ens_test_f1,
            "ensemble_train_f1_macro": ens_train_f1,
            "member_solo_mean_test_accuracy": member_solo_accs[-1],
            "member_mean_train_f1_macro": float(np.mean(member_final_train_f1)),
        })
        log(f"      fold {fold['fold']:>2} subj {fold['held_out_subject']}: "
            f"ensemble_acc={ens_test_acc:.4f} ensemble_f1={ens_test_f1:.4f} "
            f"(mean single-model acc={member_solo_accs[-1]:.4f})  ({n_members} members)")

        if keep_models:
            models.append({"fold": fold["fold"], "models": fold_models, "scalers": scaled["scalers"]})

    preds_df = pd.concat(rows, ignore_index=True)
    preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    y_true = preds_df["y_true"].to_numpy(int)
    y_pred = preds_df["y_pred"].to_numpy(int)
    probs = preds_df[pcols].to_numpy(float)
    pooled = M.compute_metrics(y_true, y_pred, probs, ds.n_classes, class_names)
    per_subject = M.per_subject_metrics(preds_df, ds.n_classes, class_names, pcols)
    fold_agg = M.aggregate_folds(per_subject)

    tr_f1 = np.array([r["ensemble_train_f1_macro"] for r in fold_records], float)
    te_f1 = np.array([r["ensemble_test_f1_macro"] for r in fold_records], float)
    gap = {
        "train_f1_macro_mean": float(tr_f1.mean()),
        "heldout_f1_macro_mean": float(te_f1.mean()),
        "train_minus_heldout": float((tr_f1 - te_f1).mean()),
        "mean_member_solo_test_accuracy": float(np.mean(member_solo_accs)),
        "ensemble_vs_solo_accuracy_gain": float(
            fold_agg["accuracy"]["mean"] - np.mean(member_solo_accs)
        ),
    }

    record = {
        "experiment_id": exp_id, "spec": spec, "n_members": n_members,
        "n_folds": len(fold_specs), "runtime_seconds": float(time.time() - t0),
        "pooled": pooled, "fold_aggregate": fold_agg, "per_subject": per_subject,
        "per_fold": fold_records, "generalization_gap": gap,
        "leakage_audit_passed": all(a["passed"] for a in fold_audits),
        "n_params": None,
    }
    from research.archs import build_model, count_params
    record["n_params"] = count_params(build_model(spec, ds.schema, ds.n_classes)) * n_members

    write_json(os.path.join(out_dir, "metrics.json"), pooled)
    write_json(os.path.join(out_dir, "record.json"), {k: v for k, v in record.items() if k != "per_subject"})
    write_json(os.path.join(out_dir, "leakage_audit.json"), fold_audits)
    log(f"    -> ENSEMBLE ({n_members}x) dev-LOSO acc {fold_agg['accuracy']['mean']:.4f} "
        f"f1 {fold_agg['f1_macro']['mean']:.4f}  vs mean single-model acc "
        f"{gap['mean_member_solo_test_accuracy']:.4f}  gain {gap['ensemble_vs_solo_accuracy_gain']:+.4f}  "
        f"gap(F1) {gap['train_minus_heldout']*100:+.1f}pts  ({record['runtime_seconds']:.0f}s)")
    return record, preds_df, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "final"], default="dev")
    ap.add_argument("--n-members", type=int, default=N_MEMBERS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lockbox", type=int, default=8)
    ap.add_argument("--tag", default="ensemble-campaign")
    ap.add_argument("--quick", action="store_true", help="4 dev subjects, smoke test only")
    args = ap.parse_args()

    run = RunContext(repo_root=ROOT, tag=args.tag)
    try:
        run.stage("setup")
        BASELINE_DATA["binary"] = True
        camp = Campaign(run, ROOT, device=args.device, seed=args.seed, n_lockbox=args.n_lockbox,
                        quick=args.quick)
        base_ds = camp.cache.get(BASELINE_DATA)
        audit = leakage.audit_dataset(base_ds, probe_subjects=base_ds.unique_subjects[:4], strict=True)
        run.log("dataset leakage audit: " + ("PASSED" if audit["passed"] else "FAILED"))
        if not audit["passed"]:
            raise leakage.LeakageError("dataset-level leakage audit failed")

        if args.stage == "dev":
            run.stage("dev_screen")
            run.log(f"=== ensemble dev-LOSO screen: {args.n_members} members, "
                    f"served architecture, 18 development folds ===")
            folds = list(loso_folds(base_ds, camp.dev_subjects, val_fraction=camp.val_fraction,
                                    seed=camp.seed))
            rec, preds, _ = run_ensemble_experiment(
                base_ds, BASELINE_SPEC, folds, "ENSEMBLE_dev", run.path("ensemble", "dev"),
                base_seed=args.seed, n_members=args.n_members, device=args.device, log=run.log,
            )
            write_json(run.path("ensemble_dev_summary.json"), {
                k: v for k, v in rec.items() if k != "per_subject"
            })
            run.complete({"stage": "dev", "acc": rec["fold_aggregate"]["accuracy"]["mean"],
                         "f1": rec["fold_aggregate"]["f1_macro"]["mean"]})

        else:  # final
            run.stage("final_ensemble")
            run.log(f"=== ensemble FINAL: {args.n_members} members, 26-fold LOSO + lockbox ===")
            loso_folds_list = list(loso_folds(base_ds, camp.all_subjects, val_fraction=camp.val_fraction,
                                              seed=camp.seed))
            loso_rec, loso_preds, _ = run_ensemble_experiment(
                base_ds, BASELINE_SPEC, loso_folds_list, "ENSEMBLE_FINAL_loso26",
                run.path("ensemble", "final_loso26"), base_seed=args.seed,
                n_members=args.n_members, device=args.device, log=run.log,
            )
            adjusted = []
            for f in loso_folds(base_ds, camp.lockbox_subjects, val_fraction=camp.val_fraction, seed=camp.seed):
                held = f["held_out_subject"]
                g = holdout_fold(base_ds, camp.dev_subjects, [held],
                                 val_fraction=camp.val_fraction, seed=camp.seed + int(held))
                g["fold"] = f["fold"]
                g["held_out_subject"] = held
                adjusted.append(g)
            lock_rec, lock_preds, _ = run_ensemble_experiment(
                base_ds, BASELINE_SPEC, adjusted, "ENSEMBLE_FINAL_lockbox",
                run.path("ensemble", "final_lockbox"), base_seed=args.seed,
                n_members=args.n_members, device=args.device, log=run.log,
            )
            loso_preds.to_csv(run.path("predictions", "predictions.csv"), index=False)
            write_json(run.path("metrics", "metrics.json"), loso_rec["pooled"])
            summary = {
                "n_members": args.n_members, "spec": BASELINE_SPEC,
                "loso26": loso_rec["fold_aggregate"], "loso26_pooled": loso_rec["pooled"],
                "loso26_gap": loso_rec["generalization_gap"],
                "lockbox": lock_rec["fold_aggregate"], "lockbox_pooled": lock_rec["pooled"],
                "lockbox_gap": lock_rec["generalization_gap"],
            }
            write_json(run.path("run_summary.json"), summary)
            run.log(f"\nENSEMBLE FINAL  loso26 acc={loso_rec['fold_aggregate']['accuracy']['mean']:.4f}  "
                    f"lockbox acc={lock_rec['fold_aggregate']['accuracy']['mean']:.4f}")
            run.complete({"loso26_acc": loso_rec["fold_aggregate"]["accuracy"]["mean"],
                         "lockbox_acc": lock_rec["fold_aggregate"]["accuracy"]["mean"]})

        print(f"\nRun complete: {run.dir}")
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
