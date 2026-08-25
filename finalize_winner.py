#!/usr/bin/env python
"""
finalize_winner.py — promote ONE dev-LOSO winner from target_campaign.py to
the project's headline protocol: 26-fold LOSO over all subjects + the
8-subject lockbox, evaluated once. Also re-runs the SAME two protocols for
the untouched baseline in this same run/environment, so the comparison is
paired and not contaminated by cross-run/cross-torch-version noise.

Writes a run directory in the standard outputs/runs/ layout (predictions.csv,
metrics.json, loso_metrics.json, ...) so research.verify_run can independently
recompute the metrics from predictions alone (integrity constraint 4).

    .venv/Scripts/python finalize_winner.py --candidate C2_narrow_wd
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))

from research import leakage
from research import metrics as M
from research.campaign import BASELINE_DATA, BASELINE_SPEC, Campaign, paired_vs
from research.runctx import RunContext, write_json
from research.verify_run import verify
from target_campaign import build_candidates, data_with, spec_with


def run_final(camp, run, spec, dcfg, exp_id_prefix):
    loso_rec, loso_preds, *_ = camp.evaluate_full_loso(spec, dcfg, exp_id=f"{exp_id_prefix}_loso26")
    lock_rec, lock_preds, *_ = camp.evaluate_lockbox(spec, dcfg, exp_id=f"{exp_id_prefix}_lockbox")
    return loso_rec, loso_preds, lock_rec, lock_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="candidate id from target_campaign.build_candidates()")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lockbox", type=int, default=8)
    ap.add_argument("--tag", default="target-campaign-final")
    args = ap.parse_args()

    candidates = dict(build_candidates())
    if args.candidate not in candidates:
        raise SystemExit(f"unknown candidate {args.candidate!r}; choices: {list(candidates)}")
    cfg = candidates[args.candidate]
    spec = spec_with(BASELINE_SPEC, **cfg["spec"])
    dcfg = data_with(BASELINE_DATA, **cfg["data"])

    run = RunContext(repo_root=ROOT, tag=args.tag)
    try:
        run.stage("setup")
        BASELINE_DATA["binary"] = True
        camp = Campaign(run, ROOT, device=args.device, seed=args.seed, n_lockbox=args.n_lockbox)
        base_ds = camp.cache.get(BASELINE_DATA)
        audit = leakage.audit_dataset(base_ds, probe_subjects=base_ds.unique_subjects[:4], strict=True)
        run.log("dataset leakage audit: " + ("PASSED" if audit["passed"] else "FAILED"))
        if not audit["passed"]:
            raise leakage.LeakageError("dataset-level leakage audit failed")

        run.write_metadata(
            config={"candidate": args.candidate, "spec": spec, "data": dcfg,
                    "note": cfg["note"], "baseline_spec": BASELINE_SPEC,
                    "baseline_data": BASELINE_DATA, "seed": args.seed},
            dataset_info=base_ds.info({
                "development_subjects": camp.dev_subjects.tolist(),
                "lockbox_subjects": camp.lockbox_subjects.tolist(),
            }),
        )

        run.stage("final_baseline")
        run.log("=== FINAL: baseline, 26-fold LOSO + lockbox (fresh, this run) ===")
        base_loso, base_loso_preds, base_lock, base_lock_preds = run_final(
            camp, run, BASELINE_SPEC, BASELINE_DATA, "BASELINE")

        run.stage("final_candidate")
        run.log(f"=== FINAL: {args.candidate}, 26-fold LOSO + lockbox ===")
        cand_loso, cand_loso_preds, cand_lock, cand_lock_preds = run_final(
            camp, run, spec, dcfg, "CANDIDATE")

        run.stage("artifacts")
        cand_loso_preds.to_csv(run.path("predictions", "predictions.csv"), index=False)
        pooled = cand_loso["pooled"]
        write_json(run.path("metrics", "metrics.json"), pooled)
        write_json(run.path("metrics", "loso_metrics.json"), {
            "protocol": f"{cand_loso['n_folds']}-fold leave-one-subject-out",
            "fold_aggregate": cand_loso["fold_aggregate"],
            "generalization_gap": cand_loso["generalization_gap"],
            "pooled": pooled,
            "lockbox": {
                "subjects": camp.lockbox_subjects.tolist(),
                "fold_aggregate": cand_lock["fold_aggregate"],
                "pooled": cand_lock["pooled"],
            },
            "baseline_comparison": {
                "loso26": base_loso["fold_aggregate"],
                "lockbox": base_lock["fold_aggregate"],
                "paired_loso26": paired_vs(cand_loso, base_loso),
                "paired_lockbox": paired_vs(cand_lock, base_lock),
            },
        })
        pd.DataFrame(cand_loso["per_subject"]).to_csv(
            run.path("metrics", "per_subject_metrics.csv"), index=False)

        summary = {
            "run_id": run.run_id,
            "candidate": args.candidate,
            "note": cfg["note"],
            "spec": spec,
            "data": dcfg,
            "n_params": cand_loso["n_params"],
            "results": {
                "candidate": {
                    "loso26": cand_loso["fold_aggregate"],
                    "loso26_pooled": cand_loso["pooled"],
                    "lockbox": cand_lock["fold_aggregate"],
                    "lockbox_pooled": cand_lock["pooled"],
                    "generalization_gap_f1": cand_loso["generalization_gap"],
                },
                "baseline": {
                    "loso26": base_loso["fold_aggregate"],
                    "loso26_pooled": base_loso["pooled"],
                    "lockbox": base_lock["fold_aggregate"],
                    "lockbox_pooled": base_lock["pooled"],
                    "generalization_gap_f1": base_loso["generalization_gap"],
                },
                "paired_tests": {
                    "loso26_candidate_vs_baseline": paired_vs(cand_loso, base_loso),
                    "lockbox_candidate_vs_baseline": paired_vs(cand_lock, base_lock),
                },
            },
        }
        write_json(run.path("run_summary.json"), summary)

        run.log(f"\nCANDIDATE  loso26 acc={cand_loso['fold_aggregate']['accuracy']['mean']:.4f}  "
                f"f1={cand_loso['fold_aggregate']['f1_macro']['mean']:.4f}  "
                f"lockbox acc={cand_lock['fold_aggregate']['accuracy']['mean']:.4f}  "
                f"gap(train-heldout f1)={cand_loso['generalization_gap']['train_minus_heldout']:.4f}")
        run.log(f"BASELINE   loso26 acc={base_loso['fold_aggregate']['accuracy']['mean']:.4f}  "
                f"f1={base_loso['fold_aggregate']['f1_macro']['mean']:.4f}  "
                f"lockbox acc={base_lock['fold_aggregate']['accuracy']['mean']:.4f}  "
                f"gap(train-heldout f1)={base_loso['generalization_gap']['train_minus_heldout']:.4f}")

        run.complete({"candidate": args.candidate,
                      "loso26_acc": cand_loso["fold_aggregate"]["accuracy"]["mean"],
                      "lockbox_acc": cand_lock["fold_aggregate"]["accuracy"]["mean"]})

        vres = verify(run.dir)
        print(f"\nverify_run: {vres['status']}")
        print(f"\nRun complete: {run.dir}")
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
