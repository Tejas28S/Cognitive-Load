#!/usr/bin/env python
"""
target_campaign.py — target-driven iteration on top of the binary-model
improvement campaign, per TRAINING_ITERATION_INSTRUCTIONS.md.

Goal (a SHAPE, not a magic number):
    held-out LOSO accuracy  ~65%  (63-67% acceptable)
    train accuracy          ~70-72%
    train - heldout gap     ~5-8 points

The baseline (A0_baseline, RESULTS_BINARY_IMPROVEMENT.md) is already at 63.09%
LOSO accuracy -- inside the target band -- but its train accuracy is ~80%,
so the gap (~17.5 points) is too wide. This script searches for a config that
keeps LOSO accuracy in-band while shrinking the gap, using only levers NOT
already tried (unchanged) in the 42-experiment campaign.

Every candidate runs the SAME 18-fold development LOSO used by run_campaign.py
(cheap signal). It never touches the lockbox. Every fold runs the full leakage
audit (research/leakage.py) before training -- unmodified. Every attempt is
appended to outputs/runs/target_campaign_log.md as it finishes, win or lose.

    .venv/Scripts/python target_campaign.py                  # full sweep
    .venv/Scripts/python target_campaign.py --only C2 C5      # just some candidates
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))

from research import leakage
from research.campaign import BASELINE_DATA, BASELINE_SPEC, Campaign, score_of, stderr_of
from research.datamod import holdout_fold, loso_folds, scale_fold
from research.experiment import run_experiment
from research.runctx import RunContext, write_json

LOG_MD = os.path.join(ROOT, "outputs", "runs", "target_campaign_log.md")
LOG_HEADER = (
    "| run_id | change made | dev-LOSO acc | dev-LOSO F1 | train acc "
    "| train-heldout gap (acc) | verdict |\n"
    "|---|---|---:|---:|---:|---:|---|\n"
)


def spec_with(base, **kw):
    s = copy.deepcopy(base)
    s.update(kw)
    return s


def data_with(base, **kw):
    d = copy.deepcopy(base)
    d.update(kw)
    return d


# ── candidates: every one is a genuinely new combo, not a repeat of the
# 42 already-tried arms in RESULTS_BINARY_IMPROVEMENT.md. Ordered per the
# method list in TRAINING_ITERATION_INSTRUCTIONS.md: overfitting-side joint
# grids first, then data-side levers, then a final hyperparameter pass. ──
def build_candidates():
    S = lambda spec=None, data=None, note="": {"spec": spec or {}, "data": data or {}, "note": note}
    return [
        ("C2_narrow_wd", S(dict(proj_dim=32, enc_dim=32, fc_units=[32], weight_decay=0.02),
                           note="narrow capacity (S5_proj32_enc32) + weight_decay 0.02 [joint, new combo]")),
        ("C3_narrow_lessdrop", S(dict(proj_dim=32, enc_dim=32, fc_units=[32],
                                       encoder_dropout=0.4, fc_dropout=0.5),
                                 note="narrow capacity + LOWER dropout (less capacity needs less dropout)")),
        ("C4_narrow_earlystop", S(dict(proj_dim=32, enc_dim=32, fc_units=[32],
                                        early_stop_patience=4),
                                  note="narrow capacity + aggressive early stop (patience 4)")),
        ("C5_medium_capacity", S(dict(proj_dim=64, enc_dim=32, fc_units=[32, 16]),
                                 note="untried capacity point between baseline (96/48) and narrow (32/32)")),
        ("C6_medium_wd", S(dict(proj_dim=64, enc_dim=32, fc_units=[32, 16], weight_decay=0.02),
                           note="medium capacity + weight_decay 0.02")),
        ("C7_aggressive_earlystop", S(dict(early_stop_patience=3, lr_patience=2),
                                      note="baseline capacity, much shorter patience (untried direction; "
                                           "42-list only tried LONGER budgets)")),
        ("C8_epoch_cap", S(dict(max_epochs=20, early_stop_patience=6),
                           note="hard epoch budget cap (forces less training; opposite of S2 longer-budget arms)")),
        ("C9_blocks1_wd", S(dict(n_blocks=1, weight_decay=0.02),
                            note="1 residual block (S5_blocks1) + weight_decay 0.02 [joint, new combo]")),
        ("C10_dense_narrow", S(dict(proj_dim=32, enc_dim=32, fc_units=[32]),
                               data=dict(dataset="sliding_10s_1s"),
                               note="dense windows (S10_stride1s) + narrow capacity [joint, new combo]")),
        ("C11_dense_wd", S(dict(weight_decay=0.02), data=dict(dataset="sliding_10s_1s"),
                           note="dense windows (S10_stride1s) + weight_decay 0.02 [joint, new combo]")),
        ("C12_joint_mild", S(dict(weight_decay=0.02, encoder_dropout=0.55, fc_dropout=0.65),
                             note="mild joint bump: wd + encoder_dropout + fc_dropout together "
                                  "(42-list only varied ONE axis at a time)")),
        ("C13_modality_wd", S(dict(modality_dropout_eeg=0.05, modality_dropout_fnirs=0.05,
                                    weight_decay=0.015),
                              note="mild modality dropout (untried 0.05/0.05 level) + weight_decay 0.015")),
        ("C14_bs32", S(dict(batch_size=32, lr=2e-4),
                       note="batch 32 / lr 2e-4 (untried batch size; 42-list only tried 64/128)")),
        # Follow-ups: C4 (narrow capacity + patience 4) landed closest to the
        # target shape of all 15 first-round candidates (train acc 71.3%, gap
        # 9.7pts) but held-out fell just under 63%. C2 (narrow, default
        # patience 7) had held-out 64.3% but gap 14.1pts. Interpolating the
        # early-stop patience between them is a direct, motivated probe of
        # the boundary these two results bracket -- not a blind repeat.
        ("D1_narrow_patience5", S(dict(proj_dim=32, enc_dim=32, fc_units=[32],
                                        early_stop_patience=5),
                                  note="narrow capacity + patience 5 (interpolates C4 patience4 "
                                       "and C2 patience7, which bracketed the target)")),
        ("D2_narrow_patience6", S(dict(proj_dim=32, enc_dim=32, fc_units=[32],
                                        early_stop_patience=6),
                                  note="narrow capacity + patience 6 (interpolates C4 patience4 "
                                       "and C2 patience7, which bracketed the target)")),
    ]


def train_accuracy_gap(hists, fold_records, ds_class_names):
    """Mean train/val accuracy at each fold's selected (best) epoch.

    `hists` entries carry the FULL per-epoch history (train_accuracy computed
    train.eval()-mode, i.e. without dropout/noise, so it is a genuine fit
    measure) plus `best_epoch` (1-indexed). This is the literal "accuracy"
    the target is specified in, as opposed to the macro-F1 gap the framework
    natively tracks in generalization_gap.
    """
    tr_acc, va_acc = [], []
    for h in hists:
        be = h["best_epoch"]
        tr_acc.append(h["train_accuracy"][be - 1])
        va_acc.append(h["val_accuracy"][be - 1])
    return float(np.mean(tr_acc)), float(np.mean(va_acc))


def append_log(run_id, change, dev_acc, dev_f1, train_acc, gap_acc, verdict):
    os.makedirs(os.path.dirname(LOG_MD), exist_ok=True)
    new = not os.path.exists(LOG_MD)
    with open(LOG_MD, "a", encoding="utf-8") as fh:
        if new:
            fh.write("# Target campaign log\n\n")
            fh.write("Per TRAINING_ITERATION_INSTRUCTIONS.md. One row per attempt, "
                     "win or lose. dev-LOSO = 18-fold development LOSO (selection "
                     "signal, never the lockbox).\n\n")
            fh.write(LOG_HEADER)
        fh.write(
            f"| {run_id} | {change} | {dev_acc:.4f} | {dev_f1:.4f} | "
            f"{train_acc:.4f} | {gap_acc * 100:+.1f} pts | {verdict} |\n"
        )


def evaluate_dev(run, camp, exp_id, spec, dcfg, note):
    """One arm on the 18-fold development LOSO, with train-accuracy gap."""
    ds = camp.cache.get(dcfg)
    folds = list(loso_folds(ds, camp.dev_subjects, val_fraction=camp.val_fraction,
                            seed=camp.seed))
    out_dir = run.path("target_experiments", exp_id)
    t0 = time.time()
    record, preds, hists, curves, _ = run_experiment(
        ds, spec, folds, exp_id, out_dir, seed=camp.seed, device=camp.device,
        log=run.log, use_domains=bool(spec.get("use_dann")),
    )
    train_acc, val_acc = train_accuracy_gap(hists, record["per_fold"], ds.class_names)
    heldout_acc = record["fold_aggregate"]["accuracy"]["mean"]
    gap_acc = train_acc - heldout_acc
    record.update({"note": note, "data": dcfg, "train_accuracy_mean": train_acc,
                   "val_accuracy_mean": val_acc, "heldout_accuracy_mean": heldout_acc,
                   "gap_accuracy": gap_acc})
    write_json(os.path.join(out_dir, "record.json"),
               {k: v for k, v in record.items() if k != "per_subject"})
    run.log(f"    dev-LOSO acc {heldout_acc:.4f}  f1 {record['fold_aggregate']['f1_macro']['mean']:.4f}  "
            f"train_acc {train_acc:.4f}  gap(acc) {gap_acc * 100:+.1f}pts  ({time.time() - t0:.0f}s)")
    return record


def in_band(x, lo, hi):
    return lo <= x <= hi


def shape_verdict(heldout_acc, train_acc, gap_pts):
    held_ok = in_band(heldout_acc, 0.63, 0.67)
    train_ok = in_band(train_acc, 0.68, 0.74)  # small margin around 70-72
    gap_ok = in_band(gap_pts, 4.0, 9.0)         # small margin around 5-8
    if held_ok and train_ok and gap_ok:
        return "target hit"
    return "rejected - no improvement"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lockbox", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run only these candidate ids")
    ap.add_argument("--tag", default="target-campaign")
    args = ap.parse_args()

    run = RunContext(repo_root=ROOT, tag=args.tag)
    winners = []
    try:
        run.stage("setup")
        BASELINE_DATA["binary"] = True
        camp = Campaign(run, ROOT, device=args.device, seed=args.seed, n_lockbox=args.n_lockbox)
        base_ds = camp.cache.get(BASELINE_DATA)
        audit = leakage.audit_dataset(base_ds, probe_subjects=base_ds.unique_subjects[:4], strict=True)
        run.log("dataset leakage audit: " + ("PASSED" if audit["passed"] else "FAILED"))
        if not audit["passed"]:
            raise leakage.LeakageError("dataset-level leakage audit failed")

        run.stage("baseline_reconfirm")
        run.log("=== A0_baseline reconfirm (must match RESULTS_BINARY_IMPROVEMENT.md: "
                "dev-LOSO macro F1 0.6451, acc 0.6537) ===")
        base_rec = evaluate_dev(run, camp, "A0_baseline_reconfirm", BASELINE_SPEC, BASELINE_DATA,
                                "baseline reconfirm")
        base_f1 = base_rec["fold_aggregate"]["f1_macro"]["mean"]
        base_acc = base_rec["fold_aggregate"]["accuracy"]["mean"]
        reproduced = abs(base_f1 - 0.6451) < 0.01
        run.log(f"reconfirm: f1={base_f1:.4f} (expected 0.6451) acc={base_acc:.4f} "
                f"(expected 0.6537) -- {'REPRODUCED' if reproduced else 'DID NOT REPRODUCE'}")
        append_log("A0_baseline_reconfirm", "baseline (config.py + served architecture), reconfirmation",
                   base_acc, base_f1, base_rec["train_accuracy_mean"], base_rec["gap_accuracy"],
                   "reproduced" if reproduced else "MISMATCH vs documented baseline")
        if not reproduced:
            run.warn("Baseline did not reproduce the documented dev-LOSO score within tolerance. "
                     "Proceeding, but treat all deltas below with extra caution.")

        base_incumbent_f1 = base_f1
        se = stderr_of(base_rec)

        candidates = build_candidates()
        if args.only:
            candidates = [(cid, cfg) for cid, cfg in candidates if cid in args.only]

        run.stage("candidates")
        all_records = {"A0_baseline_reconfirm": base_rec}
        for exp_id, cfg in candidates:
            run.log(f"\n=== {exp_id}: {cfg['note']} ===")
            spec = spec_with(BASELINE_SPEC, **cfg["spec"])
            dcfg = data_with(BASELINE_DATA, **cfg["data"])
            try:
                rec = evaluate_dev(run, camp, exp_id, spec, dcfg, cfg["note"])
            except Exception as exc:
                run.warn(f"{exp_id} failed: {type(exc).__name__}: {exc}")
                append_log(exp_id, cfg["note"], float("nan"), float("nan"),
                          float("nan"), float("nan"), f"FAILED: {type(exc).__name__}: {exc}")
                continue
            all_records[exp_id] = rec
            heldout_acc = rec["heldout_accuracy_mean"]
            train_acc = rec["train_accuracy_mean"]
            gap_pts = rec["gap_accuracy"] * 100
            f1 = rec["fold_aggregate"]["f1_macro"]["mean"]
            within_noise = (base_incumbent_f1 - f1) <= se
            verdict = shape_verdict(heldout_acc, train_acc, gap_pts)
            if verdict != "target hit" and within_noise and f1 >= base_incumbent_f1 - se:
                verdict += " (within noise floor of baseline)"
            append_log(exp_id, cfg["note"], heldout_acc, f1, train_acc, rec["gap_accuracy"], verdict)
            if verdict.startswith("target hit"):
                winners.append(exp_id)
            write_json(run.path("target_experiments", "all_records_summary.json"), {
                k: {"note": v.get("note"), "spec": v["spec"], "data": v["data"],
                    "dev_loso_acc": v["fold_aggregate"]["accuracy"]["mean"],
                    "dev_loso_f1": v["fold_aggregate"]["f1_macro"]["mean"],
                    "train_acc": v.get("train_accuracy_mean"),
                    "gap_acc_pts": v.get("gap_accuracy", 0) * 100}
                for k, v in all_records.items()
            })

        run.log(f"\n=== sweep complete: {len(winners)} candidate(s) hit the target shape "
                f"on dev-LOSO: {winners} ===")
        run.complete({"n_candidates": len(candidates), "winners": winners})
        print(f"\nRun complete: {run.dir}")
        print(f"Log: {LOG_MD}")
        if winners:
            print(f"WINNERS (promote to full LOSO + lockbox): {winners}")
        else:
            print("No candidate hit the target shape on dev-LOSO.")
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
