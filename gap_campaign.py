#!/usr/bin/env python
"""
gap_campaign.py - close the train/held-out gap to 5-8 points WITHOUT paying for
it in held-out accuracy.

Why another campaign
--------------------
`target_campaign.py` already chased this exact shape and logged 17 attempts to
outputs/runs/target_campaign_log.md. Reading that log, every arm that reached
the 5-8 point band got there the same way - by cutting the model's ability to
fit at all - and dragged held-out accuracy down with it:

    A0_baseline            held 0.6304  train 0.7804  gap 15.0
    C2_narrow_wd           held 0.6430  train 0.7836  gap 14.1   <- best held-out
    D2_narrow_patience6    held 0.6471  train 0.7628  gap 11.6
    C4_narrow_earlystop    held 0.6168  train 0.7133  gap  9.7
    C14_bs32               held 0.5994  train 0.6876  gap  8.8
    C7_aggressive_earlystop held 0.5543 train 0.6288  gap  7.5   <- in band, acc gone

Every one of those levers (capacity, weight decay, early-stop patience, epoch
cap, batch size) is a CAPACITY or BUDGET lever: it removes the model's ability
to fit anything, so train and held-out fall together and the gap closes from
both ends. That is why the frontier looks like a straight trade.

The lever that is missing from all 58 recorded configurations is one that
attacks MEMORISATION specifically while leaving discriminative capacity intact.
Mixup (Zhang et al., 2018) is the canonical one: the network trains on convex
combinations of pairs of windows with the matching combination of their losses,
so it cannot memorise a hard label for any individual training row, but nothing
about its ability to separate Low from High is taken away. The prediction this
campaign tests is therefore falsifiable and specific:

    mixup should move train accuracy DOWN by ~8-10 points while held-out
    accuracy stays inside the baseline's noise floor.

If that prediction is wrong, the honest conclusion is that a 5-8 point gap is
not reachable at this held-out accuracy on this dataset, and the log will say so.

Protocol
--------
Identical to target_campaign.py, deliberately: the same 18-fold development
LOSO, the same seed, the same dev/lockbox split, the same per-fold leakage audit
that aborts rather than produce a flattered score. The lockbox is never touched
here. Every attempt is appended to the SAME log, win or lose, so the whole
history of this goal stays in one file.

    .venv/Scripts/python gap_campaign.py                 # full sweep
    .venv/Scripts/python gap_campaign.py --only G3 G5    # just some candidates
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch

torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))

from research import leakage
from research.campaign import BASELINE_DATA, BASELINE_SPEC, Campaign, stderr_of
from research.runctx import RunContext, write_json
from target_campaign import (
    LOG_MD,
    append_log,
    data_with,
    evaluate_dev,
    spec_with,
)

# The narrow + weight-decay family. C2_narrow_wd holds the highest dev-LOSO
# accuracy of any arm in the whole 58-config record (0.6430), so it is the right
# base to try to shrink the gap UNDER - starting from the best held-out number
# available leaves the most room to lose before falling out of band.
NARROW_WD = dict(proj_dim=32, enc_dim=32, fc_units=[32], weight_decay=0.02)

# Target shape. Held-out is allowed to sit at or above the baseline's own
# dev-LOSO accuracy minus one standard error; the gap must land in 5-8.
GAP_LO, GAP_HI = 5.0, 8.0


def build_candidates():
    """One axis, swept, on the two capacity families that matter.

    Mixup strength is the only thing that varies within a family, so the sweep
    reads as a curve rather than a bag of one-offs: if the mechanism works, gap
    should fall monotonically with alpha and held-out should stay flat until it
    does not. alpha ~ Beta(a, a): 0.2 mixes rarely and mildly, 1.0 draws the
    mixing weight uniformly and mixes hard.
    """
    S = lambda spec=None, data=None, note="": {
        "spec": spec or {},
        "data": data or {},
        "note": note,
    }
    return [
        ("G1_mixup02", S(
            dict(mixup_alpha=0.2),
            note="baseline capacity + mixup 0.2 [NEW AXIS: attacks memorisation, not capacity]")),
        ("G2_mixup05", S(
            dict(mixup_alpha=0.5),
            note="baseline capacity + mixup 0.5 [NEW AXIS]")),
        ("G3_mixup10", S(
            dict(mixup_alpha=1.0),
            note="baseline capacity + mixup 1.0 (uniform mixing) [NEW AXIS]")),
        ("G4_narrow_mixup05", S(
            dict(**NARROW_WD, mixup_alpha=0.5),
            note="narrow+wd (best held-out arm, C2) + mixup 0.5 [NEW AXIS on best base]")),
        ("G5_narrow_mixup10", S(
            dict(**NARROW_WD, mixup_alpha=1.0),
            note="narrow+wd (best held-out arm, C2) + mixup 1.0 [NEW AXIS on best base]")),
    ] + build_round2()


# Mixup strength carried into round 2. Set from the round-1 sweep: whichever
# alpha bought the most held-out accuracy is the one worth spending.
MIX = 0.5


def build_round2():
    """Round 2: spend the held-out headroom mixup buys on a shorter budget.

    Round 1 falsified the original prediction. Mixup did NOT suppress train
    fit - train accuracy went UP slightly - but it did lift held-out accuracy
    about a point (0.6304 -> 0.6389 at alpha 0.5). So its role changes: it is
    not the mechanism that closes the gap, it is a BUDGET. Two arms in the
    existing log missed the target only narrowly, and both missed in a
    direction a point of held-out would fix:

        C14_bs32   held 0.5994  train 0.6876  gap 8.8  (held 0.7 pts under floor)
        C4_narrow  held 0.6168  train 0.7133  gap 9.7  (held fine, train 2 pts high)

    Each arm below pairs mixup with one budget lever, so the only thing that
    varies against the round-1 controls is the budget. max_epochs is included
    as a continuous dial because early_stop_patience is an integer and 3 vs 4
    already brackets the target too coarsely (gap 7.5 vs 9.7).
    """
    S = lambda spec=None, data=None, note="": {
        "spec": spec or {},
        "data": data or {},
        "note": note,
    }
    return [
        ("H1_bs32_mixup", S(
            dict(batch_size=32, lr=2e-4, mixup_alpha=MIX),
            note=f"C14_bs32 budget + mixup {MIX} [C14 missed only on held-out, by 0.7 pts]")),
        ("H2_narrow_p4_mixup", S(
            dict(**NARROW_WD, early_stop_patience=4, mixup_alpha=MIX),
            note=f"C4 narrow+patience4 budget + mixup {MIX} [C4 missed only on train, by 2 pts]")),
        ("H3_epochs15_mixup", S(
            dict(max_epochs=15, mixup_alpha=MIX),
            note=f"baseline + hard 15-epoch cap + mixup {MIX} [continuous budget dial]")),
        ("H4_epochs12_mixup", S(
            dict(max_epochs=12, mixup_alpha=MIX),
            note=f"baseline + hard 12-epoch cap + mixup {MIX} [continuous budget dial]")),
        ("H5_p4_mixup", S(
            dict(early_stop_patience=4, mixup_alpha=MIX),
            note=f"baseline capacity + patience 4 + mixup {MIX} [best-held-out base + budget]")),
    ]


def verdict_for(heldout_acc, train_acc, gap_pts, base_acc, se):
    """In-band on the gap, and held-out not meaningfully below the baseline.

    The held-out test is one standard error of the baseline's own fold spread,
    which is the same acceptance bar the rest of the project uses. A config that
    lands in the gap band by giving up real accuracy is NOT a hit - that is the
    trade every previous arm already made.
    """
    gap_ok = GAP_LO <= gap_pts <= GAP_HI
    held_ok = heldout_acc >= base_acc - se
    if gap_ok and held_ok:
        return "TARGET HIT - gap in band, held-out within 1 SE of baseline"
    if gap_ok and not held_ok:
        return (f"gap in band but held-out fell {(base_acc - heldout_acc) * 100:.1f} pts "
                f"below baseline (> 1 SE = {se * 100:.1f} pts) - rejected")
    if held_ok and not gap_ok:
        return f"held-out held up but gap {gap_pts:.1f} pts is outside {GAP_LO}-{GAP_HI} - rejected"
    return "rejected - missed on both axes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lockbox", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tag", default="gap-campaign")
    ap.add_argument("--skip-reconfirm", action="store_true",
                    help="trust the logged baseline instead of re-running it")
    args = ap.parse_args()

    run = RunContext(repo_root=ROOT, tag=args.tag)
    winners = []
    try:
        run.stage("setup")
        BASELINE_DATA["binary"] = True
        camp = Campaign(run, ROOT, device=args.device, seed=args.seed,
                        n_lockbox=args.n_lockbox)
        base_ds = camp.cache.get(BASELINE_DATA)
        audit = leakage.audit_dataset(base_ds, probe_subjects=base_ds.unique_subjects[:4],
                                      strict=True)
        run.log("dataset leakage audit: " + ("PASSED" if audit["passed"] else "FAILED"))
        if not audit["passed"]:
            raise leakage.LeakageError("dataset-level leakage audit failed")

        # Control arm. It has reproduced at 0.6304 / 0.7804 three times in the
        # existing log, but it is re-run here anyway: the comparison that decides
        # every candidate has to come from this session's own environment.
        run.stage("baseline_reconfirm")
        base_rec = evaluate_dev(run, camp, "G0_baseline_control", BASELINE_SPEC,
                                BASELINE_DATA, "baseline control for the gap campaign")
        base_acc = base_rec["heldout_accuracy_mean"]
        base_f1 = base_rec["fold_aggregate"]["f1_macro"]["mean"]
        se = stderr_of(base_rec)
        run.log(f"control: held-out acc {base_acc:.4f}  train {base_rec['train_accuracy_mean']:.4f}  "
                f"gap {base_rec['gap_accuracy'] * 100:+.1f} pts  (1 SE = {se * 100:.2f} pts)")
        append_log("G0_baseline_control", "baseline control for the gap campaign",
                   base_acc, base_f1, base_rec["train_accuracy_mean"],
                   base_rec["gap_accuracy"], "control")

        candidates = build_candidates()
        if args.only:
            candidates = [(cid, cfg) for cid, cfg in candidates if cid in args.only]

        run.stage("candidates")
        records = {"G0_baseline_control": base_rec}
        for exp_id, cfg in candidates:
            run.log(f"\n=== {exp_id}: {cfg['note']} ===")
            spec = spec_with(BASELINE_SPEC, **cfg["spec"])
            dcfg = data_with(BASELINE_DATA, **cfg["data"])
            try:
                rec = evaluate_dev(run, camp, exp_id, spec, dcfg, cfg["note"])
            except Exception as exc:
                run.warn(f"{exp_id} failed: {type(exc).__name__}: {exc}")
                append_log(exp_id, cfg["note"], float("nan"), float("nan"),
                           float("nan"), float("nan"),
                           f"FAILED: {type(exc).__name__}: {exc}")
                continue
            records[exp_id] = rec
            held = rec["heldout_accuracy_mean"]
            tr = rec["train_accuracy_mean"]
            gap = rec["gap_accuracy"] * 100
            f1 = rec["fold_aggregate"]["f1_macro"]["mean"]
            verdict = verdict_for(held, tr, gap, base_acc, se)
            append_log(exp_id, cfg["note"], held, f1, tr, rec["gap_accuracy"], verdict)
            if verdict.startswith("TARGET HIT"):
                winners.append(exp_id)
            write_json(run.path("target_experiments", "gap_campaign_summary.json"), {
                k: {"note": v.get("note"), "spec": v["spec"],
                    "dev_loso_acc": v["fold_aggregate"]["accuracy"]["mean"],
                    "dev_loso_f1": v["fold_aggregate"]["f1_macro"]["mean"],
                    "train_acc": v.get("train_accuracy_mean"),
                    "gap_acc_pts": v.get("gap_accuracy", 0) * 100}
                for k, v in records.items()
            })

        run.log(f"\n=== sweep complete: {len(winners)} hit the target: {winners} ===")
        run.complete({"n_candidates": len(candidates), "winners": winners})
        print(f"\nRun complete: {run.dir}")
        print(f"Log: {LOG_MD}")
        print(f"WINNERS: {winners}" if winners else "No candidate hit the target shape.")
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
