"""
verify_run.py — independently recompute a run's metrics from its predictions.

The project's rule is that every number shown anywhere traces to an artifact.
This tool enforces the stronger version of that rule for the run archive: it
reads ONLY predictions.csv, recomputes the whole metric record from scratch, and
diffs it against the metrics.json the run wrote. Any disagreement is a bug.

    .venv/bin/python -m research.verify_run outputs/runs/<run_id>
    .venv/bin/python -m research.verify_run --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from .metrics import aggregate_folds, compute_metrics, per_subject_metrics

TOL = 1e-9


def verify(run_dir, verbose=True):
    preds_path = os.path.join(run_dir, "predictions", "predictions.csv")
    metrics_path = os.path.join(run_dir, "metrics", "metrics.json")
    if not (os.path.isfile(preds_path) and os.path.isfile(metrics_path)):
        return {"run": os.path.basename(run_dir), "status": "skipped",
                "reason": "no predictions.csv / metrics.json"}

    df = pd.read_csv(preds_path)
    stored = json.load(open(metrics_path, encoding="utf-8"))
    class_names = stored["class_names"]
    n_classes = stored["n_classes"]
    pcols = [f"prob_{c}" for c in class_names]

    recomputed = compute_metrics(
        df["y_true"].to_numpy(int),
        df["y_pred"].to_numpy(int),
        df[pcols].to_numpy(float),
        n_classes,
        class_names,
    )

    diffs = []
    for key, want in stored.items():
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            got = recomputed.get(key)
            if got is None:
                diffs.append((key, want, None))
            elif abs(float(got) - float(want)) > TOL:
                diffs.append((key, want, got))
    if stored.get("confusion_matrix") != recomputed.get("confusion_matrix"):
        diffs.append(("confusion_matrix", stored.get("confusion_matrix"),
                      recomputed.get("confusion_matrix")))
    for a, b in zip(stored.get("per_class", []), recomputed.get("per_class", [])):
        for k in ("precision", "recall", "f1", "support", "tp", "fp", "fn", "tn"):
            if a.get(k) is None or b.get(k) is None:
                continue
            if abs(float(a[k]) - float(b[k])) > TOL:
                diffs.append((f"per_class.{a['class_name']}.{k}", a[k], b[k]))

    # per-subject + fold aggregate, when the run recorded them
    loso_path = os.path.join(run_dir, "metrics", "loso_metrics.json")
    agg_diffs = []
    if os.path.isfile(loso_path):
        loso = json.load(open(loso_path, encoding="utf-8"))
        stored_agg = loso.get("fold_aggregate") or {}
        rows = per_subject_metrics(df, n_classes, class_names, pcols)
        got_agg = aggregate_folds(rows)
        for key, want in stored_agg.items():
            if not isinstance(want, dict):
                continue
            got = got_agg.get(key) or {}
            for stat in ("mean", "std"):
                a, b = want.get(stat), got.get(stat)
                if a is None or b is None:
                    continue
                if abs(float(a) - float(b)) > 1e-9:
                    agg_diffs.append((f"fold_aggregate.{key}.{stat}", a, b))

    ok = not diffs and not agg_diffs
    result = {
        "run": os.path.basename(run_dir),
        "status": "verified" if ok else "MISMATCH",
        "n_predictions": int(len(df)),
        "n_metrics_checked": sum(
            1 for k, v in stored.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ),
        "mismatches": [{"metric": k, "stored": a, "recomputed": b}
                       for k, a, b in diffs + agg_diffs],
    }
    if verbose:
        mark = "OK  " if ok else "FAIL"
        print(f"  {mark} {result['run']}: {result['n_predictions']} predictions, "
              f"{result['n_metrics_checked']} scalar metrics recomputed"
              + ("" if ok else f", {len(result['mismatches'])} MISMATCHES"))
        for mm in result["mismatches"][:10]:
            print(f"        {mm['metric']}: stored={mm['stored']} recomputed={mm['recomputed']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--all", action="store_true", help="verify every run in the archive")
    args = ap.parse_args()

    if args.all:
        root = os.path.join("outputs", "runs")
        dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
    elif args.run_dir:
        dirs = [args.run_dir]
    else:
        ap.error("pass a run directory or --all")

    print(f"Recomputing metrics from predictions for {len(dirs)} run(s):")
    results = [verify(d) for d in dirs]
    bad = [r for r in results if r["status"] == "MISMATCH"]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"\n{len(results) - len(bad) - len(skipped)} verified, "
          f"{len(bad)} mismatched, {len(skipped)} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
