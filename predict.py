#!/usr/bin/env python
"""
predict.py: one entry point for predicting from ANY supported dataset file.

    python predict.py --input path/to/file.csv
    python predict.py --input eeg.mat --input nirs.mat --dataset stroop
    python predict.py --input data.csv --version 3class --json out.json

Reads the file, auto-detects its structure, maps it onto the canonical schema,
and runs the existing frozen model. Prints the prediction summary AND the schema
mapping report, so it is always visible how the input was interpreted.

Exit codes:
    0  prediction produced
    2  refused - the input could not be mapped onto the model with enough
       confidence to be trustworthy (the reason is printed)
    1  an error occurred
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from ingestion.canonical import IngestionError, load_dataset_profiles   # noqa: E402
from ingestion.pipeline import BUNDLES, UniversalPredictor              # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Predict cognitive load from any supported EEG+fNIRS dataset file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported formats: .csv .tsv .txt .mat .npz .npy .edf .bdf\n"
            "Raw recordings usually need one file per modality - pass --input twice.\n"
            "See docs/adding_a_new_dataset.md to onboard a new dataset."
        ),
    )
    # Not argparse-`required`, so `--list-datasets` works on its own; the
    # requirement is enforced in main() instead.
    p.add_argument("--input", "-i", action="append", metavar="PATH",
                   help="Input file. Repeat for multi-modality recordings.")
    p.add_argument("--format", default="auto",
                   help="Reserved; format is detected from the file (default: auto).")
    p.add_argument("--dataset", default=None,
                   help="Force a dataset profile instead of auto-detecting.")
    p.add_argument("--version", default="binary", choices=sorted(BUNDLES),
                   help="Which serving bundle to predict with (default: binary).")
    p.add_argument("--subject-id", default=None,
                   help="Override the subject id used for per-subject normalization.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap the number of windows extracted from a raw recording.")
    p.add_argument("--json", metavar="PATH", default=None,
                   help="Write the full result (including reports) to a JSON file.")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Print only the summary line.")
    p.add_argument("--list-datasets", action="store_true",
                   help="List the known dataset profiles and exit.")
    return p


def _print_report(res, quiet=False):
    rep = res.get("schema_mapping_report") or {}
    cov = rep.get("coverage") or {}

    if res["refused"]:
        print(f"\n  REFUSED  [{res['error_code']}]")
        print(f"  {res['message']}\n")
        detail = res.get("detail") or {}
        for mod in ("eeg", "fnirs"):
            d = detail.get(mod)
            if d:
                print(f"  {mod:>6} coverage {d['coverage']:.1%}  ({d['verdict']}) - "
                      f"{d['n_exact']} exact, {d['n_neighbour']} approximate, "
                      f"{d['n_missing']} missing")
        if detail.get("missing_required"):
            miss = detail["missing_required"]
            print(f"\n  Missing required fields ({detail.get('n_missing', len(miss))}):")
            for m in miss[:15]:
                print(f"    - {m}")
            if len(miss) > 15:
                print(f"    ... and {detail.get('n_missing', len(miss)) - 15} more")
        if detail.get("remedy"):
            print(f"\n  {detail['remedy']}")
        return

    print(f"\n  {res['n_rows']} window(s) predicted   [verdict: {res['verdict']}]")
    print(f"  distribution: {res['distribution']}")
    if res.get("accuracy_vs_uploaded_labels") is not None:
        print(f"  accuracy vs supplied labels: "
              f"{res['accuracy_vs_uploaded_labels']:.1%}")
    if cov:
        print("  coverage: " + ", ".join(f"{k}={v:.1%}" for k, v in cov.items()))
    for pr in res.get("projection_reports", []):
        print(f"    {pr['modality']:>6}: {pr['coverage']:.1%} ({pr['verdict']}) - "
              f"{pr['n_exact']} exact, {pr['n_neighbour']} approximate, "
              f"{pr['n_missing']} missing")

    if not quiet:
        src = res.get("source") or {}
        print(f"\n  source: format={src.get('format')} "
              f"dataset={src.get('detected_dataset')} layout={src.get('layout')}")
        print(f"  normalization: {res.get('normalization_mode')}")
        unmatched = rep.get("unmatched_input") or []
        if unmatched:
            print(f"  ignored {len(unmatched)} unrecognized column(s): "
                  f"{unmatched[:6]}{' ...' if len(unmatched) > 6 else ''}")
        print("\n  first predictions:")
        for r in res["predictions"][:8]:
            flag = "  (abstain)" if r.get("abstain") else ""
            print(f"    row {r['row']:>3}  {r['predicted_label']:<6} "
                  f"conf={r['confidence']:.3f}{flag}")
        if len(res["predictions"]) > 8:
            print(f"    ... and {len(res['predictions']) - 8} more")

    for w in res.get("warnings", [])[:6]:
        print(f"\n  ! {w}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_datasets:
        profiles = load_dataset_profiles()
        print("Known dataset profiles (schema/datasets/):")
        for name, prof in sorted(profiles.items()):
            ref = "  [reference dataset]" if prof.get("is_reference_dataset") else ""
            print(f"  {name:<12} {prof.get('display_name', '')}{ref}")
        return 0

    if not args.input:
        print("error: --input is required (or use --list-datasets).", file=sys.stderr)
        return 1

    for path in args.input:
        if not os.path.exists(path):
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1

    try:
        predictor = UniversalPredictor(version=args.version)
        res = predictor.predict(
            [{"path": p} for p in args.input],
            dataset=args.dataset,
            subject_id=args.subject_id,
            max_windows=args.max_windows,
        )
    except IngestionError as exc:
        print(f"\n  CANNOT READ INPUT\n  {exc}", file=sys.stderr)
        if exc.detail:
            print(f"  detail: {json.dumps(exc.detail, indent=2, default=str)}",
                  file=sys.stderr)
        return 1
    except Exception as exc:                      # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_report(res, quiet=args.quiet)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\n  full result written to {args.json}")

    return 0 if res["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
