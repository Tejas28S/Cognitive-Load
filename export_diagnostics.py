"""
export_diagnostics.py — calibration & reliability diagnostics for the LOCKED
served ensemble, computed on the held-out demo windows (fixed-split).

This adds NO change to the served model. It runs the frozen Hybrid + RF
ensemble (via the real backend ModelService — the same call path the API uses)
over the 150 committed demo windows and writes an aggregate diagnostics
artifact the frontend can plot:

    serving/frontend_data/diagnostics.json
      - reliability     : confidence-bin table + scalar ECE (calibration)
      - coverage_curve  : accuracy vs coverage (validates the abstain thresholds)
      - per_subject     : accuracy + mean confidence for each demo subject
      - demo_confusion  : aggregate 3x3 confusion matrix on the demo windows
      - latency         : mean / p95 single-prediction latency (ms)

Every number here traces to this committed artifact + the frozen weights, in
the same spirit as the other frontend_data/*.json bundles.

Interpreter: C:/Program Files/Python314/python.exe  (the repo .venv is broken)
Usage (from repo root):
    & "C:/Program Files/Python314/python.exe" export_diagnostics.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, "backend")
from model_service import ModelService, ABSTAIN_CONFIDENCE, ABSTAIN_MARGIN  # noqa: E402

LABELS = ["Low", "Medium", "High"]
SERVING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving")
FRONTEND_DIR = os.path.join(SERVING_DIR, "frontend_data")


def load_demo(serving_dir):
    with open(os.path.join(serving_dir, "demo_samples.json"), encoding="utf-8") as f:
        d = json.load(f)
    return d["samples"], d.get("note", "")


def reliability_table(confidences, correct, n_bins=10):
    """Per-bin calibration table + Expected Calibration Error (ECE).

    ECE matches the definition in compare_936_vs_live.py: the confidence-
    weighted average gap between mean confidence and accuracy per bin.
    """
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    table, ece = [], 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        count = int(m.sum())
        if count:
            avg_conf = float(conf[m].mean())
            acc = float(correct[m].mean())
            ece += (count / n) * abs(acc - avg_conf)
        else:
            avg_conf, acc = None, None
        table.append({
            "bin_lo": round(float(lo), 3),
            "bin_hi": round(float(hi), 3),
            "avg_confidence": avg_conf,
            "accuracy": acc,
            "count": count,
        })
    return table, float(ece)


def coverage_curve(confidences, correct, n_points=21):
    """Accuracy vs coverage: sort by confidence descending, and for each
    coverage fraction report the accuracy on the most-confident subset. This is
    the evidence behind the shipped abstain behaviour — if the model is well
    ordered by confidence, accuracy should rise as coverage falls."""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(conf)[::-1]
    correct_sorted = correct[order]
    n = len(conf)
    out = []
    for frac in np.linspace(1.0, 0.05, n_points):
        k = max(1, int(round(frac * n)))
        out.append({
            "coverage": round(k / n, 4),
            "accuracy": round(float(correct_sorted[:k].mean()), 4),
            "n": k,
        })
    # De-dup identical coverage points (small n causes repeats), keep last.
    dedup = {}
    for row in out:
        dedup[row["coverage"]] = row
    return sorted(dedup.values(), key=lambda r: r["coverage"])


def main():
    svc = ModelService(serving_dir=SERVING_DIR)
    samples, demo_note = load_demo(SERVING_DIR)

    y_true, y_pred, confidences, correct, latencies = [], [], [], [], []
    per_subject = defaultdict(lambda: {"n": 0, "correct": 0, "conf_sum": 0.0})

    for s in samples:
        t0 = time.perf_counter()
        out = svc.predict_single(s["features"])
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred = out["predicted_index"]
        truth = int(s["true_label"])
        conf = float(out["uncertainty"]["confidence"])
        is_correct = int(pred == truth)

        y_true.append(truth)
        y_pred.append(pred)
        confidences.append(conf)
        correct.append(is_correct)

        ps = per_subject[s["subject_id"]]
        ps["n"] += 1
        ps["correct"] += is_correct
        ps["conf_sum"] += conf

    reliability, ece = reliability_table(confidences, correct)
    cov = coverage_curve(confidences, correct)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    lat = np.asarray(latencies)

    per_subject_out = [
        {
            "subject_id": sid,
            "n": v["n"],
            "accuracy": round(v["correct"] / v["n"], 4),
            "mean_confidence": round(v["conf_sum"] / v["n"], 4),
        }
        for sid, v in sorted(per_subject.items())
    ]

    diagnostics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "export_diagnostics.py — frozen hybrid_rf_ensemble over serving/demo_samples.json",
        "note": (
            "Calibration/diagnostics on the held-out demo windows (fixed-split, "
            f"{len(samples)} windows, {len(per_subject_out)} subjects). The served "
            "model is unchanged; these are read-only analysis of its outputs. Not "
            "a LOSO estimate."
        ),
        "served_model": svc.served_model,
        "n_windows": len(samples),
        "overall": {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "ece": round(ece, 4),
        },
        "reliability": {
            "labels": LABELS,
            "ece": round(ece, 4),
            "bins": reliability,
        },
        "coverage_curve": cov,
        "abstain_thresholds": {
            "confidence": ABSTAIN_CONFIDENCE,
            "margin": ABSTAIN_MARGIN,
        },
        "per_subject": per_subject_out,
        "demo_confusion": {"labels": LABELS, "matrix": cm},
        "latency": {
            "mean_ms": round(float(lat.mean()), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
        },
    }

    os.makedirs(FRONTEND_DIR, exist_ok=True)
    out_path = os.path.join(FRONTEND_DIR, "diagnostics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    o = diagnostics["overall"]
    print(f"Wrote {out_path}")
    print(f"  windows={diagnostics['n_windows']}  subjects={len(per_subject_out)}")
    print(f"  accuracy={o['accuracy']}  macro_f1={o['macro_f1']}  ECE={o['ece']}")
    print(f"  latency mean={diagnostics['latency']['mean_ms']}ms "
          f"p95={diagnostics['latency']['p95_ms']}ms")


if __name__ == "__main__":
    main()
