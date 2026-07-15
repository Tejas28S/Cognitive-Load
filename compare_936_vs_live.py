"""
Compare the 936 (EEG+HbO+HbR) prototype vs the live 468 (EEG+HbO) served
ensemble, loading BOTH through the real backend ModelService so this also
proves the backend can serve the 936 artifacts end-to-end.

Metrics on the fixed-split demo test windows: macro F1, accuracy, per-class
recall, calibration (ECE + mean confidence), and per-prediction latency.

Usage: python compare_936_vs_live.py
Output: outputs/compare_936_vs_live.json
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, "backend")
from model_service import ModelService  # noqa: E402

LABELS = ["Low", "Medium", "High"]


def load_demo(serving_dir):
    with open(os.path.join(serving_dir, "demo_samples.json"), encoding="utf-8") as f:
        d = json.load(f)
    feats = [s["features"] for s in d["samples"]]
    y = np.array([s["true_label"] for s in d["samples"]])
    return feats, y


def ece(confidences, correct, n_bins=10):
    """Expected Calibration Error over max-prob confidence."""
    conf = np.asarray(confidences)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def eval_dir(name, serving_dir):
    svc = ModelService(serving_dir=serving_dir)
    feats, y = load_demo(serving_dir)

    # Latency: time each single prediction (the backend's real call path).
    preds, confs, times = [], [], []
    probs_all = []
    for f in feats:
        t0 = time.perf_counter()
        out = svc.predict_single(f)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
        p = np.array([out["probabilities"][c] for c in LABELS])
        probs_all.append(p)
        preds.append(out["predicted_index"])
        confs.append(float(p.max()))
    preds = np.array(preds)

    correct = (preds == y)
    res = {
        "serving_dir": serving_dir,
        "n_fnirs": int(svc.cfg["n_fnirs_features"]),
        "n_windows": int(len(y)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "accuracy": float(accuracy_score(y, preds)),
        "per_class_recall": [float(x) for x in
                             recall_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0)],
        "ece": ece(confs, correct),
        "mean_confidence": float(np.mean(confs)),
        "mean_confidence_correct": float(np.mean([c for c, ok in zip(confs, correct) if ok]) if correct.any() else 0.0),
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
    }
    print(f"[{name}] n_fnirs={res['n_fnirs']} f1={res['macro_f1']:.4f} "
          f"acc={res['accuracy']:.4f} ece={res['ece']:.4f} "
          f"conf={res['mean_confidence']:.3f} lat={res['latency_ms_mean']:.2f}ms")
    return res


def main():
    live = eval_dir("live 468", "serving")
    proto = eval_dir("proto 936", "serving_936")
    out = {
        "live_468": live,
        "proto_936": proto,
        "delta_936_minus_468": {
            k: round(proto[k] - live[k], 4)
            for k in ("macro_f1", "accuracy", "ece", "mean_confidence",
                      "latency_ms_mean")
        },
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/compare_936_vs_live.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\ndelta (936 - 468):", out["delta_936_minus_468"])
    print("Saved outputs/compare_936_vs_live.json")


if __name__ == "__main__":
    main()
