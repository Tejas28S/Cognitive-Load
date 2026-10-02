"""
export_binary_attention.py - rewrite serving/frontend_data/attention.json with
the BINARY model's real attention-gate weights.

The binary serving bundle shipped the original 3-class study's attention
summary (export_for_serving.py reused outputs/attention_weights_final.csv
unconditionally, and that CSV was only ever produced against the old 3-class
checkpoint). So the "Modality trust by class" panel showed a "Medium (2-back)
load" row for a model that can't predict Medium.

This runs the real served model (backend.model_service.ModelService, the same
class + serving/model_weights.pt the API uses - so there is no risk of
reconstructing the wrong architecture) over serving/demo_samples.json, the
same fixed preprocessed windows export_diagnostics.py evaluates, and
aggregates the per-sample attention_weights already returned by
predict_single() by true label.

    .venv/Scripts/python export_binary_attention.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "backend")
from model_service import ModelService  # noqa: E402

SERVING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving")
OUT_PATH = os.path.join(SERVING_DIR, "frontend_data", "attention.json")


def main():
    svc = ModelService(serving_dir=SERVING_DIR)
    with open(os.path.join(SERVING_DIR, "demo_samples.json"), encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    eeg_by_label = {i: [] for i in range(len(svc.class_names))}
    fnirs_by_label = {i: [] for i in range(len(svc.class_names))}
    eeg_all, fnirs_all = [], []

    for s in samples:
        out = svc.predict_single(s["features"])
        aw = out["attention_weights"]
        truth = int(s["true_label"])
        eeg_by_label[truth].append(aw["eeg"])
        fnirs_by_label[truth].append(aw["fnirs"])
        eeg_all.append(aw["eeg"])
        fnirs_all.append(aw["fnirs"])

    def summarize(eeg_vals, fnirs_vals):
        e, f = float(np.mean(eeg_vals)), float(np.mean(fnirs_vals))
        total = e + f
        return {
            "eeg": e, "fnirs": f,
            "eeg_share": e / total if total else 0.5,
            "fnirs_share": f / total if total else 0.5,
        }

    overall = summarize(eeg_all, fnirs_all)
    per_class = [
        {"class": name, **summarize(eeg_by_label[i], fnirs_by_label[i])}
        for i, name in enumerate(svc.class_names)
    ]

    out = {"overall": overall, "per_class": per_class}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_PATH}  (n={len(samples)})")
    print(f"  overall eeg_share={overall['eeg_share']:.4f} fnirs_share={overall['fnirs_share']:.4f}")
    for row in per_class:
        print(f"  {row['class']:<8} eeg_share={row['eeg_share']:.4f} fnirs_share={row['fnirs_share']:.4f}")


if __name__ == "__main__":
    main()
