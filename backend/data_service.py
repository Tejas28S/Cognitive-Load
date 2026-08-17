"""
data_service.py — loads demo windows and precomputed evaluation artifacts
from serving/ and serving/frontend_data/. No training-pipeline imports.
"""

import json
import os
from collections import defaultdict

SERVING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serving"
)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DataService:
    def __init__(self, serving_dir=SERVING_DIR):
        self.serving_dir = serving_dir
        demo = _load(os.path.join(serving_dir, "demo_samples.json"))
        self.samples = demo["samples"]
        self.n_eeg = demo["n_eeg"]
        self.n_fnirs = demo["n_fnirs"]
        self.demo_note = demo.get("note", "")

        # Group windows into per-subject/trial playback sessions.
        self.sessions = defaultdict(list)
        for s in self.samples:
            self.sessions[(s["subject_id"], s["trial_id"] // 1000)].append(s)
        # In this dataset each subject is one session (trial_id is the window idx).
        self.subject_sessions = defaultdict(list)
        for s in self.samples:
            self.subject_sessions[s["subject_id"]].append(s)
        for k in self.subject_sessions:
            self.subject_sessions[k].sort(key=lambda x: x["window_index"])

        self.frontend_dir = os.path.join(serving_dir, "frontend_data")
        self.frontend = {}
        for name in ["confusion_matrices", "shap", "attention", "metrics", "research", "diagnostics"]:
            path = os.path.join(self.frontend_dir, f"{name}.json")
            self.frontend[name] = _load(path) if os.path.exists(path) else {}

        # The served model's architecture/ensemble card (for a live "model card").
        cfg_path = os.path.join(serving_dir, "model_config.json")
        self.config = _load(cfg_path) if os.path.exists(cfg_path) else {}

    # ── Demo sessions ─────────────────────────────────────────────────────

    def list_subjects(self):
        out = []
        for subject_id, windows in sorted(self.subject_sessions.items()):
            out.append({
                "subject_id": subject_id,
                "trial_id": 0,
                "num_windows": len(windows),
                "true_labels": [w["true_label"] for w in windows],
                "true_label_names": [w["true_label_name"] for w in windows],
            })
        return out

    def get_session(self, subject_id, trial_id=0):
        windows = self.subject_sessions.get(subject_id, [])
        return [
            {
                "window_index": w["window_index"],
                "timestamp_s": w["timestamp_s"],
                "true_label": w["true_label"],
                "true_label_name": w["true_label_name"],
                "features": w["features"],
            }
            for w in windows
        ]

    def get_window(self, subject_id, window_index):
        for w in self.subject_sessions.get(subject_id, []):
            if w["window_index"] == window_index:
                return w
        return None

    # ── Precomputed artifacts ─────────────────────────────────────────────

    def metrics(self):
        return self.frontend.get("metrics", {})

    def confusion_matrix(self, model):
        conf = self.frontend.get("confusion_matrices", {})
        key = {"hybrid": "hybrid", "eeg": "eeg", "fnirs": "fnirs"}.get(model)
        if not key or key not in conf:
            return None
        return {"labels": conf.get("labels", ["Low", "Medium", "High"]),
                "matrix": conf[key], "model": model}

    def shap(self):
        return self.frontend.get("shap", {})

    def attention(self):
        return self.frontend.get("attention", {})

    def research(self):
        """Consolidated research-result artifacts (DANN, protocol benchmark, HbR
        ablation, classical sweep, significance) for the /research narrative."""
        return self.frontend.get("research", {})

    def diagnostics(self):
        """Calibration/reliability diagnostics for the frozen served ensemble,
        computed on the held-out demo windows (see export_diagnostics.py)."""
        return self.frontend.get("diagnostics", {})

    def model_config(self):
        """The served model's architecture + ensemble card (model_config.json)."""
        return self.config
