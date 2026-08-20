"""
runctx.py — run archive.

A RunContext is created BEFORE any data is touched, so a run that crashes
during loading still leaves a directory behind with its metadata, config and
error. Nothing is ever overwritten: the run id embeds a timestamp plus a
random suffix, and creation fails loudly if the directory already exists.

Layout (section 25 of the brief, adapted to this project):

    outputs/runs/run_<YYYYmmdd>_<HHMMSS>_<hex4>/
        run_metadata.json      environment, dataset, git commit, seed
        config.json            the exact resolved config
        dataset_info.json      dataset provenance + schema
        run_status.json        created | running | completed | failed
        run_summary.json/.md   final summary (written at the end)
        leakage_audit.json     result of the leakage assertions
        metrics/  predictions/  confusion_matrix/  curves/
        feature_importance/  representations/  model/  experiments/  logs/
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import secrets
import subprocess
import sys
import traceback
from datetime import datetime, timezone

RUNS_ROOT = os.path.join("outputs", "runs")

SUBDIRS = (
    "metrics",
    "predictions",
    "confusion_matrix",
    "curves",
    "feature_importance",
    "representations",
    "model",
    "experiments",
    "logs",
    "folds",
)


def _git_info(repo_root):
    def _run(args):
        try:
            return subprocess.check_output(
                args, cwd=repo_root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _json_default(obj):
    """Make numpy / torch scalars and arrays JSON-serialisable."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _sanitize(obj):
    """Recursively turn NaN/Inf into null so the JSON stays strictly valid.

    A NaN is a real result (an undefined metric), not a serialisation problem,
    so it becomes an explicit null the frontend can render as "n/a" rather
    than a token that breaks JSON.parse.
    """
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_sanitize(payload), fh, indent=2, default=_json_default,
                  allow_nan=False)
    os.replace(tmp, path)
    return path


class RunContext:
    """Owns one run directory and everything written into it."""

    def __init__(self, repo_root=None, tag=None, runs_root=None):
        self.repo_root = os.path.abspath(repo_root or os.getcwd())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(2)
        self.run_id = f"run_{stamp}_{suffix}"
        self.tag = tag or ""
        root = runs_root or os.path.join(self.repo_root, RUNS_ROOT)
        self.dir = os.path.join(root, self.run_id)

        # exist_ok=False: a collision must never silently overwrite a run.
        os.makedirs(self.dir, exist_ok=False)
        for sub in SUBDIRS:
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)

        self.created_at = datetime.now(timezone.utc).isoformat()
        self._log_path = self.path("logs", "training.log")
        self._log_fh = open(self._log_path, "a", encoding="utf-8")
        self._status = "created"
        self._stage = "init"
        self._warnings = []

        self.write_status("created", stage="init")
        self.log(f"[run] {self.run_id} created at {self.dir}")

    # ── paths ──────────────────────────────────────────────────────────
    def path(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    # ── logging ────────────────────────────────────────────────────────
    def log(self, message, echo=True):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {message}"
        self._log_fh.write(line + "\n")
        self._log_fh.flush()
        if echo:
            print(line, flush=True)

    def warn(self, message):
        self._warnings.append(message)
        self.log(f"[warn] {message}")

    @property
    def warnings(self):
        return list(self._warnings)

    # ── status ─────────────────────────────────────────────────────────
    def write_status(self, status, stage=None, error=None, extra=None):
        self._status = status
        if stage:
            self._stage = stage
        payload = {
            "run_id": self.run_id,
            "status": status,
            "stage": self._stage,
            "tag": self.tag,
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": self._warnings,
        }
        if error:
            payload["error"] = error
        if extra:
            payload.update(extra)
        write_json(self.path("run_status.json"), payload)
        return payload

    def stage(self, name):
        self.log(f"[stage] {name}")
        self.write_status("running", stage=name)

    # ── metadata ───────────────────────────────────────────────────────
    def write_metadata(self, config, dataset_info, extra=None):
        import numpy
        import sklearn
        import torch

        meta = {
            "run_id": self.run_id,
            "tag": self.tag,
            "created_at": self.created_at,
            "git": _git_info(self.repo_root),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "user": getpass.getuser(),
                "torch": torch.__version__,
                "numpy": numpy.__version__,
                "sklearn": sklearn.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_available": bool(
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                ),
                "torch_threads": torch.get_num_threads(),
            },
            "config": config,
            "dataset": dataset_info,
        }
        if extra:
            meta.update(extra)
        write_json(self.path("run_metadata.json"), meta)
        write_json(self.path("config.json"), config)
        write_json(self.path("dataset_info.json"), dataset_info)
        return meta

    # ── lifecycle ──────────────────────────────────────────────────────
    def complete(self, summary=None):
        self.write_status("completed", stage="done", extra={"summary": summary or {}})
        self.log(f"[run] {self.run_id} completed")
        self.close()

    def fail(self, exc, stage=None):
        """Preserve everything; record the failure stage and traceback."""
        err = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.write_status("failed", stage=stage or self._stage, error=err)
        self.log(f"[run] {self.run_id} FAILED in stage '{self._stage}': {exc}")
        with open(self.path("logs", "error.log"), "w", encoding="utf-8") as fh:
            fh.write(err["traceback"])
        self.close()

    def close(self):
        try:
            self._log_fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.fail(exc)
            return False
        if self._status not in ("completed", "failed"):
            self.complete()
        return False
