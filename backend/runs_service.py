"""
runs_service.py: read-only access to the run archive under outputs/runs/.

Every endpoint built on this returns bytes that a training run wrote to disk.
Nothing is computed here and nothing is cached across requests beyond a cheap
directory listing, so the API can never disagree with the artifacts.

A run is self-describing: its own dataset_info.json carries the class names,
the classification mode and the evaluation protocol, so the frontend reads
labels from the run rather than assuming binary or 3-class.
"""

from __future__ import annotations

import json
import os
import time

RUNS_DIRNAME = os.path.join("outputs", "runs")


class RunNotFound(Exception):
    pass


class ArtifactNotFound(Exception):
    pass


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class RunsService:
    def __init__(self, repo_root, runs_root=None):
        self.repo_root = os.path.abspath(repo_root)
        self.runs_root = os.path.abspath(
            runs_root or os.path.join(self.repo_root, RUNS_DIRNAME)
        )
        self._index = None
        self._index_at = 0.0
        self._ttl = 2.0  # seconds; a run in progress rewrites status often

    # ── paths ──────────────────────────────────────────────────────────
    def run_dir(self, run_id):
        # Reject anything that is not a plain directory name.
        if not run_id or os.path.sep in run_id or run_id in (".", ".."):
            raise RunNotFound(run_id)
        d = os.path.join(self.runs_root, run_id)
        if os.path.commonpath([os.path.abspath(d), self.runs_root]) != self.runs_root:
            raise RunNotFound(run_id)
        if not os.path.isdir(d):
            raise RunNotFound(run_id)
        return d

    def artifact_path(self, run_id, relpath):
        base = self.run_dir(run_id)
        target = os.path.abspath(os.path.join(base, relpath))
        if os.path.commonpath([target, os.path.abspath(base)]) != os.path.abspath(base):
            raise ArtifactNotFound(relpath)
        if not os.path.isfile(target):
            raise ArtifactNotFound(relpath)
        return target

    def read_artifact(self, run_id, relpath):
        return _read_json(self.artifact_path(run_id, relpath))

    def try_read(self, run_id, relpath, default=None):
        try:
            return self.read_artifact(run_id, relpath)
        except (ArtifactNotFound, json.JSONDecodeError):
            return default

    def has(self, run_id, relpath):
        try:
            self.artifact_path(run_id, relpath)
            return True
        except (RunNotFound, ArtifactNotFound):
            return False

    # ── listing ────────────────────────────────────────────────────────
    def _summarize(self, run_id):
        status = self.try_read(run_id, "run_status.json", {}) or {}
        meta = self.try_read(run_id, "run_metadata.json", {}) or {}
        summary = self.try_read(run_id, "run_summary.json", {}) or {}
        ds = meta.get("dataset", {}) or {}

        # run_summary.json has two shapes in the archive: the canonical
        # pipeline's "selected" schema (results.selected.loso_all_subjects/
        # lockbox) and the per-candidate "promote" schema that
        # target_campaign.py writes (results.candidate.loso26/lockbox). Both
        # name the LOSO block after its own fold count, so try both instead
        # of assuming the run came from one particular script.
        results = summary.get("results") or {}
        candidate_block = results.get("selected") or results.get("candidate") or {}
        loso = candidate_block.get("loso_all_subjects") or candidate_block.get("loso26") or {}
        lock = candidate_block.get("lockbox") or {}

        def m(block, key):
            v = (block or {}).get(key) or {}
            return v.get("mean") if isinstance(v, dict) else None

        protocol = (summary.get("protocol") or {}).get("headline")
        if not protocol and loso.get("n_folds"):
            protocol = f"{loso['n_folds']}-fold LOSO"
            candidate_name = summary.get("candidate")
            if candidate_name:
                protocol += f" ({candidate_name})"

        dataset_name = ds.get("csv_basename") or (summary.get("dataset") or {}).get("name")
        mode = ds.get("classification_mode")
        class_names = ds.get("class_names")
        n_experiments = summary.get("n_experiments")
        if n_experiments is None:
            n_experiments = (status.get("summary") or {}).get("n_candidates")

        # run_metadata.json is only written by the canonical pipeline
        # (research/runctx.py's write_metadata). target_campaign.py and
        # ensemble_campaign.py never call it, so a run from either one has no
        # run_metadata.json at all, so recover the same facts from whatever
        # artifact that script actually wrote, rather than leaving genuinely
        # known values blank.
        if not dataset_name or not mode:
            candidates = self.try_read(
                run_id, "target_experiments/all_records_summary.json"
            )
            if isinstance(candidates, dict) and candidates:
                first = next(iter(candidates.values()), None) or {}
                cand_data = first.get("data") or {}
                dataset_name = dataset_name or cand_data.get("dataset")
                if mode is None and "binary" in cand_data:
                    mode = "binary" if cand_data["binary"] else "3class"
                if n_experiments is None:
                    n_experiments = len(candidates)
            else:
                ens = self.try_read(run_id, "ensemble_dev_summary.json")
                pooled = (ens or {}).get("pooled") or {}
                if pooled.get("class_names"):
                    class_names = class_names or pooled["class_names"]
                    if mode is None:
                        mode = "binary" if pooled.get("n_classes") == 2 else "3class"

        # A run whose process was killed keeps status "running" forever. Report
        # that as stale rather than rewriting what the run itself recorded.
        state = status.get("status", "unknown")
        stale = False
        if state == "running":
            updated = status.get("updated_at")
            if updated:
                try:
                    from datetime import datetime, timezone
                    ts = datetime.fromisoformat(updated)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - ts).total_seconds()
                    stale = age > 1800
                except ValueError:
                    pass

        return {
            "run_id": run_id,
            "status": state,
            "stale": stale,
            "stage": status.get("stage"),
            "tag": status.get("tag") or meta.get("tag"),
            "created_at": status.get("created_at") or meta.get("created_at"),
            "updated_at": status.get("updated_at"),
            "error": status.get("error"),
            "warnings": status.get("warnings") or [],
            "dataset": dataset_name,
            "n_samples": ds.get("n_samples"),
            "n_subjects": ds.get("n_subjects"),
            "classification_mode": mode,
            "class_names": class_names,
            "protocol": protocol,
            "selected_experiment": (summary.get("model") or {}).get("selected_experiment"),
            "n_experiments": n_experiments,
            "git_commit": (meta.get("git") or {}).get("commit"),
            "metrics": {
                "loso_f1_macro": m(loso, "f1_macro"),
                "loso_accuracy": m(loso, "accuracy"),
                "loso_roc_auc": m(loso, "roc_auc"),
                "loso_mcc": m(loso, "mcc"),
                "lockbox_f1_macro": m(lock, "f1_macro"),
                "lockbox_accuracy": m(lock, "accuracy"),
            },
            "has_summary": bool(summary),
        }

    def list_runs(self, force=False):
        now = time.time()
        if not force and self._index is not None and now - self._index_at < self._ttl:
            return self._index
        rows = []
        if os.path.isdir(self.runs_root):
            for name in os.listdir(self.runs_root):
                if not os.path.isdir(os.path.join(self.runs_root, name)):
                    continue
                try:
                    rows.append(self._summarize(name))
                except Exception as exc:  # a half-written run must not 500 the list
                    rows.append({
                        "run_id": name, "status": "unreadable",
                        "error": {"message": f"{type(exc).__name__}: {exc}"},
                        "warnings": [], "metrics": {},
                    })
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        self._index, self._index_at = rows, now
        return rows

    # ── detail ─────────────────────────────────────────────────────────
    def get_run(self, run_id):
        self.run_dir(run_id)
        return {
            "run_id": run_id,
            "status": self.try_read(run_id, "run_status.json", {}),
            "metadata": self.try_read(run_id, "run_metadata.json", {}),
            "config": self.try_read(run_id, "config.json", {}),
            "dataset_info": self.try_read(run_id, "dataset_info.json", {}),
            "summary": self.try_read(run_id, "run_summary.json", {}),
            "leakage_audit": self.try_read(run_id, "leakage_audit.json", {}),
            "available": self.available_artifacts(run_id),
        }

    def available_artifacts(self, run_id):
        """What this run actually produced: the UI hides sections without data."""
        checks = {
            "metrics": "metrics/metrics.json",
            "loso": "metrics/loso_metrics.json",
            "per_class": "metrics/per_class_metrics.json",
            "per_subject": "metrics/per_subject_metrics.csv",
            "confusion_matrix": "confusion_matrix/confusion_matrix.json",
            "curves": "curves/curves.json",
            "training_history": "curves/training_history.json",
            "feature_importance": "feature_importance/shap_values.json",
            "feature_ranking": "feature_importance/feature_ranking.csv",
            "representations": "representations/representations.json",
            "predictions": "predictions/predictions.csv",
            "summary": "run_summary.json",
            "summary_md": "run_summary.md",
            "model": "model/model_config.json",
        }
        out = {k: self.has(run_id, v) for k, v in checks.items()}
        out["experiments"] = self.has(run_id, "experiments/experiment_results.json")
        out["leaderboard"] = self.has(run_id, "experiments/leaderboard.csv")
        out["figures"] = self.list_figures(run_id)
        return out

    def list_figures(self, run_id):
        base = self.run_dir(run_id)
        figs = []
        for sub in ("curves", "confusion_matrix", "feature_importance",
                    "representations", "experiments"):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if name.lower().endswith(".png"):
                    figs.append(f"{sub}/{name}")
        return figs

    # ── experiments ────────────────────────────────────────────────────
    def experiments(self, run_id):
        records = self.try_read(run_id, "experiments/experiment_results.json", []) or []
        stage_log = self.try_read(run_id, "experiments/stage_log.json", []) or []
        compact = []
        for r in records:
            agg = r.get("fold_aggregate") or {}

            def m(key, stat="mean"):
                v = agg.get(key) or {}
                return v.get(stat) if isinstance(v, dict) else None

            compact.append({
                "experiment_id": r.get("experiment_id"),
                "stage": r.get("stage"),
                "note": r.get("note"),
                "dataset": r.get("dataset_name"),
                "encoder": (r.get("spec") or {}).get("encoder"),
                "fusion": (r.get("spec") or {}).get("fusion"),
                "n_params": r.get("n_params"),
                "n_folds": r.get("n_folds"),
                "runtime_seconds": r.get("runtime_seconds"),
                "leakage_audit_passed": r.get("leakage_audit_passed"),
                "spec": r.get("spec"),
                "data": r.get("data"),
                "f1_macro_mean": m("f1_macro"),
                "f1_macro_std": m("f1_macro", "std"),
                "accuracy_mean": m("accuracy"),
                "balanced_accuracy_mean": m("balanced_accuracy"),
                "roc_auc_mean": m("roc_auc"),
                "mcc_mean": m("mcc"),
                "cohen_kappa_mean": m("cohen_kappa"),
                "pooled_f1_macro": (r.get("pooled") or {}).get("f1_macro"),
                "generalization_gap": r.get("generalization_gap"),
                "vs_baseline": r.get("vs_baseline"),
                "modality_gates": r.get("modality_gates"),
            })
        compact.sort(key=lambda r: (r["f1_macro_mean"] is None, -(r["f1_macro_mean"] or 0)))
        return {"experiments": compact, "stage_log": stage_log}

    def experiment_detail(self, run_id, exp_id):
        if os.path.sep in exp_id or exp_id in (".", ".."):
            raise ArtifactNotFound(exp_id)
        rec = self.try_read(run_id, f"experiments/{exp_id}/record.json")
        if rec is None:
            raise ArtifactNotFound(exp_id)
        return {
            "record": rec,
            "metrics": self.try_read(run_id, f"experiments/{exp_id}/metrics.json"),
            "curves": self.try_read(run_id, f"experiments/{exp_id}/curves.json"),
            "per_subject": self.try_read(
                run_id, f"experiments/{exp_id}/per_subject_metrics.json"),
            "fold_aggregate": self.try_read(
                run_id, f"experiments/{exp_id}/fold_aggregate.json"),
        }

    # ── predictions ────────────────────────────────────────────────────
    def predictions(self, run_id, limit=500, offset=0, subject=None):
        import pandas as pd

        path = self.artifact_path(run_id, "predictions/predictions.csv")
        df = pd.read_csv(path)
        total = len(df)
        if subject is not None:
            df = df[df["subject_id"] == int(subject)]
        filtered = len(df)
        page = df.iloc[offset: offset + limit]
        return {
            "total": int(total),
            "filtered": int(filtered),
            "offset": int(offset),
            "limit": int(limit),
            "columns": list(df.columns),
            "rows": json.loads(page.to_json(orient="records")),
        }
