"""
pipeline.py: the end-to-end universal prediction path.

    raw file(s) -> DatasetParser -> SchemaNormalizer -> [ChannelProjector ->
    feature extraction] -> EXISTING UploadService preprocessing -> EXISTING
    frozen model -> prediction + mapping report

The last two stages are the untouched originals: `UploadService.preprocess`
applies the exact per-subject z-score / log / saved-scaler recipe the model was
fit under, and `ModelService.predict_batch` is the frozen checkpoint. Nothing in
this package modifies model weights, architecture, or the LOSO evaluation logic.

Every prediction is returned WITH its mapping and projection reports. When
channel coverage falls below the refusal threshold in
schema/canonical_schema.yaml, this module returns a structured refusal naming
exactly what is missing instead of a confidently-wrong number.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from .canonical import (          # noqa: E402
    AliasIndex,
    IngestionError,
    load_canonical_schema,
    load_dataset_profiles,
)
from .features import extract_feature_frame, label_windows_from_events   # noqa: E402
from .normalizer import SchemaNormalizer                                  # noqa: E402
from .parser import DatasetParser                                         # noqa: E402
from .projection import ChannelProjector, combine_verdicts                # noqa: E402

BUNDLES = {"binary": "serving", "3class": "serving_3class"}


class UniversalPredictor:
    """Predict from any supported dataset file, against any serving bundle."""

    def __init__(self, version="binary", serving_dir=None):
        from model_service import ModelService
        from upload_service import UploadService

        self.version = version
        self.serving_dir = serving_dir or os.path.join(
            _ROOT, BUNDLES.get(version, "serving"))

        self.model_service = ModelService(serving_dir=self.serving_dir)
        self.upload_service = UploadService(serving_dir=self.serving_dir)

        self.schema = load_canonical_schema()
        self.profiles = load_dataset_profiles()
        self.aliases = AliasIndex()
        self.parser = DatasetParser(self.profiles, self.aliases)
        self.normalizer = SchemaNormalizer(self.aliases, self.schema)
        self.projector = ChannelProjector(self.schema, self.aliases)

        self.target_columns = (self.upload_service.eeg_cols
                               + self.upload_service.fnirs_cols)

    # ── Public entry point ──────────────────────────────────────────────────

    def predict(self, files, dataset=None, subject_id=None, max_windows=None):
        """Run the full pipeline.

        `files` is a list of {path|content, filename} dicts. Raw recordings
        often arrive as one file per modality (an EEG file and an fNIRS file),
        so multiple files are merged into a single recording.
        """
        parsed = []
        for f in files:
            parsed.append(self.parser.parse(
                path=f.get("path"), content=f.get("content"),
                filename=f.get("filename"), dataset=dataset,
            ))

        kinds = {p.kind for p in parsed}
        if kinds == {"features"}:
            return self._predict_from_features(parsed, dataset)
        if kinds == {"signals"}:
            return self._predict_from_signals(parsed, dataset, subject_id, max_windows)
        raise IngestionError(
            "Mixed input types: some files are extracted feature tables and "
            "others are raw recordings. Upload one kind at a time.",
            {"kinds": sorted(kinds)},
        )

    # ── Feature-table path ──────────────────────────────────────────────────

    def _predict_from_features(self, parsed, dataset):
        if len(parsed) > 1:
            raise IngestionError(
                "Multiple feature tables were uploaded; send one at a time.",
                {"n_files": len(parsed)},
            )
        result = parsed[0]
        df = result.dataframe

        frame, report = self.normalizer.normalize_features(
            df, self.target_columns,
            label_column=self.upload_service.label_col,
            subject_column=self.upload_service.subject_col,
            dataset=dataset or result.detected_dataset,
            optional_ignored_prefixes=tuple(
                self.upload_service.schema.get("optional_ignored_prefixes", [])),
        )
        report.warnings.extend(result.warnings)

        if not report.ok:
            return self._refusal(
                "missing_required_features",
                f"{len(report.missing_required)} required feature column(s) are "
                f"missing after alias matching.",
                {"missing_required": report.missing_required[:50],
                 "n_missing": len(report.missing_required),
                 "coverage": report.coverage},
                mapping_report=report, source=result,
            )

        return self._run_model(frame, report, source=result,
                               projection_reports=[], verdict="confident")

    # ── Raw-signal path ─────────────────────────────────────────────────────

    def _predict_from_signals(self, parsed, dataset, subject_id, max_windows):
        # Merge per-modality files into one recording.
        merged = None
        mapping_reports = []
        for result in parsed:
            rec, rep = self.normalizer.normalize_recording(result.recording)
            rep.warnings.extend(result.warnings)
            mapping_reports.append(rep)
            if merged is None:
                merged = rec
                continue
            if rec.eeg is not None and merged.eeg is None:
                merged.eeg, merged.eeg_channels, merged.eeg_fs = (
                    rec.eeg, rec.eeg_channels, rec.eeg_fs)
                merged.events = merged.events or rec.events
            if rec.hbo is not None and merged.hbo is None:
                merged.hbo, merged.hbr, merged.fnirs_fs = (
                    rec.hbo, rec.hbr, rec.fnirs_fs)
            merged.warnings.extend(rec.warnings)
            merged.provenance.update(rec.provenance)
            merged.dataset = merged.dataset or rec.dataset
            merged.subject_id = merged.subject_id or rec.subject_id

        combined = _merge_reports(mapping_reports)
        profile = self.profiles.get(dataset or merged.dataset or "", {})

        # Project onto the bundle's channel layout.
        eeg_arr, eeg_rep = self.projector.project_eeg(merged)
        hbo_arr, hbr_arr, fn_rep = self.projector.project_fnirs(merged, profile)
        verdict, blame = combine_verdicts(eeg_rep, fn_rep)

        if verdict == "refuse":
            return self._refusal(
                "insufficient_channel_coverage",
                f"The '{blame}' channels of this dataset cannot be mapped onto "
                f"the '{self.version}' model's expected layout with enough "
                f"coverage to produce a trustworthy prediction.",
                {
                    "verdict": verdict,
                    "blame_modality": blame,
                    "eeg": eeg_rep.to_dict() if eeg_rep else None,
                    "fnirs": fn_rep.to_dict() if fn_rep else None,
                    "thresholds": self.schema["projection"]["thresholds"],
                    "remedy": (
                        "This model was trained on a different sensor montage. "
                        "Predicting for this dataset requires a serving bundle "
                        "trained on its own feature space — see "
                        "docs/adding_a_new_dataset.md."
                    ),
                },
                mapping_report=combined,
                projection_reports=[r for r in (eeg_rep, fn_rep) if r],
            )

        # Match the bundle's sample rates before extracting features.
        fl = self.schema["feature_level"]
        eeg_target_fs = float(profile.get("eeg", {}).get("resample_to_hz")
                              or merged.eeg_fs or 200.0)
        fn_target_fs = float(profile.get("fnirs", {}).get("resample_to_hz")
                             or merged.fnirs_fs or 10.0)
        eeg_arr, eeg_rs = self.projector.resample(eeg_arr, merged.eeg_fs, eeg_target_fs)
        hbo_arr, fn_rs = self.projector.resample(hbo_arr, merged.fnirs_fs, fn_target_fs)
        hbr_arr, _ = self.projector.resample(hbr_arr, merged.fnirs_fs, fn_target_fs)

        win = fl["windowing"]
        frame = extract_feature_frame(
            eeg=eeg_arr, eeg_fs=eeg_target_fs,
            hbo=hbo_arr, hbr=hbr_arr, fnirs_fs=fn_target_fs,
            window_sec=float(win["window_sec"]),
            stride_sec=float(win["stride_sec"]),
            subject_id=subject_id or merged.subject_id or "UPLOAD",
            eeg_band=tuple(fl["filtering"]["eeg_bandpass_hz"]),
            fnirs_band=tuple(fl["filtering"]["fnirs_bandpass_hz"]),
            max_windows=max_windows,
        )

        # Labels from events, when the profile says they live there.
        labels_cfg = profile.get("labels") or {}
        if labels_cfg.get("source") == "eeg_events" and merged.events:
            label_map = self.aliases.label_value_map(labels_cfg.get("value_map", ""))
            if label_map:
                frame = label_windows_from_events(
                    frame, merged.events, label_map,
                    window_sec=float(win["window_sec"]),
                    label_column=self.upload_service.label_col,
                )

        if max_windows and len(frame) >= max_windows:
            combined.warnings.append(
                f"Limited to the first {max_windows} windows of the recording.")

        combined.warnings.extend(merged.warnings)
        combined.notes["resampling"] = {"eeg": eeg_rs, "fnirs": fn_rs}

        frame, feat_report = self.normalizer.normalize_features(
            frame, self.target_columns,
            label_column=self.upload_service.label_col,
            subject_column=self.upload_service.subject_col,
            dataset=merged.dataset,
        )
        combined.coverage.update(feat_report.coverage)
        combined.warnings.extend(feat_report.warnings)

        if not feat_report.ok:
            return self._refusal(
                "missing_required_features",
                "Feature extraction did not produce every column the model "
                "requires.",
                {"missing_required": feat_report.missing_required[:50],
                 "n_missing": len(feat_report.missing_required)},
                mapping_report=combined,
                projection_reports=[r for r in (eeg_rep, fn_rep) if r],
            )

        return self._run_model(
            frame, combined, source=parsed[0],
            projection_reports=[r for r in (eeg_rep, fn_rep) if r],
            verdict=verdict,
        )

    # ── Model invocation (existing services, unchanged) ─────────────────────

    def _run_model(self, frame, report, source, projection_reports, verdict):
        # A NaN feature would silently become 0 inside the existing
        # preprocessing; refuse instead, so an unfillable channel can never be
        # laundered into a real-looking value.
        feature_block = frame[self.target_columns]
        nan_cols = feature_block.columns[feature_block.isna().any()].tolist()
        if nan_cols:
            return self._refusal(
                "unfilled_features",
                f"{len(nan_cols)} feature column(s) could not be filled from "
                f"this dataset and would have to be fabricated to run the model.",
                {"unfilled_columns": nan_cols[:50], "n_unfilled": len(nan_cols)},
                mapping_report=report, projection_reports=projection_reports,
            )

        features, subjects, labels, norm_mode = self.upload_service.preprocess(frame)
        results = self.model_service.predict_batch(features.tolist())

        short = self.upload_service.class_names_short
        rows, dist = [], {name: 0 for name in short}
        for i, r in enumerate(results):
            dist[short[r["predicted_index"]]] += 1
            row = {
                "row": i,
                "subject_id": str(subjects[i]),
                "predicted_label": r["predicted_label"],
                "predicted_index": r["predicted_index"],
                "probabilities": r["probabilities"],
                "confidence": r["uncertainty"]["confidence"],
                "abstain": r["uncertainty"]["abstain"],
            }
            if labels is not None:
                row["true_label"] = (int(labels[i])
                                     if str(labels[i]).lstrip("-").isdigit() else None)
            rows.append(row)

        accuracy = None
        if labels is not None:
            scored = [r for r in rows
                      if r.get("true_label") is not None and r["true_label"] >= 0]
            if scored:
                accuracy = sum(1 for r in scored
                               if r["true_label"] == r["predicted_index"]) / len(scored)

        if verdict == "degraded":
            report.warnings.insert(0, (
                "DEGRADED PREDICTION: some model input channels were filled by "
                "approximate spatial-neighbour substitution rather than the "
                "electrodes the model was trained on. Treat these outputs as "
                "indicative, not measured."
            ))

        return {
            "ok": True,
            "refused": False,
            "verdict": verdict,
            "model_version": self.version,
            "n_rows": len(rows),
            "normalization_mode": norm_mode,
            "distribution": dist,
            "accuracy_vs_uploaded_labels": accuracy,
            "predictions": rows,
            "schema_mapping_report": report.to_dict(),
            "projection_reports": [r.to_dict() for r in projection_reports],
            "source": {
                "format": source.source_format,
                "detected_dataset": source.detected_dataset,
                "layout": source.layout,
            },
            "warnings": report.warnings,
        }

    # ── Refusal ─────────────────────────────────────────────────────────────

    def _refusal(self, code, message, detail, mapping_report=None,
                 projection_reports=None, source=None):
        return {
            "ok": False,
            "refused": True,
            "verdict": "refuse",
            "error_code": code,
            "message": message,
            "detail": detail,
            "model_version": self.version,
            "schema_mapping_report": (mapping_report.to_dict()
                                      if mapping_report else None),
            "projection_reports": [r.to_dict() for r in (projection_reports or [])],
            "source": ({"format": source.source_format,
                        "detected_dataset": source.detected_dataset,
                        "layout": source.layout} if source else None),
            "warnings": (mapping_report.warnings if mapping_report else []),
        }


def _merge_reports(reports):
    """Fold several per-file mapping reports into one."""
    from .normalizer import MappingReport
    out = MappingReport()
    for r in reports:
        out.dataset = out.dataset or r.dataset
        out.mapping.update(r.mapping)
        out.unmatched_input.extend(r.unmatched_input)
        out.missing_required.extend(r.missing_required)
        out.missing_optional.extend(r.missing_optional)
        out.filled_defaults.update(r.filled_defaults)
        out.approximate.update(r.approximate)
        out.warnings.extend(r.warnings)
        out.coverage.update(r.coverage)
        out.notes.update(r.notes)
    return out
