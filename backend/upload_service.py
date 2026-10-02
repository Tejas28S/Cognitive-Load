"""
upload_service.py: validate + preprocess an uploaded CSV of already-extracted
TU Berlin features, reproducing the served pipeline WITHOUT importing the
training code.

The preprocessing here is a faithful, self-contained reimplementation of
data_loader.CognitiveLoadDataLoader's final-config pipeline
(PER_SUBJECT_NORMALIZE_STAGE="before_log", EEG_LOG_MODE="clip",
FNIRS_SIGNED_LOG_TRANSFORM=True), verified byte-exact against it. The column
order + preprocessing params come from serving/feature_schema.json; the fitted
scalers come from serving/feature_scaler.pkl.
"""

import json
import os

import joblib
import numpy as np

SERVING_DIR = os.environ.get(
    "SERVING_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serving"),
)


class UploadService:
    def __init__(self, serving_dir=SERVING_DIR):
        # Kept so callers can locate sibling artifacts in this bundle (the
        # /api/predict/upload/template download reads upload_template.csv here).
        self.serving_dir = serving_dir
        with open(os.path.join(serving_dir, "feature_schema.json"), encoding="utf-8") as f:
            self.schema = json.load(f)
        self.eeg_cols = self.schema["eeg_columns"]
        self.fnirs_cols = self.schema["fnirs_columns"]
        self.label_col = self.schema["label_column"]
        self.subject_col = self.schema["subject_column"]
        self.prep = self.schema["preprocessing"]
        self.eps = float(self.prep["per_subject_eps"])
        self.class_names_short = self.schema["class_names_short"]

        scal = joblib.load(os.path.join(serving_dir, "feature_scaler.pkl"))
        self.eeg_scaler = scal["eeg_scaler"]
        self.fnirs_scaler = scal["fnirs_scaler"]

        self.required = self.eeg_cols + self.fnirs_cols
        self.known = set(
            self.eeg_cols + self.fnirs_cols
            + [self.label_col, self.subject_col]
            + self.schema.get("optional_ignored_columns", [])
        )
        self.ignored_prefixes = tuple(self.schema.get("optional_ignored_prefixes", []))

    # ── Validation ────────────────────────────────────────────────────────

    def validate_columns(self, columns):
        """Return (ok, detail). Rejects only on MISSING required feature columns;
        the standard-but-unused TU Berlin columns (hbr_*, task) are tolerated and
        truly unrecognized columns are reported as warnings, not hard failures."""
        cols = list(columns)
        colset = set(cols)
        missing = [c for c in self.required if c not in colset]
        has_subject = self.subject_col in colset
        has_label = self.label_col in colset
        unexpected = [
            c for c in cols
            if c not in self.known and not c.lower().startswith(self.ignored_prefixes)
        ]
        ok = len(missing) == 0
        return ok, {
            "ok": ok,
            "n_missing_required": len(missing),
            "missing_required": missing[:50],
            "has_subject_id": has_subject,
            "has_label": has_label,
            "unexpected_columns": unexpected[:50],
            "expected_eeg": len(self.eeg_cols),
            "expected_fnirs": len(self.fnirs_cols),
        }

    # ── Preprocessing (self-contained; verified against data_loader) ───────

    def _per_subject_zscore(self, arr, subjects):
        out = arr.copy()
        for s in np.unique(subjects):
            mask = subjects == s
            block = out[mask]
            mean = np.nanmean(block, axis=0)
            std = np.nanstd(block, axis=0, ddof=1)
            std = np.nan_to_num(std, nan=self.eps, posinf=self.eps, neginf=self.eps)
            std[std < self.eps] = self.eps
            out[mask] = (block - mean) / std
        return out

    def preprocess(self, df):
        """df -> (features [n,648] float32, subjects, labels|None, normalization_mode)."""
        eeg = df[self.eeg_cols].to_numpy(dtype=np.float32)
        fnirs = df[self.fnirs_cols].to_numpy(dtype=np.float32)

        # Clean NaNs / Infs (matches data_loader).
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
        fnirs = np.nan_to_num(fnirs, nan=0.0, posinf=0.0, neginf=0.0)

        if self.subject_col in df.columns:
            subjects = df[self.subject_col].astype(str).to_numpy()
            norm_mode = "per_subject"
        else:
            # No subject id: treat the whole upload as one group. Degraded,
            # the model was trained with per-subject stats, so flag it loudly.
            subjects = np.array(["__ALL__"] * len(df))
            norm_mode = "global_fallback_no_subject_id"

        # Per-subject z-score BEFORE log (final config stage).
        eeg = self._per_subject_zscore(eeg, subjects)
        fnirs = self._per_subject_zscore(fnirs, subjects)

        # Log transforms: EEG log1p on non-negative clip; fNIRS signed log1p.
        eeg = np.log1p(np.clip(eeg, a_min=0.0, a_max=None))
        fnirs = np.sign(fnirs) * np.log1p(np.abs(fnirs))

        # Train-fitted scalers.
        eeg = self.eeg_scaler.transform(eeg)
        fnirs = self.fnirs_scaler.transform(fnirs)

        features = np.hstack([eeg, fnirs]).astype(np.float32)

        labels = None
        if self.label_col in df.columns:
            labels = df[self.label_col].to_numpy()

        return features, subjects, labels, norm_mode
