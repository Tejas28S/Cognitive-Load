"""
datamod.py — dataset loading, preprocessing and leakage-safe fold generation.

Preprocessing contract (mirrors the project's locked order, made explicit and
auditable):

    nan_to_num  ->  [per-subject z-score]  ->  log compression  ->  scalers

* per-subject normalization reads ONLY the rows of the subject it is
  normalizing.  It therefore cannot move information between subjects, and
  `leakage.assert_subject_local` proves that numerically rather than asserting
  it in prose.  It IS transductive within a subject (it needs that subject's
  whole session), which is a deployment caveat, so a `calibration` mode is
  provided that uses only the first K windows of a session.
* the log transform is pointwise, so it is fold-independent by construction.
* the RobustScaler/StandardScaler are the only fold-dependent step and are fit
  on TRAIN INDICES ONLY, refit from scratch for every fold.

Splits are always grouped by subject.  Windows overlap in time within a trial,
so a subject-grouped split is the only one that keeps overlapping windows off
both sides of the split.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

# Label semantics of the TU Berlin n-back task, from preprocess_tuberlin.py.
RAW_CLASS_NAMES = {0: "Low (0-back)", 1: "Medium (2-back)", 2: "High (3-back)"}

META_COLUMNS = (
    "subject_id",
    "trial_id",
    "window_id",
    "window_start_sec",
    "window_sec",
    "source_trial_start_ms",
    "task",
    "Cognitive_Load",
)


def natural_key(value):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", value)]


def _sha256_file(path, limit_bytes=None):
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_bytes and read >= limit_bytes:
                break
    return h.hexdigest()


class FeatureSchema:
    """Column lists plus the (channels, features) grid each modality forms.

    The grid is what makes a 1D CNN legitimate here: the columns are
    channel-major (all of ch00's features, then ch01's, ...), so the block
    reshapes exactly to (n_channels, n_features_per_channel) and a conv can
    slide along the electrode / optode axis with the feature types as its
    input channels.
    """

    def __init__(self, eeg_cols, hbo_cols, hbr_cols):
        self.eeg_cols = eeg_cols
        self.hbo_cols = hbo_cols
        self.hbr_cols = hbr_cols
        self.fnirs_cols = hbo_cols + hbr_cols

        self.eeg_grid = self._grid(eeg_cols, r"^eeg_ch(\d+)_(.+)$")
        self.hbo_grid = self._grid(hbo_cols, r"^hbo_ch(\d+)_(.+)$")
        self.hbr_grid = self._grid(hbr_cols, r"^hbr_ch(\d+)_(.+)$")

    @staticmethod
    def _grid(cols, pattern):
        """Return (n_channels, n_feats, feat_names) if cols form a full grid."""
        if not cols:
            return None
        parsed = []
        for c in cols:
            m = re.match(pattern, c)
            if not m:
                return None
            parsed.append((int(m.group(1)), m.group(2)))
        channels = sorted({c for c, _ in parsed})
        feats = sorted({f for _, f in parsed})
        if len(parsed) != len(channels) * len(feats):
            return None
        # channel-major ordering check: cols must equal the flattened grid
        expected = [
            f"{cols[0].split('_ch')[0]}_ch{ch:02d}_{f}" for ch in channels for f in feats
        ]
        if expected != list(cols):
            return None
        return (len(channels), len(feats), feats)

    def to_dict(self):
        def g(grid):
            if grid is None:
                return None
            return {"n_channels": grid[0], "n_features_per_channel": grid[1],
                    "feature_names": grid[2]}

        return {
            "n_eeg": len(self.eeg_cols),
            "n_hbo": len(self.hbo_cols),
            "n_hbr": len(self.hbr_cols),
            "n_fnirs": len(self.fnirs_cols),
            "eeg_grid": g(self.eeg_grid),
            "hbo_grid": g(self.hbo_grid),
            "hbr_grid": g(self.hbr_grid),
            "eeg_columns_sha256": hashlib.sha256(
                "|".join(self.eeg_cols).encode()
            ).hexdigest()[:16],
            "fnirs_columns_sha256": hashlib.sha256(
                "|".join(self.fnirs_cols).encode()
            ).hexdigest()[:16],
        }


class Dataset:
    """Loaded + preprocessed (pre-scaler) feature matrices and metadata."""

    def __init__(self, csv_path, binary=True, per_subject_norm="transductive",
                 per_subject_norm_stage="before_log", calibration_windows=30,
                 eeg_log=True, fnirs_log=True, drop_hbr=False):
        self.csv_path = csv_path
        self.binary = binary
        self.per_subject_norm = per_subject_norm
        self.per_subject_norm_stage = per_subject_norm_stage
        self.calibration_windows = int(calibration_windows)
        self.eeg_log = eeg_log
        self.fnirs_log = fnirs_log
        self.drop_hbr = drop_hbr
        self._load()

    # ── loading ────────────────────────────────────────────────────────
    def _load(self):
        df = pd.read_csv(self.csv_path)
        cols = list(df.columns)

        eeg = sorted([c for c in cols if c.startswith("eeg_")], key=natural_key)
        hbo = sorted([c for c in cols if c.startswith("hbo_")], key=natural_key)
        hbr = sorted([c for c in cols if c.startswith("hbr_")], key=natural_key)
        if self.drop_hbr:
            hbr = []
        self.schema = FeatureSchema(eeg, hbo, hbr)

        y_raw = df["Cognitive_Load"].to_numpy(np.int64)
        keep = np.ones(len(df), bool)
        if self.binary:
            keep = y_raw != 1  # drop Medium (2-back)
        df = df.loc[keep].reset_index(drop=True)
        y_raw = y_raw[keep]

        if self.binary:
            y = np.where(y_raw == 2, 1, 0).astype(np.int64)
            self.class_names = ["Low", "High"]
            self.class_semantics = ["0-back", "3-back"]
        else:
            y = y_raw
            self.class_names = ["Low", "Medium", "High"]
            self.class_semantics = ["0-back", "2-back", "3-back"]

        self.y = y
        self.n_classes = len(self.class_names)
        self.subjects = df["subject_id"].to_numpy(np.int64)
        self.trials = df["trial_id"].astype(str).to_numpy()
        self.window_ids = (
            df["window_id"].to_numpy(np.int64)
            if "window_id" in df.columns
            else np.zeros(len(df), np.int64)
        )
        self.window_start = (
            df["window_start_sec"].to_numpy(np.float64)
            if "window_start_sec" in df.columns
            else np.zeros(len(df))
        )
        self.window_sec = (
            float(df["window_sec"].iloc[0]) if "window_sec" in df.columns else float("nan")
        )

        eeg_raw = df[self.schema.eeg_cols].to_numpy(np.float32)
        fn_raw = df[self.schema.fnirs_cols].to_numpy(np.float32)

        # 1. NaN / Inf handling
        eeg_raw = np.nan_to_num(eeg_raw, nan=0.0, posinf=0.0, neginf=0.0)
        fn_raw = np.nan_to_num(fn_raw, nan=0.0, posinf=0.0, neginf=0.0)

        self.eeg, self.fnirs = self._preprocess(eeg_raw, fn_raw)
        self.n_eeg = self.eeg.shape[1]
        self.n_fnirs = self.fnirs.shape[1]
        self.unique_subjects = np.unique(self.subjects)

        # Row fingerprints, used by the duplicate-leakage audit.
        self._row_hashes = None

    # ── preprocessing steps ────────────────────────────────────────────
    def _log_eeg(self, x):
        # EEG band powers are non-negative and right-skewed -> clip + log1p.
        return np.log1p(np.clip(x, 0.0, None))

    def _log_fnirs(self, x):
        # HbO / HbR go negative -> signed log1p.
        return np.sign(x) * np.log1p(np.abs(x))

    def _subject_norm_mask(self, subject):
        """Rows of `subject` whose statistics are allowed to be used."""
        rows = np.flatnonzero(self.subjects == subject)
        if self.per_subject_norm != "calibration":
            return rows
        # Deployment-realistic: only the first K windows of the session, in
        # acquisition order, act as a calibration block.
        order = np.lexsort((self.window_ids[rows], self.trials[rows]))
        return rows[order][: self.calibration_windows]

    def _apply_subject_norm(self, eeg, fn):
        eps = 1e-8
        for s in np.unique(self.subjects):
            rows = np.flatnonzero(self.subjects == s)
            stat_rows = self._subject_norm_mask(s)
            for block in (eeg, fn):
                ref = block[stat_rows]
                mu = ref.mean(axis=0)
                sd = ref.std(axis=0, ddof=1) if len(ref) > 1 else np.ones(block.shape[1])
                sd = np.nan_to_num(sd, nan=eps, posinf=eps, neginf=eps)
                sd[sd < eps] = eps
                block[rows] = (block[rows] - mu) / sd
        return eeg, fn

    def _preprocess(self, eeg, fn):
        eeg = eeg.copy()
        fn = fn.copy()
        do_norm = self.per_subject_norm in ("transductive", "calibration")

        if do_norm and self.per_subject_norm_stage == "before_log":
            eeg, fn = self._apply_subject_norm(eeg, fn)

        if self.eeg_log:
            eeg = self._log_eeg(eeg) if not do_norm or \
                self.per_subject_norm_stage == "after_log" else \
                np.sign(eeg) * np.log1p(np.abs(eeg))
        if self.fnirs_log:
            fn = self._log_fnirs(fn)

        if do_norm and self.per_subject_norm_stage == "after_log":
            eeg, fn = self._apply_subject_norm(eeg, fn)

        return eeg.astype(np.float32), fn.astype(np.float32)

    # ── helpers ────────────────────────────────────────────────────────
    def row_hashes(self):
        if self._row_hashes is None:
            joined = np.hstack([self.eeg, self.fnirs])
            self._row_hashes = np.array(
                [hashlib.sha1(r.tobytes()).hexdigest() for r in joined]
            )
        return self._row_hashes

    def info(self, extra=None):
        counts = {
            self.class_names[c]: int((self.y == c).sum()) for c in range(self.n_classes)
        }
        per_subject = (
            pd.DataFrame({"subject": self.subjects, "y": self.y})
            .groupby("subject")["y"]
            .agg(["size", "mean"])
            .rename(columns={"size": "n_windows", "mean": "positive_rate"})
        )
        d = {
            "csv_path": self.csv_path,
            "csv_basename": os.path.basename(os.path.realpath(self.csv_path)),
            "csv_realpath": os.path.realpath(self.csv_path),
            "csv_bytes": os.path.getsize(os.path.realpath(self.csv_path)),
            "csv_sha256_first_16mb": _sha256_file(
                os.path.realpath(self.csv_path), limit_bytes=16 << 20
            ),
            "n_samples": int(len(self.y)),
            "n_subjects": int(len(self.unique_subjects)),
            "subjects": self.unique_subjects.tolist(),
            "n_trials": int(len(np.unique(self.trials))),
            "n_windows_per_subject": {
                str(k): int(v) for k, v in per_subject["n_windows"].items()
            },
            "window_sec": self.window_sec,
            "classification_mode": "binary" if self.binary else "3class",
            "class_names": self.class_names,
            "class_semantics": self.class_semantics,
            "class_distribution": counts,
            "raw_class_names": RAW_CLASS_NAMES,
            "preprocessing": {
                "nan_to_num": True,
                "per_subject_norm": self.per_subject_norm,
                "per_subject_norm_stage": self.per_subject_norm_stage,
                "calibration_windows": (
                    self.calibration_windows
                    if self.per_subject_norm == "calibration"
                    else None
                ),
                "eeg_log": self.eeg_log,
                "fnirs_log": self.fnirs_log,
                "drop_hbr": self.drop_hbr,
                "scaler_eeg": "RobustScaler (fit on train fold only)",
                "scaler_fnirs": "StandardScaler (fit on train fold only)",
                "order": "nan_to_num -> per_subject_norm -> log -> scaler",
            },
            "feature_schema": self.schema.to_dict(),
        }
        if extra:
            d.update(extra)
        return d


# ── fold generation ────────────────────────────────────────────────────

def split_dev_lockbox(subjects, n_lockbox, seed):
    """Partition subjects into a development pool and a lockbox.

    The lockbox is never used for any experiment comparison, scheduler choice,
    hyperparameter choice or checkpoint choice.  It is evaluated once, at the
    end, for the single configuration selected on the development pool.
    """
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(subjects))
    perm = rng.permutation(len(subs))
    lock = np.sort(subs[perm[:n_lockbox]])
    dev = np.sort(subs[perm[n_lockbox:]])
    return dev, lock


def loso_folds(ds, pool_subjects, val_fraction=0.2, seed=42):
    """Leave-one-subject-out over `pool_subjects`.

    Yields dicts with train/val/test row indices.  The held-out subject
    contributes rows to `test` only — never to train, val, or any statistic
    fitted inside the fold.
    """
    pool_subjects = np.asarray(sorted(pool_subjects))
    pool_mask = np.isin(ds.subjects, pool_subjects)
    pool_idx = np.flatnonzero(pool_mask)

    for fold, held in enumerate(pool_subjects):
        test_idx = pool_idx[ds.subjects[pool_idx] == held]
        fit_idx = pool_idx[ds.subjects[pool_idx] != held]

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed + fold
        )
        tr_rel, va_rel = next(
            splitter.split(fit_idx, ds.y[fit_idx], ds.subjects[fit_idx])
        )
        yield {
            "fold": fold,
            "held_out_subject": int(held),
            "train_idx": fit_idx[tr_rel],
            "val_idx": fit_idx[va_rel],
            "test_idx": test_idx,
        }


def holdout_fold(ds, train_subjects, test_subjects, val_fraction=0.2, seed=42):
    """One fold: train on `train_subjects`, evaluate on `test_subjects`.

    Used for the lockbox evaluation, and yields one entry per lockbox subject
    so per-subject metrics stay available.
    """
    train_pool = np.flatnonzero(np.isin(ds.subjects, train_subjects))
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_rel, va_rel = next(
        splitter.split(train_pool, ds.y[train_pool], ds.subjects[train_pool])
    )
    return {
        "fold": 0,
        "held_out_subject": None,
        "train_idx": train_pool[tr_rel],
        "val_idx": train_pool[va_rel],
        "test_idx": np.flatnonzero(np.isin(ds.subjects, test_subjects)),
    }


def scale_fold(ds, fold):
    """Fit scalers on the fold's TRAIN rows only, transform all three splits."""
    tr, va, te = fold["train_idx"], fold["val_idx"], fold["test_idx"]
    eeg_scaler = RobustScaler()
    fn_scaler = StandardScaler()

    eeg_tr = eeg_scaler.fit_transform(ds.eeg[tr]).astype(np.float32)
    fn_tr = fn_scaler.fit_transform(ds.fnirs[tr]).astype(np.float32)

    def tx(idx):
        return (
            eeg_scaler.transform(ds.eeg[idx]).astype(np.float32),
            fn_scaler.transform(ds.fnirs[idx]).astype(np.float32),
        )

    eeg_va, fn_va = tx(va)
    eeg_te, fn_te = tx(te)
    return {
        "train": (eeg_tr, fn_tr, ds.y[tr]),
        "val": (eeg_va, fn_va, ds.y[va]),
        "test": (eeg_te, fn_te, ds.y[te]),
        "scalers": (eeg_scaler, fn_scaler),
        "scaler_fit_rows": tr,
    }
