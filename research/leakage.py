"""
leakage.py — automated leakage assertions.

Every check returns a record; `audit_fold` and `audit_run` raise
LeakageError on any failure so a run cannot silently produce a flattered
score.  Results are written to the run directory as leakage_audit.json.

Checks implemented (section 8 of the brief):
  * subject overlap between train / val / test
  * trial overlap between train / val / test
  * duplicate windows (same subject+trial+window_id) across splits
  * duplicate feature rows (byte-identical) across splits
  * index overlap across splits
  * scaler fitted only on train rows
  * the held-out subject contributes no row to any fitted statistic
  * per-subject normalization is subject-local (proved numerically)
  * no NaN / Inf survives preprocessing
  * label sanity (no remapping surprises, classes present)
"""

from __future__ import annotations

import numpy as np


class LeakageError(AssertionError):
    """Raised when a fold or run fails a leakage check."""


def _ok(name, passed, detail=""):
    return {"check": name, "passed": bool(passed), "detail": detail}


def check_index_disjoint(fold):
    tr = set(map(int, fold["train_idx"]))
    va = set(map(int, fold["val_idx"]))
    te = set(map(int, fold["test_idx"]))
    inter = [
        ("train&val", tr & va),
        ("train&test", tr & te),
        ("val&test", va & te),
    ]
    bad = {k: len(v) for k, v in inter if v}
    return _ok("row_index_disjoint", not bad, f"overlaps={bad}" if bad else "no shared rows")


def check_subject_disjoint(ds, fold):
    tr = set(ds.subjects[fold["train_idx"]].tolist())
    va = set(ds.subjects[fold["val_idx"]].tolist())
    te = set(ds.subjects[fold["test_idx"]].tolist())
    bad = {}
    if tr & te:
        bad["train&test"] = sorted(tr & te)
    if va & te:
        bad["val&test"] = sorted(va & te)
    if tr & va:
        bad["train&val"] = sorted(tr & va)
    return _ok(
        "subject_disjoint",
        not bad,
        f"shared subjects={bad}" if bad else
        f"train={len(tr)} val={len(va)} test={len(te)} subjects, all disjoint",
    )


def check_trial_disjoint(ds, fold):
    tr = set(ds.trials[fold["train_idx"]].tolist())
    va = set(ds.trials[fold["val_idx"]].tolist())
    te = set(ds.trials[fold["test_idx"]].tolist())
    bad = {}
    for name, s in (("train&test", tr & te), ("val&test", va & te), ("train&val", tr & va)):
        if s:
            bad[name] = len(s)
    return _ok(
        "trial_disjoint",
        not bad,
        f"shared trials={bad}" if bad else f"{len(tr)}/{len(va)}/{len(te)} trials, disjoint",
    )


def check_window_disjoint(ds, fold):
    """Overlapping sliding windows must not straddle a split boundary."""
    def keys(idx):
        return set(
            zip(
                ds.subjects[idx].tolist(),
                ds.trials[idx].tolist(),
                ds.window_ids[idx].tolist(),
            )
        )

    tr, va, te = keys(fold["train_idx"]), keys(fold["val_idx"]), keys(fold["test_idx"])
    bad = {}
    for name, s in (("train&test", tr & te), ("val&test", va & te), ("train&val", tr & va)):
        if s:
            bad[name] = len(s)
    return _ok(
        "window_identity_disjoint",
        not bad,
        f"shared windows={bad}" if bad else "no (subject,trial,window) key on both sides",
    )


def check_duplicate_rows(ds, fold):
    """Byte-identical feature rows appearing on both sides of the split."""
    h = ds.row_hashes()
    tr = set(h[fold["train_idx"]].tolist())
    va = set(h[fold["val_idx"]].tolist())
    te = set(h[fold["test_idx"]].tolist())
    bad = {}
    for name, s in (("train&test", tr & te), ("val&test", va & te), ("train&val", tr & va)):
        if s:
            bad[name] = len(s)
    return _ok(
        "duplicate_feature_rows",
        not bad,
        f"identical rows across splits={bad}" if bad else "no identical rows across splits",
    )


def check_scaler_fit_rows(fold, scaled):
    fit_rows = set(map(int, scaled["scaler_fit_rows"]))
    train_rows = set(map(int, fold["train_idx"]))
    extra = fit_rows - train_rows
    return _ok(
        "scaler_fit_on_train_only",
        not extra,
        f"{len(extra)} non-train rows used to fit scalers" if extra else
        f"scalers fitted on {len(fit_rows)} train rows only",
    )


def check_heldout_excluded(ds, fold):
    held = fold.get("held_out_subject")
    if held is None:
        held_set = set(ds.subjects[fold["test_idx"]].tolist())
    else:
        held_set = {int(held)}
    fitted = set(ds.subjects[fold["train_idx"]].tolist()) | set(
        ds.subjects[fold["val_idx"]].tolist()
    )
    bad = held_set & fitted
    return _ok(
        "heldout_subject_never_fitted",
        not bad,
        f"held-out subject(s) {sorted(bad)} appear in train/val" if bad else
        f"held-out {sorted(held_set)} absent from train and val",
    )


def check_no_nan(scaled):
    bad = []
    for split in ("train", "val", "test"):
        eeg, fn, _ = scaled[split]
        if not np.isfinite(eeg).all():
            bad.append(f"{split}.eeg")
        if not np.isfinite(fn).all():
            bad.append(f"{split}.fnirs")
    return _ok("no_nan_or_inf", not bad, f"non-finite in {bad}" if bad else "all finite")


def check_labels(ds, fold):
    problems = []
    for split in ("train_idx", "val_idx", "test_idx"):
        y = ds.y[fold[split]]
        if len(y) == 0:
            problems.append(f"{split} empty")
            continue
        if y.min() < 0 or y.max() >= ds.n_classes:
            problems.append(f"{split} label out of range")
    if len(np.unique(ds.y[fold["train_idx"]])) < 2:
        problems.append("train fold has a single class")
    return _ok("labels_valid", not problems, "; ".join(problems) if problems else
               f"labels in [0,{ds.n_classes - 1}], train has both classes")


def assert_subject_local(ds, subject, atol=1e-5):
    """Prove per-subject normalization used only that subject's own rows.

    Rebuilds the preprocessing for one subject from that subject's raw rows in
    isolation and checks the result matches the full-dataset computation.  If
    any cross-subject statistic had leaked in, the isolated recomputation would
    differ.
    """
    import pandas as pd

    if ds.per_subject_norm not in ("transductive", "calibration"):
        return _ok("per_subject_norm_is_subject_local", True, "per-subject norm disabled")

    df = pd.read_csv(ds.csv_path)
    y_raw = df["Cognitive_Load"].to_numpy(np.int64)
    if ds.binary:
        df = df.loc[y_raw != 1].reset_index(drop=True)
    rows = np.flatnonzero(df["subject_id"].to_numpy(np.int64) == subject)

    eeg = np.nan_to_num(
        df.loc[rows, ds.schema.eeg_cols].to_numpy(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    fn = np.nan_to_num(
        df.loc[rows, ds.schema.fnirs_cols].to_numpy(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    # Replay the same pipeline on this subject alone.
    sub = _SingleSubjectView(ds, subject, rows)
    eeg_iso, fn_iso = sub.preprocess(eeg, fn)

    mine = np.flatnonzero(ds.subjects == subject)
    d_eeg = float(np.max(np.abs(eeg_iso - ds.eeg[mine])))
    d_fn = float(np.max(np.abs(fn_iso - ds.fnirs[mine])))
    passed = d_eeg <= atol and d_fn <= atol
    return _ok(
        "per_subject_norm_is_subject_local",
        passed,
        f"subject {subject} recomputed in isolation: max|delta| eeg={d_eeg:.2e} "
        f"fnirs={d_fn:.2e} (tol {atol:.0e})",
    )


class _SingleSubjectView:
    """Replays Dataset._preprocess for one subject using only its own rows."""

    def __init__(self, ds, subject, rows):
        self.ds = ds
        self.subject = subject
        self.rows = rows

    def preprocess(self, eeg, fn):
        ds = self.ds
        eeg = eeg.copy()
        fn = fn.copy()
        do_norm = ds.per_subject_norm in ("transductive", "calibration")

        def norm(block):
            eps = 1e-8
            if ds.per_subject_norm == "calibration":
                mine = np.flatnonzero(ds.subjects == self.subject)
                order = np.lexsort((ds.window_ids[mine], ds.trials[mine]))
                local = order[: ds.calibration_windows]
            else:
                local = np.arange(len(block))
            ref = block[local]
            mu = ref.mean(axis=0)
            sd = ref.std(axis=0, ddof=1) if len(ref) > 1 else np.ones(block.shape[1])
            sd = np.nan_to_num(sd, nan=eps, posinf=eps, neginf=eps)
            sd[sd < eps] = eps
            return (block - mu) / sd

        if do_norm and ds.per_subject_norm_stage == "before_log":
            eeg, fn = norm(eeg), norm(fn)
        if ds.eeg_log:
            eeg = (
                np.log1p(np.clip(eeg, 0.0, None))
                if not do_norm or ds.per_subject_norm_stage == "after_log"
                else np.sign(eeg) * np.log1p(np.abs(eeg))
            )
        if ds.fnirs_log:
            fn = np.sign(fn) * np.log1p(np.abs(fn))
        if do_norm and ds.per_subject_norm_stage == "after_log":
            eeg, fn = norm(eeg), norm(fn)
        return eeg.astype(np.float32), fn.astype(np.float32)


def audit_fold(ds, fold, scaled, strict=True):
    checks = [
        check_index_disjoint(fold),
        check_subject_disjoint(ds, fold),
        check_trial_disjoint(ds, fold),
        check_window_disjoint(ds, fold),
        check_duplicate_rows(ds, fold),
        check_scaler_fit_rows(fold, scaled),
        check_heldout_excluded(ds, fold),
        check_no_nan(scaled),
        check_labels(ds, fold),
    ]
    failed = [c for c in checks if not c["passed"]]
    if failed and strict:
        raise LeakageError(
            "Leakage detected in fold "
            f"{fold.get('fold')} (held-out subject {fold.get('held_out_subject')}): "
            + "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
        )
    return {
        "fold": fold.get("fold"),
        "held_out_subject": fold.get("held_out_subject"),
        "checks": checks,
        "passed": not failed,
    }


def audit_dataset(ds, probe_subjects=None, strict=True):
    """Dataset-level checks that do not depend on a fold."""
    probe_subjects = probe_subjects if probe_subjects is not None else ds.unique_subjects[:3]
    checks = [assert_subject_local(ds, int(s)) for s in probe_subjects]

    h = ds.row_hashes()
    _, counts = np.unique(h, return_counts=True)
    n_dup = int((counts > 1).sum())
    checks.append(
        _ok(
            "no_duplicate_rows_in_dataset",
            n_dup == 0,
            f"{n_dup} feature vectors appear more than once" if n_dup else
            f"all {len(h)} feature rows unique",
        )
    )
    checks.append(
        _ok(
            "features_finite",
            bool(np.isfinite(ds.eeg).all() and np.isfinite(ds.fnirs).all()),
            "preprocessed features are finite",
        )
    )
    failed = [c for c in checks if not c["passed"]]
    if failed and strict:
        raise LeakageError(
            "Dataset-level leakage check failed: "
            + "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
        )
    return {"checks": checks, "passed": not failed}
