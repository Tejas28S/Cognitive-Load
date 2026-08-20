"""
test_leakage.py — negative tests for the leakage audit.

An audit that never fires proves nothing. Each test here deliberately INJECTS a
specific kind of leakage into an otherwise clean fold and asserts that the audit
catches that exact check. The final test asserts a clean fold passes everything,
so the suite fails if the audit becomes vacuously true.

    .venv/bin/python -m research.test_leakage
"""

from __future__ import annotations

import copy
import sys

import numpy as np

from .datamod import Dataset, loso_folds, scale_fold, split_dev_lockbox
from .leakage import LeakageError, audit_dataset, audit_fold

CSV = "data/tuberlin_sliding.csv"


def _clean_fold(ds):
    dev, _ = split_dev_lockbox(ds.unique_subjects, n_lockbox=8, seed=42)
    return next(iter(loso_folds(ds, dev, seed=42)))


def _expect_failure(ds, fold, scaled, check_name):
    """The audit must raise, and the message must name this specific check."""
    try:
        audit_fold(ds, fold, scaled, strict=True)
    except LeakageError as exc:
        if check_name not in str(exc):
            raise AssertionError(
                f"audit raised but did not name '{check_name}': {exc}"
            ) from None
        return str(exc)
    raise AssertionError(f"audit did NOT catch injected leakage for '{check_name}'")


def test_clean_fold_passes(ds):
    fold = _clean_fold(ds)
    scaled = scale_fold(ds, fold)
    result = audit_fold(ds, fold, scaled, strict=True)
    assert result["passed"], "a clean fold must pass every check"
    names = {c["check"] for c in result["checks"]}
    assert len(names) == 9, f"expected 9 fold checks, got {len(names)}: {names}"
    return f"clean fold passed all {len(names)} checks"


def test_subject_leak_detected(ds):
    """Put one of the held-out subject's windows into the training set."""
    fold = copy.deepcopy(_clean_fold(ds))
    stolen = fold["test_idx"][:5]
    fold["train_idx"] = np.concatenate([fold["train_idx"], stolen])
    scaled = scale_fold(ds, fold)
    return _expect_failure(ds, fold, scaled, "subject_disjoint")


def test_window_leak_detected(ds):
    """Move a held-out window into validation — the sliding-window failure mode."""
    fold = copy.deepcopy(_clean_fold(ds))
    fold["val_idx"] = np.concatenate([fold["val_idx"], fold["test_idx"][:3]])
    scaled = scale_fold(ds, fold)
    return _expect_failure(ds, fold, scaled, "row_index_disjoint")


def test_scaler_leak_detected(ds):
    """Fit the scalers on train + test, the classic preprocessing leak."""
    fold = _clean_fold(ds)
    scaled = scale_fold(ds, fold)
    leaked = dict(scaled)
    leaked["scaler_fit_rows"] = np.concatenate(
        [fold["train_idx"], fold["test_idx"]]
    )
    return _expect_failure(ds, fold, leaked, "scaler_fit_on_train_only")


def test_duplicate_row_leak_detected(ds):
    """Copy a training row's features onto a test row: an exact duplicate."""
    fold = copy.deepcopy(_clean_fold(ds))
    ds2 = copy.copy(ds)
    ds2.eeg = ds.eeg.copy()
    ds2.fnirs = ds.fnirs.copy()
    ds2._row_hashes = None
    src, dst = fold["train_idx"][0], fold["test_idx"][0]
    ds2.eeg[dst] = ds2.eeg[src]
    ds2.fnirs[dst] = ds2.fnirs[src]
    scaled = scale_fold(ds2, fold)
    return _expect_failure(ds2, fold, scaled, "duplicate_feature_rows")


def test_nan_detected(ds):
    """A NaN surviving into a split must be caught, not silently trained on."""
    fold = _clean_fold(ds)
    scaled = scale_fold(ds, fold)
    broken = dict(scaled)
    eeg, fn, y = scaled["test"]
    eeg = eeg.copy()
    eeg[0, 0] = np.nan
    broken["test"] = (eeg, fn, y)
    return _expect_failure(ds, fold, broken, "no_nan_or_inf")


def test_heldout_never_fitted(ds):
    """The held-out subject must be absent from the fitted statistics."""
    fold = copy.deepcopy(_clean_fold(ds))
    held = fold["held_out_subject"]
    extra = np.flatnonzero(ds.subjects == held)[:4]
    fold["val_idx"] = np.concatenate([fold["val_idx"], extra])
    scaled = scale_fold(ds, fold)
    return _expect_failure(ds, fold, scaled, "row_index_disjoint")


def test_subject_local_normalization(ds):
    """Per-subject normalization must be reproducible from one subject alone."""
    result = audit_dataset(ds, probe_subjects=ds.unique_subjects[:3], strict=True)
    assert result["passed"]
    detail = [c["detail"] for c in result["checks"]
              if c["check"] == "per_subject_norm_is_subject_local"]
    return detail[0] if detail else "subject-local check ran"


def main():
    print(f"Loading {CSV} ...")
    ds = Dataset(CSV)
    print(f"  {len(ds.y)} windows, {len(ds.unique_subjects)} subjects, "
          f"classes {ds.class_names}\n")

    tests = [
        ("clean fold passes every check", test_clean_fold_passes),
        ("subject leakage is caught", test_subject_leak_detected),
        ("window leakage is caught", test_window_leak_detected),
        ("scaler-fit leakage is caught", test_scaler_leak_detected),
        ("duplicate-row leakage is caught", test_duplicate_row_leak_detected),
        ("non-finite values are caught", test_nan_detected),
        ("held-out subject in val is caught", test_heldout_never_fitted),
        ("per-subject norm is subject-local", test_subject_local_normalization),
    ]

    failures = 0
    for name, fn in tests:
        try:
            detail = fn(ds)
            print(f"  PASS  {name}")
            if detail:
                print(f"        {str(detail)[:150]}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
