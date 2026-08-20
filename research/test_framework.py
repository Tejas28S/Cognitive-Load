"""
test_framework.py — correctness tests for the protocol machinery.

These assert the properties the evaluation depends on: that the
development/lockbox partition is real, that LOSO covers each subject exactly
once, that lockbox folds never train on lockbox subjects, that scalers are
fitted on training rows only, that modality dropout is training-only and never
removes both modalities, and that early stopping restores the weights that
actually achieved the best validation score.

    .venv/bin/python -m research.test_framework
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from .archs import build_model
from .datamod import (
    Dataset, holdout_fold, loso_folds, scale_fold, split_dev_lockbox,
)
from .train import evaluate, make_loader, train_fold

CSV = "data/tuberlin_sliding.csv"


def test_partition(ds):
    dev, lock = split_dev_lockbox(ds.unique_subjects, 8, 42)
    assert set(dev).isdisjoint(set(lock)), "dev and lockbox overlap"
    assert set(dev) | set(lock) == set(ds.unique_subjects.tolist()), \
        "partition does not cover every subject"
    return f"dev={len(dev)} lockbox={len(lock)}, disjoint and complete"


def test_loso_coverage(ds):
    dev, _ = split_dev_lockbox(ds.unique_subjects, 8, 42)
    folds = list(loso_folds(ds, dev, seed=42))
    held = [f["held_out_subject"] for f in folds]
    assert sorted(held) == sorted(dev.tolist()), "LOSO does not cover the pool"
    assert len(set(held)) == len(held), "a subject was held out twice"
    return f"{len(folds)} folds, each dev subject held out exactly once"


def test_lockbox_never_trained_on(ds):
    dev, lock = split_dev_lockbox(ds.unique_subjects, 8, 42)
    for s in lock:
        g = holdout_fold(ds, dev, [s], seed=42)
        fitted = set(ds.subjects[g["train_idx"]].tolist()) | set(
            ds.subjects[g["val_idx"]].tolist()
        )
        assert fitted.isdisjoint(set(lock.tolist())), \
            f"a lockbox subject appears in training for held-out subject {s}"
    return f"all {len(lock)} lockbox folds train only on development subjects"


def test_scaler_fit_on_train(ds):
    dev, _ = split_dev_lockbox(ds.unique_subjects, 8, 42)
    fold = next(iter(loso_folds(ds, dev, seed=42)))
    sc = scale_fold(ds, fold)
    mean = float(sc["train"][1].mean())
    assert abs(mean) < 1e-4, f"train fNIRS mean should be ~0, got {mean}"
    assert set(map(int, sc["scaler_fit_rows"])) == set(map(int, fold["train_idx"]))
    return f"train fNIRS mean = {mean:.2e}; fit rows == train rows"


def test_modality_dropout(ds):
    m = build_model(
        {"modality_dropout_eeg": 0.5, "modality_dropout_fnirs": 0.5}, ds.schema, 2
    )
    e = torch.randn(2000, ds.n_eeg)
    f = torch.randn(2000, ds.n_fnirs)
    emb_e, emb_f = m.eeg_encoder(e), m.fnirs_encoder(f)

    m.train()
    de, df = m._modality_dropout(emb_e.clone(), emb_f.clone())
    zero_e = de.abs().sum(1) == 0
    zero_f = df.abs().sum(1) == 0
    assert not (zero_e & zero_f).any(), "both modalities dropped for some sample"
    assert zero_e.any() and zero_f.any(), "modality dropout never fired"

    m.eval()
    de2, df2 = m._modality_dropout(emb_e.clone(), emb_f.clone())
    assert torch.equal(de2, emb_e) and torch.equal(df2, emb_f), \
        "modality dropout is active outside training"
    return (f"train: eeg-dropped {zero_e.float().mean():.2f}, "
            f"fnirs-dropped {zero_f.float().mean():.2f}, both 0.00; eval: disabled")


def test_early_stopping_restores_best(ds):
    dev, _ = split_dev_lockbox(ds.unique_subjects, 8, 42)
    fold = next(iter(loso_folds(ds, dev, seed=42)))
    sc = scale_fold(ds, fold)
    res = train_fold(
        {"max_epochs": 25, "early_stop_patience": 4, "batch_size": 64},
        ds.schema, 2, sc, seed=1, device="cpu", log=lambda *_: None,
    )
    hist = res["history"]
    best = res["best_epoch"]
    assert hist["val_f1_macro"][best - 1] == max(hist["val_f1_macro"]), \
        "best_epoch is not the argmax of validation macro F1"

    va = make_loader(*sc["val"], None, 512, False)
    got = evaluate(res["model"], va, torch.device("cpu"), None, 2)["f1_macro"]
    assert abs(got - res["best_val_f1_macro"]) < 1e-6, \
        f"restored weights score {got}, expected {res['best_val_f1_macro']}"
    return (f"best epoch {best}/{res['epochs_run']}, restored weights reproduce "
            f"val macro F1 {got:.6f}")


def main():
    print(f"Loading {CSV} ...")
    ds = Dataset(CSV)
    print(f"  {len(ds.y)} windows, {len(ds.unique_subjects)} subjects\n")

    tests = [
        ("development/lockbox partition is real", test_partition),
        ("LOSO covers each subject exactly once", test_loso_coverage),
        ("lockbox subjects are never trained on", test_lockbox_never_trained_on),
        ("scalers are fitted on training rows only", test_scaler_fit_on_train),
        ("modality dropout is training-only, never both", test_modality_dropout),
        ("early stopping restores the best weights", test_early_stopping_restores_best),
    ]
    failures = 0
    for name, fn in tests:
        try:
            print(f"  PASS  {name}\n        {fn(ds)}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
