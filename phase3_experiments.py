"""
Phase 3 accuracy experiments.

Examples:
    python phase3_experiments.py priority1
    python phase3_experiments.py cross-attention
    python phase3_experiments.py window-commands
    python phase3_experiments.py loso --max-folds 10 --epochs 80 --early-stop 10
"""

import argparse
import copy
import os
import random
import re
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader

from config import Config
from data_loader import CognitiveLoadDataLoader, CognitiveLoadDataset, configured_class_weights
from evaluate import evaluate_model, predict
from models.architectures import build_model
from trainer import train_model


BASELINE_MACRO_F1 = 0.4678
MIN_CLASS_RECALL = 0.20
METRIC_EPSILON = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_config(overrides=None):
    cfg = copy.copy(Config())
    for key, value in (overrides or {}).items():
        setattr(cfg, key, value)
    return cfg


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def csv_feature_overrides(csv_path):
    df = pd.read_csv(csv_path, nrows=0)
    eeg_cols = sorted([c for c in df.columns if c.startswith("eeg_")], key=natural_key)
    hbo_cols = sorted([c for c in df.columns if c.startswith("hbo_")], key=natural_key)
    hbr_cols = sorted([c for c in df.columns if c.startswith("hbr_")], key=natural_key)
    overrides = {
        "MERGED_CSV": csv_path,
        "EEG_COLUMNS": eeg_cols,
        "MODALITY2_COLUMNS": sorted(hbo_cols + hbr_cols, key=natural_key),
        "EEG_FEATURE_PREFIX": "eeg_",
        "FNIRS_FEATURE_PREFIX": "hbo_",
    }
    if "trial_id" in df.columns:
        overrides["SPLIT_GROUP_COLUMNS"] = ["subject_id", "trial_id"]
    return overrides


def metric_summary(result):
    recalls = result["per_class_recall"]
    return (
        f"Acc={result['acc']:.4f}, Macro F1={result['f1_macro']:.4f}, "
        f"AUC={result['auc']:.4f}, recall=["
        + ", ".join(f"{r:.4f}" for r in recalls)
        + "]"
    )


def passes_guardrail(result):
    return (
        result["f1_macro"] + METRIC_EPSILON >= BASELINE_MACRO_F1
        and float(np.min(result["per_class_recall"])) + METRIC_EPSILON >= MIN_CLASS_RECALL
    )


def append_results(section, entries):
    lines = [
        "",
        f"## Phase 3 - {section}",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for entry in entries:
        verdict = "kept" if passes_guardrail(entry["result"]) else "reverted"
        sample_count = entry.get("sample_count")
        sample_line = [f"- Samples: {sample_count:,}"] if sample_count is not None else []
        lines.extend([
            f"### {entry['name']}",
            f"- Change: {entry['change']}",
            *sample_line,
            f"- Result: {metric_summary(entry['result'])}",
            f"- Verdict: {verdict} - compared against baseline Macro F1={BASELINE_MACRO_F1:.4f} "
            f"and per-class recall floor={MIN_CLASS_RECALL:.0%}.",
            "",
        ])
    with open("RESULTS.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run_hybrid(name, change, overrides=None, model_type="hybrid_cnn_lstm"):
    cfg = clone_config(overrides)
    set_seed(cfg.RANDOM_SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, f"phase3_{safe_name}.pt")

    print("\n" + "=" * 72)
    print(f"Phase 3 experiment: {name}")
    print(f"Change: {change}")
    print("=" * 72)

    dl = CognitiveLoadDataLoader(cfg)
    train_loader, val_loader, test_loader, meta = dl.prepare()
    model = build_model(
        model_type,
        meta["n_eeg_features"],
        meta["n_fnirs_features"],
        cfg,
    )
    model, _ = train_model(model, train_loader, val_loader, meta, cfg)
    result = evaluate_model(model, test_loader, cfg, model_name=name)
    print(f"Guardrail: {'PASS' if passes_guardrail(result) else 'FAIL'}")
    return {"name": name, "change": change, "result": result}


def subject_normalization_fixed_check():
    entries = [
        run_hybrid(
            "Per-subject normalization before log",
            "Z-score EEG and fNIRS within each subject before modality log transforms.",
            {
                "PER_SUBJECT_NORMALIZE": True,
                "PER_SUBJECT_NORMALIZE_STAGE": "before_log",
            },
        ),
        run_hybrid(
            "Per-subject normalization after log",
            "Apply modality log transforms first, then z-score EEG and fNIRS within each subject.",
            {
                "PER_SUBJECT_NORMALIZE": True,
                "PER_SUBJECT_NORMALIZE_STAGE": "after_log",
            },
        ),
    ]
    append_results("Subject Normalization Fixed-Split Stage Check", entries)
    return entries


def priority1():
    entries = [
        run_hybrid(
            "EarlyStop15 Weighted CE",
            "Set EARLY_STOP=15 while keeping weighted CE and baseline scheduler settings.",
            {"EARLY_STOP": 15, "LOSS_FN": "weighted_ce"},
        ),
        run_hybrid(
            "Focal Loss Gamma2",
            "Set LOSS_FN='focal' and FOCAL_GAMMA=2.0 with EARLY_STOP=15.",
            {"EARLY_STOP": 15, "LOSS_FN": "focal", "FOCAL_GAMMA": 2.0},
        ),
        run_hybrid(
            "LR Patience 3",
            "Set LR_PATIENCE=3 with weighted CE and EARLY_STOP=15.",
            {"EARLY_STOP": 15, "LOSS_FN": "weighted_ce", "LR_PATIENCE": 3},
        ),
        run_hybrid(
            "LR Patience 8",
            "Set LR_PATIENCE=8 with weighted CE and EARLY_STOP=15.",
            {"EARLY_STOP": 15, "LOSS_FN": "weighted_ce", "LR_PATIENCE": 8},
        ),
    ]
    append_results("Priority 1: Cheap Tuning Wins", entries)


def cross_attention():
    entries = [
        run_hybrid(
            "Attention Fusion Reference",
            "Reference run using existing attention/gated fusion path.",
            {"FUSION_TYPE": "attention"},
        ),
        run_hybrid(
            "Cross-Attention Fusion",
            "Use the new CrossAttentionHybrid fusion variant with identical data split/config.",
            {"FUSION_TYPE": "cross_attention"},
            model_type="cross_attention_hybrid",
        ),
    ]
    append_results("Priority 2: Cross-Attention Fusion", entries)


def window_commands():
    commands = [
        (
            "8s window / 4s stride",
            [
                sys.executable,
                "sliding_window_extractor.py",
                "--nirs-dir",
                "data/NIRS_01-26_MATLAB",
                "--eeg-dir",
                "data/EEG_01-26_MATLAB",
                "--output",
                "data/tuberlin_sliding_8s_4s.csv",
                "--window-sec",
                "8",
                "--stride-sec",
                "4",
            ],
        ),
        (
            "12s window / 6s stride",
            [
                sys.executable,
                "sliding_window_extractor.py",
                "--nirs-dir",
                "data/NIRS_01-26_MATLAB",
                "--eeg-dir",
                "data/EEG_01-26_MATLAB",
                "--output",
                "data/tuberlin_sliding_12s_6s.csv",
                "--window-sec",
                "12",
                "--stride-sec",
                "6",
            ],
        ),
    ]
    print("Run these one at a time, then train with COGLOAD_CSV pointed at the output CSV:")
    for label, cmd in commands:
        print(f"\n{label}:")
        print(" ".join(cmd))
        print(f"$env:COGLOAD_CSV='{cmd[cmd.index('--output') + 1]}'; python main.py --no-ablation")


def run_window_ablation(window_sec, stride_sec):
    output = f"data/tuberlin_sliding_{window_sec:g}s_{stride_sec:g}s.csv"
    cmd = [
        sys.executable,
        "sliding_window_extractor.py",
        "--nirs-dir",
        "data/NIRS_01-26_MATLAB",
        "--eeg-dir",
        "data/EEG_01-26_MATLAB",
        "--output",
        output,
        "--window-sec",
        str(window_sec),
        "--stride-sec",
        str(stride_sec),
    ]
    if os.path.exists(output):
        print(f"Using existing extracted CSV: {output}")
    else:
        subprocess.run(cmd, check=True)
    sample_count = len(pd.read_csv(output, usecols=[Config.LABEL_COLUMN]))
    overrides = csv_feature_overrides(output)
    entry = run_hybrid(
        f"Window {window_sec:g}s/{stride_sec:g}s",
        f"Regenerated sliding-window CSV at {output}.",
        overrides,
    )
    entry["sample_count"] = sample_count
    return entry


def pca_quick(eeg_components=50, fnirs_components=100):
    cfg = clone_config()
    set_seed(cfg.RANDOM_SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("Phase 3 experiment: Per-modality PCA quick check")
    print(f"Change: EEG PCA={eeg_components}, fNIRS PCA={fnirs_components}")
    print("=" * 72)

    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, groups = dl.load_raw(return_groups=True)
    train_idx, val_idx, test_idx = dl.split_indices(labels, groups)

    eeg_train, eeg_val, eeg_test = eeg_raw[train_idx], eeg_raw[val_idx], eeg_raw[test_idx]
    fnirs_train, fnirs_val, fnirs_test = (
        fnirs_raw[train_idx],
        fnirs_raw[val_idx],
        fnirs_raw[test_idx],
    )
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    eeg_scaler = RobustScaler()
    fnirs_scaler = StandardScaler()
    eeg_train_s = eeg_scaler.fit_transform(eeg_train)
    eeg_val_s = eeg_scaler.transform(eeg_val)
    eeg_test_s = eeg_scaler.transform(eeg_test)
    fnirs_train_s = fnirs_scaler.fit_transform(fnirs_train)
    fnirs_val_s = fnirs_scaler.transform(fnirs_val)
    fnirs_test_s = fnirs_scaler.transform(fnirs_test)

    n_eeg = min(eeg_components, eeg_train_s.shape[0], eeg_train_s.shape[1])
    n_fnirs = min(fnirs_components, fnirs_train_s.shape[0], fnirs_train_s.shape[1])
    eeg_pca = PCA(n_components=n_eeg, random_state=cfg.RANDOM_SEED)
    fnirs_pca = PCA(n_components=n_fnirs, random_state=cfg.RANDOM_SEED)

    eeg_train_p = eeg_pca.fit_transform(eeg_train_s)
    eeg_val_p = eeg_pca.transform(eeg_val_s)
    eeg_test_p = eeg_pca.transform(eeg_test_s)
    fnirs_train_p = fnirs_pca.fit_transform(fnirs_train_s)
    fnirs_val_p = fnirs_pca.transform(fnirs_val_s)
    fnirs_test_p = fnirs_pca.transform(fnirs_test_s)

    def loader(eeg_x, fnirs_x, y, shuffle=False, drop_last=False):
        return DataLoader(
            CognitiveLoadDataset(
                eeg_x.astype(np.float32),
                fnirs_x.astype(np.float32),
                y.astype(np.int64),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,
        )

    train_loader = loader(eeg_train_p, fnirs_train_p, y_train, shuffle=True, drop_last=True)
    val_loader = loader(eeg_val_p, fnirs_val_p, y_val)
    test_loader = loader(eeg_test_p, fnirs_test_p, y_test)
    meta = {
        "n_eeg_features": n_eeg,
        "n_fnirs_features": n_fnirs,
        "class_weights": configured_class_weights(y_train, cfg),
        "y_test": y_test,
    }

    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "phase3_pca_quick_hybrid.pt")
    model = build_model("hybrid_cnn_lstm", n_eeg, n_fnirs, cfg)
    model, _ = train_model(model, train_loader, val_loader, meta, cfg)
    result = evaluate_model(model, test_loader, cfg, model_name="PCA Quick Hybrid")
    print(f"Guardrail: {'PASS' if passes_guardrail(result) else 'FAIL'}")

    entry = {
        "name": "Per-modality PCA quick check",
        "change": (
            f"Fit PCA separately per modality on train split only: "
            f"EEG {eeg_raw.shape[1]}->{n_eeg} components "
            f"({eeg_pca.explained_variance_ratio_.sum()*100:.1f}% variance), "
            f"fNIRS {fnirs_raw.shape[1]}->{n_fnirs} components "
            f"({fnirs_pca.explained_variance_ratio_.sum()*100:.1f}% variance)."
        ),
        "sample_count": len(labels),
        "result": result,
    }
    append_results("Priority 4: Per-Modality PCA", [entry])
    return entry


def make_loader(eeg, fnirs, labels, cfg, shuffle=False, drop_last=False):
    return DataLoader(
        CognitiveLoadDataset(eeg.astype(np.float32), fnirs.astype(np.float32), labels.astype(np.int64)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def loso(max_folds=None, epochs=None, early_stop=None, overrides=None, section=None):
    cfg = clone_config(overrides)
    if epochs is not None:
        cfg.EPOCHS = epochs
    if early_stop is not None:
        cfg.EARLY_STOP = early_stop

    set_seed(cfg.RANDOM_SEED)
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, _ = dl.load_raw(return_groups=True)
    df = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])
    subjects = df[cfg.SUBJECT_COLUMN].astype(str).to_numpy()
    logo = LeaveOneGroupOut()

    fold_results = []
    for fold, (fit_idx, test_idx) in enumerate(logo.split(eeg_raw, labels, subjects), 1):
        if max_folds is not None and fold > max_folds:
            break

        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.VAL_SIZE,
            random_state=cfg.RANDOM_SEED + fold,
        )
        train_rel, val_rel = next(
            val_splitter.split(
                fit_idx,
                labels[fit_idx],
                subjects[fit_idx],
            )
        )
        train_idx = fit_idx[train_rel]
        val_idx = fit_idx[val_rel]

        eeg_scaler = RobustScaler()
        fnirs_scaler = StandardScaler()
        eeg_train = eeg_scaler.fit_transform(eeg_raw[train_idx])
        eeg_val = eeg_scaler.transform(eeg_raw[val_idx])
        eeg_test = eeg_scaler.transform(eeg_raw[test_idx])
        fnirs_train = fnirs_scaler.fit_transform(fnirs_raw[train_idx])
        fnirs_val = fnirs_scaler.transform(fnirs_raw[val_idx])
        fnirs_test = fnirs_scaler.transform(fnirs_raw[test_idx])

        if dl.eeg_sequence_shape:
            steps, features = dl.eeg_sequence_shape
            eeg_train = eeg_train.reshape(len(eeg_train), steps, features)
            eeg_val = eeg_val.reshape(len(eeg_val), steps, features)
            eeg_test = eeg_test.reshape(len(eeg_test), steps, features)
            n_eeg = dl.eeg_sequence_shape
        else:
            n_eeg = eeg_raw.shape[1]

        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]

        meta = {
            "n_eeg_features": n_eeg,
            "n_fnirs_features": fnirs_raw.shape[1],
            "class_weights": configured_class_weights(y_train, cfg),
            "y_test": y_test,
        }
        train_loader = make_loader(eeg_train, fnirs_train, y_train, cfg, shuffle=True, drop_last=True)
        val_loader = make_loader(eeg_val, fnirs_val, y_val, cfg)
        test_loader = make_loader(eeg_test, fnirs_test, y_test, cfg)

        held_out = subjects[test_idx][0]
        cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, f"phase3_loso_subject_{held_out}.pt")
        print(f"\nLOSO fold {fold}: held-out subject {held_out}")
        model = build_model("hybrid_cnn_lstm", n_eeg, fnirs_raw.shape[1], cfg)
        model, _ = train_model(model, train_loader, val_loader, meta, cfg, verbose=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preds, probs, fold_labels = predict(model.to(device), test_loader, device)
        result = {
            "subject": held_out,
            "acc": accuracy_score(fold_labels, preds),
            "f1_macro": f1_score(fold_labels, preds, average="macro"),
            "per_class_recall": recall_score(
                fold_labels,
                preds,
                labels=list(range(cfg.NUM_CLASSES)),
                average=None,
                zero_division=0,
            ),
        }
        try:
            result["auc"] = roc_auc_score(
                fold_labels,
                probs,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            result["auc"] = float("nan")
        fold_results.append(result)
        print(metric_summary(result))

    f1s = np.array([r["f1_macro"] for r in fold_results], dtype=float)
    accs = np.array([r["acc"] for r in fold_results], dtype=float)
    aucs = np.array([r["auc"] for r in fold_results], dtype=float)
    recalls = np.vstack([r["per_class_recall"] for r in fold_results])

    subject_norm = getattr(cfg, "PER_SUBJECT_NORMALIZE", False)
    subject_norm_stage = getattr(cfg, "PER_SUBJECT_NORMALIZE_STAGE", "after_log")
    class_weight_mode = getattr(cfg, "CLASS_WEIGHT_MODE", "balanced")
    class_weights = getattr(cfg, "CLASS_WEIGHTS", None)
    section = section or "Priority 5: LOSO Cross-Validation"
    lines = [
        "",
        f"## Phase 3 - {section}",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Preprocessing: PER_SUBJECT_NORMALIZE={subject_norm}, "
        f"PER_SUBJECT_NORMALIZE_STAGE={subject_norm_stage}",
        f"- Class weights: mode={class_weight_mode}"
        + (f", values={class_weights}" if class_weights is not None else ""),
        f"- Folds run: {len(fold_results)}",
        f"- Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}",
        f"- Macro F1: mean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}",
        f"- AUC: mean={np.nanmean(aucs):.4f}, std={np.nanstd(aucs):.4f}",
        "- Per-class recall mean: ["
        + ", ".join(f"{r:.4f}" for r in np.mean(recalls, axis=0))
        + "]",
        "",
    ]
    with open("RESULTS.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return {
        "folds": len(fold_results),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "recall_mean": np.mean(recalls, axis=0),
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("priority1")
    sub.add_parser("cross-attention")
    sub.add_parser("window-commands")
    sub.add_parser("window-8s-4s")
    sub.add_parser("window-12s-6s")
    sub.add_parser("subject-normalization-fixed")
    pca_parser = sub.add_parser("pca")
    pca_parser.add_argument("--eeg-components", type=int, default=50)
    pca_parser.add_argument("--fnirs-components", type=int, default=100)
    loso_parser = sub.add_parser("loso")
    loso_parser.add_argument("--max-folds", type=int, default=None)
    loso_parser.add_argument("--epochs", type=int, default=None)
    loso_parser.add_argument("--early-stop", type=int, default=None)
    loso_parser.add_argument("--subject-normalize", choices=["true", "false"], default=None)
    loso_parser.add_argument(
        "--subject-normalize-stage",
        choices=["before_log", "after_log"],
        default=None,
    )
    loso_parser.add_argument(
        "--class-weights",
        nargs=3,
        type=float,
        metavar=("LOW", "MEDIUM", "HIGH"),
        default=None,
    )
    args = parser.parse_args()

    if args.command == "priority1":
        priority1()
    elif args.command == "cross-attention":
        cross_attention()
    elif args.command == "window-commands":
        window_commands()
    elif args.command == "window-8s-4s":
        append_results(
            "Priority 3: Window/Stride Ablation",
            [run_window_ablation(8.0, 4.0)],
        )
    elif args.command == "window-12s-6s":
        append_results(
            "Priority 3: Window/Stride Ablation",
            [run_window_ablation(12.0, 6.0)],
        )
    elif args.command == "subject-normalization-fixed":
        subject_normalization_fixed_check()
    elif args.command == "pca":
        pca_quick(
            eeg_components=args.eeg_components,
            fnirs_components=args.fnirs_components,
        )
    elif args.command == "loso":
        overrides = {}
        if args.subject_normalize is not None:
            overrides["PER_SUBJECT_NORMALIZE"] = args.subject_normalize == "true"
        if args.subject_normalize_stage is not None:
            overrides["PER_SUBJECT_NORMALIZE_STAGE"] = args.subject_normalize_stage
        if args.class_weights is not None:
            overrides["CLASS_WEIGHT_MODE"] = "manual"
            overrides["CLASS_WEIGHTS"] = args.class_weights
        loso(
            max_folds=args.max_folds,
            epochs=args.epochs,
            early_stop=args.early_stop,
            overrides=overrides,
        )


if __name__ == "__main__":
    main()
