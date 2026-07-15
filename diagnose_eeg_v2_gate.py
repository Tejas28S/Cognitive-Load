"""
Diagnostic pass for the EEG v2 classical gate.

Runs the Part B checks from DEBUG_BASELINE_AND_GATE.md without depending on
config.py, so it can be used while config.py is restored to a baseline snapshot.

Examples:
    python diagnose_eeg_v2_gate.py
    python diagnose_eeg_v2_gate.py --no-append-results
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler


LABEL_COLUMN = "Cognitive_Load"
KEY_COLUMNS = ["subject_id", "trial_id", "window_id"]


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def eeg_columns(df):
    return sorted([c for c in df.columns if c.startswith("eeg_")], key=natural_key)


def split_indices(labels, groups=None, test_size=0.20, val_size=0.20, random_seed=42):
    idx = np.arange(len(labels))
    if groups is None:
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            stratify=labels,
            random_state=random_seed,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=labels[train_val_idx],
            random_state=random_seed,
        )
        return train_idx, val_idx, test_idx

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_seed,
    )
    train_val_rel, test_rel = next(splitter.split(idx, labels, groups))
    train_val_idx = idx[train_val_rel]
    test_idx = idx[test_rel]

    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_seed,
    )
    train_rel, val_rel = next(
        val_splitter.split(
            train_val_idx,
            labels[train_val_idx],
            groups[train_val_idx],
        )
    )
    return train_val_idx[train_rel], train_val_idx[val_rel], test_idx


def make_groups(df):
    cols = [c for c in ["subject_id", "trial_id"] if c in df.columns]
    if not cols:
        return None, []
    return df[cols].astype(str).agg("|".join, axis=1).to_numpy(), cols


def subject_baseline_zscore(df, cols, label_column=LABEL_COLUMN):
    if "subject_id" not in df.columns:
        raise ValueError("subject_id is required for subject baseline normalization")

    out = df[cols].to_numpy(dtype=np.float32, copy=True)
    labels = df[label_column].to_numpy()
    subjects = df["subject_id"].to_numpy()

    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        baseline_mask = subject_mask & (labels == 0)
        baseline = out[baseline_mask] if baseline_mask.any() else out[subject_mask]
        mean = np.nanmean(baseline, axis=0)
        std = np.nanstd(baseline, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        out[subject_mask] = (out[subject_mask] - mean) / std

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_model(name, clf, X_test, y_test):
    probs = clf.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    return {
        "model": name,
        "acc": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "auc": roc_auc_score(y_test, probs, multi_class="ovr", average="macro"),
    }


def run_gate(csv_path, name, apply_subject_baseline=False, train_val=False, random_seed=42):
    df = pd.read_csv(csv_path)
    cols = eeg_columns(df)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    groups, group_cols = make_groups(df)
    train_idx, val_idx, test_idx = split_indices(labels, groups, random_seed=random_seed)
    fit_idx = np.concatenate([train_idx, val_idx]) if train_val else train_idx

    if apply_subject_baseline:
        X = subject_baseline_zscore(df, cols)
    else:
        X = df[cols].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X[fit_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = labels[fit_idx]
    y_test = labels[test_idx]

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_seed,
    )
    logreg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_seed,
    )
    rf.fit(X_train, y_train)
    logreg.fit(X_train, y_train)

    return {
        "name": name,
        "csv": csv_path,
        "shape": X.shape,
        "groups": group_cols,
        "feature_slice": X_train[:2, :8],
        "rf": evaluate_model("Random Forest", rf, X_test, y_test),
        "logreg": evaluate_model("Logistic Regression", logreg, X_test, y_test),
    }


def label_alignment(v1_path, v2_path):
    df_v1 = pd.read_csv(v1_path)
    df_v2 = pd.read_csv(v2_path)
    merge_cols = [c for c in KEY_COLUMNS if c in df_v1.columns and c in df_v2.columns]
    check = df_v1[merge_cols + [LABEL_COLUMN]].merge(
        df_v2[merge_cols + [LABEL_COLUMN]],
        on=merge_cols,
        suffixes=("_v1", "_v2"),
        how="outer",
        indicator=True,
    )
    mismatches = check[
        (check["_merge"] == "both")
        & (check[f"{LABEL_COLUMN}_v1"] != check[f"{LABEL_COLUMN}_v2"])
    ]
    return {
        "v1_shape": df_v1.shape,
        "v2_shape": df_v2.shape,
        "v1_labels": df_v1[LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "v2_labels": df_v2[LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "merge_cols": merge_cols,
        "merge_counts": check["_merge"].value_counts().to_dict(),
        "mismatches": len(mismatches),
    }


def baseline_stats(csv_path):
    df = pd.read_csv(csv_path)
    cols = eeg_columns(df)
    values = df[cols].to_numpy(dtype=np.float32)
    parsed = []
    for col in cols:
        match = re.match(r"^eeg_t(\d+)_(.+)$", col)
        if match:
            parsed.append((int(match.group(1)), match.group(2)))
    complete_sequence = False
    sequence_shape = None
    if parsed:
        timesteps = sorted({t for t, _ in parsed})
        base_features = sorted({name for _, name in parsed})
        complete_sequence = set(parsed) == {(t, name) for t in timesteps for name in base_features}
        sequence_shape = (len(timesteps), len(base_features))

    return {
        "csv": csv_path,
        "eeg_cols": len(cols),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "max_abs": float(np.nanmax(np.abs(values))),
        "top_zero_rates": (df[cols] == 0).mean().sort_values(ascending=False).head(10).to_dict(),
        "zero_rate_by_class": {
            int(cls): float((df.loc[df[LABEL_COLUMN] == cls, cols] == 0).to_numpy().mean())
            for cls in sorted(df[LABEL_COLUMN].unique())
        },
        "complete_sequence": complete_sequence,
        "sequence_shape": sequence_shape,
    }


def append_results(path, alignment, stats, gate_rows):
    lines = [
        "",
        "## Debug Pass - EEG v2 Gate Diagnostics",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "### B1 Row/Label Alignment",
        "",
        f"- v1 shape: `{alignment['v1_shape']}`; v2 shape: `{alignment['v2_shape']}`",
        f"- v1 labels: `{alignment['v1_labels']}`; v2 labels: `{alignment['v2_labels']}`",
        f"- merge keys: `{alignment['merge_cols']}`; merge counts: `{alignment['merge_counts']}`",
        f"- label mismatches: `{alignment['mismatches']}`",
        "",
        "### B2 Classical Gate Ablations",
        "",
        "| Variant | CSV | X shape | RF Acc | RF Macro F1 | RF AUC | LogReg Acc | LogReg Macro F1 | LogReg AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['name']} | `{row['csv']}` | `{row['shape']}` | "
            f"{row['rf']['acc']:.4f} | {row['rf']['f1_macro']:.4f} | {row['rf']['auc']:.4f} | "
            f"{row['logreg']['acc']:.4f} | {row['logreg']['f1_macro']:.4f} | {row['logreg']['auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### B3/B4 Baseline Output and Flattening Checks",
            "",
            f"- CSV: `{stats['csv']}`",
            f"- EEG columns: `{stats['eeg_cols']}`; sequence shape: `{stats['sequence_shape']}`; complete grid: `{stats['complete_sequence']}`",
            f"- NaNs: `{stats['nan_count']}`; infs: `{stats['inf_count']}`; max abs: `{stats['max_abs']:.4f}`",
            f"- zero rate by class: `{stats['zero_rate_by_class']}`",
            f"- top zero-rate columns: `{stats['top_zero_rates']}`",
            "",
        ]
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default="data/tuberlin_sliding.csv")
    parser.add_argument("--v2", default="data/tuberlin_sliding_v2.csv")
    parser.add_argument("--v2-no-baseline", default="data/tuberlin_sliding_v2_no_baseline.csv")
    parser.add_argument("--v2-fixed", default="data/tuberlin_sliding_v2_fixed.csv")
    parser.add_argument("--v2-artifact100", default="data/tuberlin_sliding_v2_artifact100.csv")
    parser.add_argument(
        "--v2-artifact100-fixed",
        default="data/tuberlin_sliding_v2_artifact100_fixed.csv",
    )
    parser.add_argument("--train-val", action="store_true")
    parser.add_argument("--no-append-results", action="store_true")
    args = parser.parse_args()

    alignment = label_alignment(args.v1, args.v2)
    stats = baseline_stats(args.v2)

    variants = [
        ("v1 flat original", args.v1, False),
        ("v1 flat + subject baseline", args.v1, True),
        ("v2 subwindow + baseline", args.v2, False),
    ]
    if os.path.exists(args.v2_no_baseline):
        variants.append(("v2 subwindow only", args.v2_no_baseline, False))
    if os.path.exists(args.v2_fixed):
        variants.append(("v2 fixed robust baseline", args.v2_fixed, False))
    if os.path.exists(args.v2_artifact100):
        variants.append(("v2 subwindow + baseline + artifact100", args.v2_artifact100, False))
    if os.path.exists(args.v2_artifact100_fixed):
        variants.append(("v2 fixed robust baseline + artifact100", args.v2_artifact100_fixed, False))

    gate_rows = [
        run_gate(path, name, apply_subject_baseline=baseline, train_val=args.train_val)
        for name, path, baseline in variants
    ]

    print("\nB1 row/label alignment")
    print("=" * 72)
    print(alignment)
    print("\nB2 gate ablations")
    print("=" * 72)
    for row in gate_rows:
        print(
            f"{row['name']:<40} RF AUC={row['rf']['auc']:.4f} "
            f"RF F1={row['rf']['f1_macro']:.4f}  LogReg AUC={row['logreg']['auc']:.4f}"
        )
        print(f"  X shape={row['shape']} groups={row['groups']}")
        print(f"  feature slice before RF.fit(): {np.round(row['feature_slice'], 4).tolist()}")
    print("\nB3/B4 stats")
    print("=" * 72)
    print(stats)

    if not args.no_append_results:
        append_results("RESULTS.md", alignment, stats, gate_rows)
        print("\nAppended diagnostics to RESULTS.md")


if __name__ == "__main__":
    main()
