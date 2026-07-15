"""
Phase 1 EEG feature validation gate.

Runs EEG-only classical classifiers on a generated v2 CSV. The gate from
EEG_REBUILD_AND_COMPLETION_PLAN.md is RF macro OVR AUC >= 0.55 before retraining
the deep hybrid model.

Example:
    python validate_eeg_v2_gate.py --csv data/tuberlin_sliding_v2.csv
"""

import argparse
import os
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


def _evaluate(name, clf, X_test, y_test):
    probs = clf.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    return {
        "model": name,
        "acc": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "auc": roc_auc_score(y_test, probs, multi_class="ovr", average="macro"),
    }


def _append_results(csv_path, rows, passed, threshold):
    lines = [
        "",
        "## Phase 1 EEG v2 Gate",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"CSV: `{csv_path}`",
        "",
        f"Gate: EEG-only Random Forest macro OVR AUC >= {threshold:.2f}.",
        "",
        "| Model | Accuracy | Macro F1 | AUC |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['acc']:.4f} | {row['f1_macro']:.4f} | {row['auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Gate result: {'PASS' if passed else 'FAIL'}.",
            "",
        ]
    )
    with open("RESULTS.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/tuberlin_sliding_v2.csv")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--train-val", action="store_true", help="Fit on train+val instead of train only")
    parser.add_argument("--no-append-results", action="store_true")
    args = parser.parse_args()

    os.environ["COGLOAD_CSV"] = args.csv

    from config import Config
    from data_loader import CognitiveLoadDataLoader

    cfg = Config()
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, _, labels, groups = dl.load_raw(return_groups=True)
    train_idx, val_idx, test_idx = dl.split_indices(labels, groups)
    fit_idx = np.concatenate([train_idx, val_idx]) if args.train_val else train_idx

    scaler = RobustScaler()
    X_train = scaler.fit_transform(eeg_raw[fit_idx])
    X_test = scaler.transform(eeg_raw[test_idx])
    y_train = labels[fit_idx]
    y_test = labels[test_idx]

    print("\nPhase 1 EEG-only classical gate")
    print("=" * 55)
    print(f"CSV          : {args.csv}")
    print(f"EEG shape    : {dl.eeg_sequence_shape or eeg_raw.shape[1]}")
    print(f"Fit samples  : {len(y_train)}")
    print(f"Test samples : {len(y_test)}")
    if groups is not None:
        print(f"Fit groups   : {len(np.unique(groups[fit_idx]))}")
        print(f"Test groups  : {len(np.unique(groups[test_idx]))}")

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.RANDOM_SEED,
    )
    rf.fit(X_train, y_train)

    logreg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=cfg.RANDOM_SEED,
    )
    logreg.fit(X_train, y_train)

    rows = [
        _evaluate("Random Forest", rf, X_test, y_test),
        _evaluate("Logistic Regression", logreg, X_test, y_test),
    ]

    print("\nResults")
    print("=" * 55)
    for row in rows:
        print(
            f"{row['model']:<20} Acc={row['acc']:.4f}  "
            f"Macro F1={row['f1_macro']:.4f}  AUC={row['auc']:.4f}"
        )

    passed = rows[0]["auc"] >= args.threshold
    print(f"\nGate result: {'PASS' if passed else 'FAIL'} (threshold={args.threshold:.2f})")

    if not args.no_append_results:
        _append_results(args.csv, rows, passed, args.threshold)
        print("Appended gate result to RESULTS.md")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
