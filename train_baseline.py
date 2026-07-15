import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config
from data_loader import CognitiveLoadDataLoader


def feature_modality(name):
    lower = name.lower()
    if lower.startswith("eeg_"):
        return "EEG"
    if lower.startswith("hbo_"):
        return "HbO"
    if lower.startswith("hbr_"):
        return "HbR"
    return "Other"


def summarize_feature_importances(model, feature_names, cfg):
    importances = model.feature_importances_
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "modality": [feature_modality(name) for name in feature_names],
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    out_path = os.path.join(cfg.OUTPUT_DIR, "rf_feature_importances.csv")
    df.to_csv(out_path, index=False)

    top20 = df.head(20).copy()
    modality_summary = (
        df.groupby("modality")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    print("\nRandom Forest feature importance")
    print("=" * 55)
    print(f"Saved full importances: {out_path}")
    print("\nImportance by modality:")
    for _, row in modality_summary.iterrows():
        print(f"{row['modality']:<5}: {row['importance']:.4f}")
    print("\nTop 20 features:")
    for _, row in top20.iterrows():
        print(f"{row['feature']:<24} {row['importance']:.5f}  {row['modality']}")

    return {
        "path": out_path,
        "top20": top20,
        "modality_summary": modality_summary,
    }


def evaluate_classifier(name, clf, X_test, y_test, cfg):
    probs = clf.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    metrics = {
        "model": name,
        "acc": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
    }
    try:
        metrics["auc"] = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except Exception:
        metrics["auc"] = float("nan")

    report = classification_report(
        y_test,
        preds,
        target_names=cfg.CLASS_NAMES,
        digits=4,
        output_dict=True,
    )
    metrics["medium_recall"] = report[cfg.CLASS_NAMES[1]]["recall"]

    print(f"\n{name} Test Results")
    print("=" * 55)
    print(f"Accuracy        : {metrics['acc']:.4f} ({metrics['acc']:.2%})")
    print(f"Macro F1        : {metrics['f1_macro']:.4f}")
    print(f"Weighted F1     : {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC (macro) : {metrics['auc']:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, preds, target_names=cfg.CLASS_NAMES, digits=4))

    return metrics


def append_results(results, cfg, rf_importance=None):
    path = "RESULTS.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    transform_note = (
        f"EEG_LOG_TRANSFORM={cfg.EEG_LOG_TRANSFORM}, "
        f"FNIRS_SIGNED_LOG_TRANSFORM={cfg.FNIRS_SIGNED_LOG_TRANSFORM}"
    )
    lines = [
        "",
        "## Classical Baseline Run",
        "",
        f"Run timestamp: {timestamp}",
        "",
        f"Preprocessing: {transform_note}. Split grouping: subject_id + trial_id when present.",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | AUC | Medium Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['acc']:.4f} | {row['f1_macro']:.4f} | "
            f"{row['f1_weighted']:.4f} | {row['auc']:.4f} | {row['medium_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: compare this baseline with the deep models to separate data ceiling from architecture effects.",
            "",
        ]
    )
    if rf_importance is not None:
        lines.extend(
            [
                "Random Forest feature importance summary:",
                "",
                "| Modality | Total Importance |",
                "|---|---:|",
            ]
        )
        for _, row in rf_importance["modality_summary"].iterrows():
            lines.append(f"| {row['modality']} | {row['importance']:.4f} |")
        lines.extend(
            [
                "",
                "Top 20 Random Forest features:",
                "",
                "| Rank | Feature | Modality | Importance |",
                "|---:|---|---|---:|",
            ]
        )
        for rank, (_, row) in enumerate(rf_importance["top20"].iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['feature']} | {row['modality']} | "
                f"{row['importance']:.5f} |"
            )
        lines.extend(
            [
                "",
                f"Full importance table saved to `{rf_importance['path']}`.",
                "",
            ]
        )
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nAppended baseline results to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm", action="store_true", help="Also train an RBF SVM baseline")
    parser.add_argument("--no-append-results", action="store_true", help="Do not append to RESULTS.md")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, groups = dl.load_raw(return_groups=True)
    X_all = np.hstack([eeg_raw, fnirs_raw]).astype(np.float32)
    feature_names = list(dl.eeg_cols) + list(dl.fnirs_cols)

    train_idx, val_idx, test_idx = dl.split_indices(labels, groups)
    train_val_idx = np.concatenate([train_idx, val_idx])
    X_train, y_train = X_all[train_val_idx], labels[train_val_idx]
    X_test, y_test = X_all[test_idx], labels[test_idx]

    print("\nClassical baseline split")
    print("=" * 55)
    print(f"Train+Val samples: {len(y_train)}")
    print(f"Test samples     : {len(y_test)}")
    if groups is not None:
        print(f"Train+Val groups : {len(np.unique(groups[train_val_idx]))}")
        print(f"Test groups      : {len(np.unique(groups[test_idx]))}")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
    results = [evaluate_classifier("Random Forest", rf, X_test, y_test, cfg)]
    rf_importance = summarize_feature_importances(rf, feature_names, cfg)

    if args.svm:
        svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=cfg.RANDOM_SEED,
            ),
        )
        svm.fit(X_train, y_train)
        results.append(evaluate_classifier("RBF SVM", svm, X_test, y_test, cfg))

    if not args.no_append_results:
        append_results(results, cfg, rf_importance=rf_importance)


if __name__ == "__main__":
    main()
