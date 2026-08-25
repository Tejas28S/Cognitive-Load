"""
export_binary_reliability.py - add reliability data to serving/frontend_data/
metrics.json for the prediction report (per-class recall/precision, lockbox
numbers, per-subject accuracy spread), and rewrite
serving/frontend_data/confusion_matrices.json with the BINARY confusion matrix.

The confusion-matrix rewrite fixes a real defect: the binary serving bundle
shipped the 3-class study's matrices (labels Low/Medium/High), so every page
asking the API for the binary model's confusion matrix got a 3x3 matrix from a
different model. The per-modality (eeg / fnirs) matrices are dropped rather
than carried over, because no binary equivalent was ever computed - a 404 is
the honest answer there, and the UI degrades to hiding those panels.

Every number here is PARSED from RESULTS_BINARY_IMPROVEMENT.md - the committed,
run-generated results doc - not hand-typed, so the report can't silently drift
from the artifact it's supposed to trace to. Existing keys in metrics.json
(loso.hybrid, ensemble_accepted, acceptance_reason) are preserved untouched;
this only adds new keys.

    .venv/Scripts/python export_binary_reliability.py
"""
import json
import re

RESULTS_MD = "RESULTS_BINARY_IMPROVEMENT.md"
METRICS_JSON = "serving/frontend_data/metrics.json"
CONFUSION_JSON = "serving/frontend_data/confusion_matrices.json"


def parse_final_results_table(text):
    """The '### Final results' table: LOSO + Lockbox rows for 'selected'."""
    m = re.search(
        r"### Final results\n\n(.*?)\n\n\*\*Paired significance", text, re.S
    )
    if not m:
        raise ValueError("could not find '### Final results' table")
    rows = [l for l in m.group(1).splitlines() if l.startswith("|")]
    out = {}
    for line in rows[2:]:  # skip header + separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        protocol, model = cells[0], cells[1]
        if model != "**selected**":
            continue
        f1_mean, f1_std = cells[2].split(" ± ")
        acc_mean, acc_std = cells[3].split(" ± ")
        auc_mean, auc_std = cells[5].split(" ± ")
        entry = {
            "f1_mean": float(f1_mean), "f1_std": float(f1_std),
            "acc_mean": float(acc_mean), "acc_std": float(acc_std),
            "auc_mean": float(auc_mean), "auc_std": float(auc_std),
        }
        if protocol.startswith("LOSO"):
            out["loso"] = entry
        elif protocol.startswith("Lockbox"):
            out["lockbox"] = entry
    return out


def parse_per_class_table(text):
    """The 'Per class:' table under 'Full metric record'."""
    m = re.search(
        r"\*\*Per class:\*\*\n\n(.*?)\n\n\*\*Confusion matrix", text, re.S
    )
    if not m:
        raise ValueError("could not find 'Per class:' table")
    rows = [l for l in m.group(1).splitlines() if l.startswith("|")]
    out = {}
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        cls = cells[0]
        out[cls] = {
            "precision": float(cells[1]), "recall": float(cells[2]),
            "f1": float(cells[3]), "specificity": float(cells[4]),
            "support": int(cells[5]),
        }
    return out


def parse_confusion_matrix(text):
    """The '**Confusion matrix**' table under 'Full metric record'.

    Returns (labels, matrix) with rows = true class, columns = predicted, in
    the order the table lists them.
    """
    m = re.search(
        r"\*\*Confusion matrix\*\* \(rows = true, columns = predicted\):\n\n(.*?)\n\n",
        text, re.S,
    )
    if not m:
        raise ValueError("could not find the confusion-matrix table")
    rows = [l for l in m.group(1).splitlines() if l.startswith("|")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    labels = [c for c in header[1:] if c]
    matrix = []
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        matrix.append([int(c) for c in cells[1:] if c])
    if len(matrix) != len(labels):
        raise ValueError(
            f"confusion matrix is {len(matrix)}x{len(matrix[0]) if matrix else 0} "
            f"but there are {len(labels)} labels"
        )
    return labels, matrix


def parse_per_subject_table(text):
    """The 'Per-subject held-out performance' table -> spread summary."""
    m = re.search(
        r"### Per-subject held-out performance\n\n(.*?)\n\nSpread across subjects",
        text, re.S,
    )
    if not m:
        raise ValueError("could not find 'Per-subject held-out performance' table")
    rows = [l for l in m.group(1).splitlines() if l.startswith("|")]
    subjects = []
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        subjects.append({
            "subject_id": int(cells[0].lstrip("S")),
            "windows": int(cells[1]),
            "accuracy": float(cells[2]),
        })
    if not subjects:
        raise ValueError("no subject rows parsed")
    best = max(subjects, key=lambda s: s["accuracy"])
    worst = min(subjects, key=lambda s: s["accuracy"])
    mean_acc = sum(s["accuracy"] for s in subjects) / len(subjects)
    return {
        "protocol": "26-fold LOSO, all subjects",
        "n_subjects": len(subjects),
        "mean_accuracy": round(mean_acc, 4),
        "best": best,
        "worst": worst,
        "note": (
            "Per-subject accuracy varies widely around the population mean "
            "(e.g. one subject reaches "
            f"{best['accuracy'] * 100:.1f}%, another falls to "
            f"{worst['accuracy'] * 100:.1f}%) - the model estimates population-level "
            "reliability, not a personalized guarantee for any one person."
        ),
    }


def main():
    with open(RESULTS_MD, encoding="utf-8") as f:
        text = f.read()
    with open(METRICS_JSON, encoding="utf-8") as f:
        metrics = json.load(f)

    final = parse_final_results_table(text)
    per_class = parse_per_class_table(text)
    per_subject = parse_per_subject_table(text)
    labels, matrix = parse_confusion_matrix(text)

    # Cross-check the matrix against the per-class table before writing it: row
    # sums must equal each class's support, and the diagonal must equal TP.
    # If the doc is ever regenerated inconsistently this fails loudly instead
    # of shipping a matrix that disagrees with the metrics beside it.
    for i, label in enumerate(labels):
        support = per_class[label]["support"]
        if sum(matrix[i]) != support:
            raise ValueError(
                f"{label}: confusion row sums to {sum(matrix[i])}, "
                f"per-class support is {support}"
            )
        recall = matrix[i][i] / support
        if abs(recall - per_class[label]["recall"]) > 5e-4:
            raise ValueError(
                f"{label}: confusion implies recall {recall:.4f}, "
                f"per-class table says {per_class[label]['recall']}"
            )

    # Merge acc_std/auc_std into the existing loso.hybrid block rather than
    # replacing it (f1_mean/f1_std/acc_mean/auc_mean/recall_mean already there
    # and already verified against this same doc).
    metrics.setdefault("loso", {}).setdefault("hybrid", {}).update({
        "acc_std": final["loso"]["acc_std"],
        "auc_std": final["loso"]["auc_std"],
    })
    metrics["lockbox"] = {"hybrid": final["lockbox"]}
    metrics["per_class"] = per_class
    metrics["per_subject_summary"] = per_subject
    metrics["source"] = (
        "RESULTS_BINARY_IMPROVEMENT.md (26-fold LOSO + 8-subject lockbox, "
        "run_20260820_141126_19cb)"
    )

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    # Only `hybrid` is written: it is the only model the binary campaign
    # actually produced a confusion matrix for. The 3-class bundle's `eeg` and
    # `fnirs` matrices are deliberately NOT carried over - serving a 3-class
    # matrix under a binary request is the defect this replaces.
    confusion = {
        "labels": labels,
        "hybrid": matrix,
        "source": (
            "RESULTS_BINARY_IMPROVEMENT.md - 26-fold LOSO, "
            "4,576 pooled held-out windows (run_20260820_141126_19cb)"
        ),
    }
    with open(CONFUSION_JSON, "w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2)
        f.write("\n")

    print(f"Updated {CONFUSION_JSON}:")
    print(f"  labels = {labels}")
    print(f"  hybrid = {matrix}")
    print(f"Updated {METRICS_JSON}:")
    print(f"  loso.hybrid.acc_std = {final['loso']['acc_std']}")
    print(f"  lockbox.hybrid = {final['lockbox']}")
    print(f"  per_class = {per_class}")
    print(f"  per_subject_summary = {per_subject}")


if __name__ == "__main__":
    main()
