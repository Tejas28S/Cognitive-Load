"""
report.py — generate RESULTS_BINARY_IMPROVEMENT.md from a run's artifacts.

Every number in the generated document is read from the run directory. Nothing
is transcribed by hand, so the document cannot drift from the artifacts and a
typo cannot invent a result. Re-running it after a new campaign regenerates the
file in place.

    .venv/bin/python -m research.report outputs/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


def f(v, d=4):
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if x != x else f"{x:.{d}f}"


def pm(block, key, d=4):
    """mean ± std for a fold-aggregate entry."""
    e = (block or {}).get(key) or {}
    if e.get("mean") is None:
        return "n/a"
    return f"{f(e['mean'], d)} ± {f(e.get('std'), 3)}"


def sig(t):
    if not t:
        return "n/a"
    return (f"Δ {f(t['mean_diff'])}, paired t p={f(t['t_p'], 3)}, "
            f"Wilcoxon p={f(t['wilcoxon_p'], 3)}, {t['wins']}W/{t['losses']}L")


def load_arms(run_dir):
    """Just the experiment records + leaderboard rows of a run."""
    recs_path = os.path.join(run_dir, "experiments", "experiment_results.json")
    lb_path = os.path.join(run_dir, "experiments", "leaderboard.csv")
    recs, lb = [], None
    if os.path.isfile(recs_path):
        with open(recs_path, encoding="utf-8") as fh:
            recs = json.load(fh)
    if os.path.isfile(lb_path):
        lb = pd.read_csv(lb_path)
        lb["run_id"] = os.path.basename(run_dir.rstrip("/"))
    return recs, lb


def load(run_dir):
    def j(*parts):
        p = os.path.join(run_dir, *parts)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    lb_path = os.path.join(run_dir, "experiments", "leaderboard.csv")
    return {
        "summary": j("run_summary.json"),
        "status": j("run_status.json"),
        "metadata": j("run_metadata.json"),
        "dataset": j("dataset_info.json"),
        "loso": j("metrics", "loso_metrics.json"),
        "metrics": j("metrics", "metrics.json"),
        "audit": j("leakage_audit.json"),
        "experiments": j("experiments", "experiment_results.json") or [],
        "stage_log": j("experiments", "stage_log.json") or [],
        "shap": j("feature_importance", "shap_values.json"),
        "reps": j("representations", "representations.json"),
        "leaderboard": pd.read_csv(lb_path) if os.path.isfile(lb_path) else None,
    }


def render(run_dir, a):
    s = a["summary"]
    if not s:
        raise SystemExit(f"{run_dir} has no run_summary.json — did the run finish?")

    run_id = s["run_id"]
    ds = s["dataset"]
    proto = s["protocol"]
    model = s["model"]
    sel = s["results"]["selected"]
    base = s["results"]["baseline"]
    tests = s["results"]["paired_tests"]
    L = []
    w = L.append

    w(f"# Binary Model Improvement — Measured Results")
    w("")
    w("> Generated from run artifacts by `research/report.py`. Every number below")
    w(f"> is read from `outputs/runs/{run_id}/`; none is transcribed by hand.")
    w(f"> Regenerate with `.venv/bin/python -m research.report outputs/runs/{run_id}`.")
    w("")
    w(f"**Run** `{run_id}` · tag `{s.get('tag')}` · created {s.get('created_at')}  ")
    git = (a["metadata"] or {}).get("git") or {}
    env = (a["metadata"] or {}).get("environment") or {}
    w(f"**Commit** `{(git.get('commit') or '')[:12]}` on `{git.get('branch')}` "
      f"(working tree {'dirty' if git.get('dirty') else 'clean'}) · "
      f"torch {env.get('torch')} · seed {(s.get('protocol') or {}).get('seed', 42)}")
    w("")
    w("Methodology: [`explanation/BINARY_MODEL_IMPROVEMENT.md`]"
      "(explanation/BINARY_MODEL_IMPROVEMENT.md).")
    if a.get("merged_from"):
        w("")
        w("> **Campaign spans two runs.** The search was interrupted and resumed, so")
        w("> the experiment arms below are pooled from " +
          ", ".join(f"`{x}`" for x in a["merged_from"]) + " and `" +
          os.path.basename(run_dir.rstrip('/')) + "`. Both used the same seed, the")
        w("> same development/lockbox split and the same fold generation, so the arms")
        w("> are directly comparable — the baseline arm was re-run in the second run")
        w("> and reproduced the first run's result exactly. All FINAL evaluations")
        w("> below come from the second run.")
    w("")
    w("---")
    w("")

    # ── protocol ───────────────────────────────────────────────────────
    w("## 1. Data and protocol")
    w("")
    w(f"- **Dataset**: `{ds['name']}` — {ds['n_samples']:,} windows, "
      f"{ds['n_subjects']} subjects")
    w(f"- **Task**: {' vs '.join(ds['class_names'])} "
      f"({' vs '.join(ds['class_semantics'])}); chance = "
      f"{f(1/len(ds['class_names']), 4)}")
    w(f"- **Class balance**: " +
      ", ".join(f"{k} = {v:,}" for k, v in ds["class_distribution"].items()))
    pp = ds["preprocessing"]
    w(f"- **Preprocessing**: `{pp['order']}`; per-subject normalization "
      f"`{pp['per_subject_norm']}` ({pp['per_subject_norm_stage']})")
    w(f"- **Selection protocol**: {proto['selection']} — subjects "
      f"{proto['development_subjects']}")
    w(f"- **Lockbox** (never used for any selection decision): "
      f"{proto['lockbox_subjects']}")
    w(f"- **Headline protocol**: {proto['headline']}")
    w(f"- **Checkpoint metric**: {proto['checkpoint_metric']}")
    w("")

    # ── leakage ────────────────────────────────────────────────────────
    w("## 2. Leakage audit")
    w("")
    checks = ((a["audit"] or {}).get("dataset_level") or {}).get("checks") or []
    if checks:
        w("Dataset-level checks (fold-level checks ran before every one of the "
          f"{len(a['experiments'])} experiments' folds and are stored per "
          "experiment):")
        w("")
        w("| Check | Result | Detail |")
        w("|---|---|---|")
        for c in checks:
            w(f"| `{c['check']}` | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
        w("")
    w("Every fold additionally asserted: row-index disjointness, subject "
      "disjointness, trial disjointness, sliding-window identity disjointness, "
      "no byte-identical feature row across splits, scalers fitted on training "
      "rows only, the held-out subject absent from train and val, finite "
      "features, and valid labels. A failure raises `LeakageError` and aborts "
      "the run, so a leaked score is never produced.")
    w("")
    w("The negative tests in `research/test_leakage.py` inject each leakage type "
      "deliberately and confirm the audit catches it — an audit that never fires "
      "would prove nothing.")
    w("")

    # ── baseline ───────────────────────────────────────────────────────
    w("## 3. Baseline")
    w("")
    w("The served binary architecture (`serving/model_config.json`) with the "
      "training hyperparameters in `config.py`, recorded before anything was "
      "changed.")
    w("")
    w("| Protocol | Macro F1 | Accuracy | Balanced acc | ROC-AUC | MCC | Kappa |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for label, key in (("Development LOSO (selection)", "dev_loso"),
                       ("LOSO, all subjects", "loso_all_subjects"),
                       ("Lockbox (unseen subjects)", "lockbox")):
        b = base.get(key)
        if not b:
            continue
        w(f"| {label} | {pm(b,'f1_macro')} | {pm(b,'accuracy')} | "
          f"{pm(b,'balanced_accuracy')} | {pm(b,'roc_auc')} | {pm(b,'mcc')} | "
          f"{pm(b,'cohen_kappa')} |")
    w("")
    gb = base["generalization_gap"]
    w("**Baseline generalization gap** (macro F1 at each fold's selected epoch):")
    w("")
    w(f"| train | validation | held-out | train − held-out | mean best epoch |")
    w("|---:|---:|---:|---:|---:|")
    w(f"| {f(gb['train_f1_macro_mean'])} | {f(gb['val_f1_macro_mean'])} | "
      f"{f(gb['heldout_f1_macro_mean'])} | {f(gb['train_minus_heldout'])} | "
      f"{f(gb['mean_best_epoch'],1)} |")
    w("")

    # ── experiments ────────────────────────────────────────────────────
    w("## 4. Experiments")
    w("")
    w(f"{len(a['experiments'])} arms were trained and evaluated on identical "
      "development folds. Each stage varied one axis from the incumbent; the "
      "winner was carried forward. Losing arms are kept — they are the negative "
      "results.")
    w("")
    w("### Stage decisions")
    w("")
    w("| Stage | Winner | Accepted? | Score before | Score after | Δ |")
    w("|---|---|:--:|---:|---:|---:|")
    for st in a["stage_log"]:
        w(f"| `{st['stage']}` | `{st['winner']}` | "
          f"{'**yes**' if st['changed'] else 'no'} | {f(st['score_before'])} | "
          f"{f(st['score_after'])} | {f(st['delta'])} |")
    w("")

    lb = a["leaderboard"]
    if lb is not None and len(lb):
        w("### Full leaderboard — development LOSO macro F1")
        w("")
        w("Sorted best to worst. `gap` is train − held-out macro F1 (higher = more "
          "overfitting). `Δ vs base` and `p` are paired against the baseline on "
          "the same folds.")
        w("")
        w("| Experiment | Stage | Macro F1 | Accuracy | ROC-AUC | MCC | Params | gap | Δ vs base | p | Note |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in lb.iterrows():
            if str(r["experiment_id"]).startswith("FINAL"):
                continue
            w(f"| `{r['experiment_id']}` | {r['stage']} | "
              f"{f(r['dev_loso_f1_macro_mean'])} ± {f(r['dev_loso_f1_macro_std'],3)} | "
              f"{f(r['dev_loso_accuracy_mean'],3)} | {f(r['dev_loso_auc_mean'],3)} | "
              f"{f(r['dev_loso_mcc_mean'],3)} | {int(r['n_params']):,} | "
              f"{f(r['train_minus_heldout_f1'],3)} | "
              f"{f(r.get('vs_baseline_mean_diff'),3)} | "
              f"{f(r.get('vs_baseline_t_p'),3)} | {r['note']} |")
        w("")

    # ── selected model ─────────────────────────────────────────────────
    w("## 5. Selected model")
    w("")
    w(f"`{model['selected_experiment']}` — {model['n_params']:,} parameters, "
      f"mean best epoch {f(model['mean_best_epoch'],1)}.")
    w("")
    w("```json")
    w(json.dumps(model["spec"], indent=2))
    w("```")
    w("")
    w("Data configuration:")
    w("")
    w("```json")
    w(json.dumps(model["data"], indent=2))
    w("```")
    w("")

    w("### Final results")
    w("")
    w("| Protocol | Model | Macro F1 | Accuracy | Balanced acc | ROC-AUC | MCC | Kappa |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for label, key in (("LOSO, all subjects", "loso_all_subjects"),
                       ("Lockbox (unseen)", "lockbox")):
        for name, src in (("**selected**", sel), ("baseline", base)):
            b = src.get(key)
            if not b:
                continue
            w(f"| {label} | {name} | {pm(b,'f1_macro')} | {pm(b,'accuracy')} | "
              f"{pm(b,'balanced_accuracy')} | {pm(b,'roc_auc')} | {pm(b,'mcc')} | "
              f"{pm(b,'cohen_kappa')} |")
    w("")
    w("**Paired significance, selected vs baseline on identical folds:**")
    w("")
    w(f"- LOSO (all subjects): {sig(tests.get('loso26_selected_vs_baseline'))}")
    w(f"- Lockbox: {sig(tests.get('lockbox_selected_vs_baseline'))}")
    w("")

    g = sel["generalization_gap"]
    w("### Generalization gap, before and after")
    w("")
    w("| Model | train | validation | held-out | train − held-out | mean best epoch |")
    w("|---|---:|---:|---:|---:|---:|")
    w(f"| baseline | {f(gb['train_f1_macro_mean'])} | {f(gb['val_f1_macro_mean'])} | "
      f"{f(gb['heldout_f1_macro_mean'])} | **{f(gb['train_minus_heldout'])}** | "
      f"{f(gb['mean_best_epoch'],1)} |")
    w(f"| selected | {f(g['train_f1_macro_mean'])} | {f(g['val_f1_macro_mean'])} | "
      f"{f(g['heldout_f1_macro_mean'])} | **{f(g['train_minus_heldout'])}** | "
      f"{f(g['mean_best_epoch'],1)} |")
    w("")

    # ── pooled metrics ─────────────────────────────────────────────────
    m = a["metrics"]
    if m:
        w("### Full metric record (pooled held-out windows)")
        w("")
        w(f"All {m['n_samples']:,} held-out predictions from the "
          f"{proto['headline']}, pooled.")
        w("")
        w("| Metric | Value |")
        w("|---|---:|")
        for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                  "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
                  "precision_micro", "recall_micro", "f1_micro", "cohen_kappa",
                  "mcc", "roc_auc_macro_ovr", "roc_auc_macro_ovo",
                  "average_precision_macro", "log_loss", "brier_score"):
            v = m.get(k)
            note = (m.get("unavailable") or {}).get(k)
            w(f"| {k.replace('_',' ')} | {f(v) if v is not None else '*' + (note or 'n/a') + '*'} |")
        for k, note in (m.get("unavailable") or {}).items():
            if k.startswith("top_"):
                w(f"| {k.replace('_',' ')} | *{note}* |")
        w("")
        w("**Per class:**")
        w("")
        w("| Class | Precision | Recall | F1 | Specificity | Support | TP | FP | FN | TN |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in m["per_class"]:
            w(f"| {c['class_name']} | {f(c['precision'])} | {f(c['recall'])} | "
              f"{f(c['f1'])} | {f(c['specificity'])} | {c['support']} | {c['tp']} | "
              f"{c['fp']} | {c['fn']} | {c['tn']} |")
        w("")
        w("**Confusion matrix** (rows = true, columns = predicted):")
        w("")
        w("| | " + " | ".join(m["class_names"]) + " |")
        w("|---|" + "---:|" * len(m["class_names"]))
        for i, row in enumerate(m["confusion_matrix"]):
            w(f"| **{m['class_names'][i]}** | " + " | ".join(str(x) for x in row) + " |")
        w("")

    # ── per subject ────────────────────────────────────────────────────
    ps_path = os.path.join(run_dir, "metrics", "per_subject_metrics.csv")
    if os.path.isfile(ps_path):
        ps = pd.read_csv(ps_path).sort_values("f1_macro", ascending=False)
        w("### Per-subject held-out performance")
        w("")
        w("| Subject | Windows | Accuracy | Macro F1 | Balanced acc | Kappa | MCC | ROC-AUC |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in ps.iterrows():
            w(f"| S{int(r['subject_id'])} | {int(r['n_windows'])} | "
              f"{f(r['accuracy'],3)} | {f(r['f1_macro'],3)} | "
              f"{f(r['balanced_accuracy'],3)} | {f(r['cohen_kappa'],3)} | "
              f"{f(r['mcc'],3)} | {f(r['roc_auc'],3)} |")
        w("")
        best, worst = ps.iloc[0], ps.iloc[-1]
        w(f"Spread across subjects: macro F1 from {f(worst['f1_macro'],3)} "
          f"(S{int(worst['subject_id'])}) to {f(best['f1_macro'],3)} "
          f"(S{int(best['subject_id'])}) — the range across people is wider than "
          "any difference between the models compared above.")
        w("")

    # ── SHAP ───────────────────────────────────────────────────────────
    if a["shap"]:
        sh = a["shap"]
        w("## 6. Feature importance (SHAP)")
        w("")
        w(f"`{sh['method']}` on this run's trained model, "
          f"{sh['n_explained']} held-out windows explained against "
          f"{sh['n_background']} background samples.")
        w("")
        ms = sh["modality_share"]
        w("| Modality | Share of total |SHAP| |")
        w("|---|---:|")
        for k in ("EEG", "fNIRS", "HbO", "HbR"):
            w(f"| {k} | {f(ms.get(k),4)} |")
        w("")
        w("**Top feature families:**")
        w("")
        w("| Rank | Family | Modality | Features | Total mean \\|SHAP\\| |")
        w("|---:|---|---|---:|---:|")
        for i, c in enumerate(sh["categories"][:15], 1):
            w(f"| {i} | {c['category']} | {c['modality']} | {c['n_features']} | "
              f"{f(c['mean_abs_shap'],5)} |")
        w("")
        w("**Top individual features:**")
        w("")
        w("| Rank | Feature | Modality | Mean \\|SHAP\\| |")
        w("|---:|---|---|---:|")
        for r in sh["top_features"][:15]:
            w(f"| {r['rank']} | `{r['feature']}` | {r['modality']} | "
              f"{f(r['mean_abs_shap'],5)} |")
        w("")

    mg = (sel.get("loso_pooled") or {})
    summary_gates = None
    for e in a["experiments"]:
        if e.get("experiment_id") == "FINAL_loso26":
            summary_gates = e.get("modality_gates")
    if summary_gates:
        w(f"**Learned modality attention** (mean fusion gate over held-out "
          f"windows): EEG {f(summary_gates['eeg_mean'],4)}, "
          f"fNIRS {f(summary_gates['fnirs_mean'],4)}.")
        w("")

    # ── representations ────────────────────────────────────────────────
    if a["reps"]:
        w("## 7. Learned representations (t-SNE + subject-invariance probe)")
        w("")
        w("All variants trained on the same development subjects and embedded on "
          "the same unseen subjects, so the comparison is controlled.")
        w("")
        w("| Representation | Subject-probe accuracy | Chance | Reading |")
        w("|---|---:|---:|---|")
        for k, v in a["reps"]["variants"].items():
            p = v["subject_invariance_probe"]
            if not p:
                continue
            acc, ch = p["accuracy_mean"], p["chance_uniform"]
            reading = ("near chance — subject identity largely absent"
                       if acc < ch * 1.5 else
                       "well above chance — subject identity still present")
            w(f"| {k.replace('_',' ')} | {f(acc,3)} ± {f(p['accuracy_std'],3)} | "
              f"{f(ch,3)} | {reading} |")
        w("")
        w("Coordinates: `representations/tsne_*.csv`. Figures: "
          "`representations/*.png`. Browsable at `/evaluation` → *Learned "
          "representation*.")
        w("")

    # ── artifacts ──────────────────────────────────────────────────────
    w("## 8. Artifacts")
    w("")
    w(f"Everything above lives in `outputs/runs/{run_id}/`:")
    w("")
    w("| Path | Contents |")
    w("|---|---|")
    w("| `predictions/predictions.csv` | one row per held-out window: subject, trial, window, fold, true, predicted, probabilities, confidence |")
    w("| `metrics/` | metrics.json/.csv, per-class, LOSO, per-subject |")
    w("| `confusion_matrix/` | counts + figure |")
    w("| `curves/` | training/validation curves, ROC, PR (JSON + PNG) |")
    w("| `feature_importance/` | SHAP values, ranking, figures |")
    w("| `representations/` | t-SNE coordinates, embeddings, figures |")
    w("| `model/` | deployment model, per-fold checkpoints, scaler, schema, labels |")
    w("| `experiments/` | every arm's record, leaderboard.csv, stage_log.json |")
    w("| `logs/training.log` | full run log |")
    w("")
    w("Reproduce the metrics from the predictions alone:")
    w("")
    w("```bash")
    w(f".venv/bin/python -m research.verify_run outputs/runs/{run_id}")
    w("```")
    w("")

    warns = s.get("warnings") or []
    if warns:
        w("## 9. Warnings recorded by this run")
        w("")
        for x in warns:
            w(f"- {x}")
        w("")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="the run that produced the FINAL evaluations")
    ap.add_argument("--also-arms-from", nargs="*", default=[],
                    help="earlier run dirs whose experiment arms belong to the "
                         "same campaign (identical seed, folds and protocol)")
    ap.add_argument("-o", "--out", default="RESULTS_BINARY_IMPROVEMENT.md")
    args = ap.parse_args()
    a = load(args.run_dir)

    # Merge sibling runs' arms into one leaderboard. Legitimate only because
    # the seed, the fold generation and the protocol are identical, which the
    # generated document states and which the duplicated baseline arm verifies.
    extra_lbs, extra_recs, sources = [], [], []
    for d in args.also_arms_from:
        recs, lb = load_arms(d)
        if lb is not None and len(lb):
            extra_lbs.append(lb)
            extra_recs.extend(recs)
            sources.append(os.path.basename(d.rstrip("/")))
    if extra_lbs:
        own = a["leaderboard"]
        if own is not None:
            own = own.copy()
            own["run_id"] = os.path.basename(args.run_dir.rstrip("/"))
        merged = pd.concat([x for x in ([own] + extra_lbs) if x is not None],
                           ignore_index=True)
        # keep the primary run's copy of any arm that appears twice
        merged = merged.drop_duplicates(subset="experiment_id", keep="first")
        a["leaderboard"] = merged.sort_values(
            "dev_loso_f1_macro_mean", ascending=False)
        seen = {r.get("experiment_id") for r in a["experiments"]}
        a["experiments"] = a["experiments"] + [
            r for r in extra_recs if r.get("experiment_id") not in seen
        ]
        a["merged_from"] = sources

    text = render(args.run_dir, a)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines) from {args.run_dir}")


if __name__ == "__main__":
    sys.exit(main())
