"""
DANN ablation — does subject-invariant (domain-adversarial) training improve
cross-subject cognitive-load classification?

The locked model's documented ceiling is subject variability. This runs the
decisive comparison on the SAME 26-fold LOSO protocol used everywhere else:
    A) baseline : Hybrid (attention fusion, weighted CE)          — the locked recipe
    B) DANN     : same Hybrid + gradient-reversal subject head    — subject-invariant
for the Hybrid network alone AND the Hybrid+RF ensemble (the served config),
with paired significance (B vs A) and a subject-invariance probe (can a fresh
classifier still recover subject identity from the learned features?).

Baseline and DANN are trained on IDENTICAL folds / val splits / scalers within
each fold, so the comparison is properly paired.

Outputs: outputs/dann_ablation.json
Usage:
    python dann_ablation.py                      # full 26-fold run (slow)
    python dann_ablation.py --max-folds 2 --epochs 6   # smoke test
    python dann_ablation.py --lambda-sweep 0.0 0.5 1.0 2.0 --max-folds 4
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import Config
from data_loader import CognitiveLoadDataLoader, configured_class_weights
from evaluate import predict
from models.architectures import build_model
from trainer import train_model
from part1_rigor import (
    RF_ESTIMATORS, RF_SEED, clone_config, make_loader,
    metrics_from, paired_tests, set_seed, summarize,
)
from dann import make_dann_loader, train_dann, subject_probe


def _fold_data(cfg, max_folds):
    """Yield per-fold arrays identical to part1_rigor.loso_run's protocol."""
    set_seed(cfg.RANDOM_SEED)
    dl = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, _ = dl.load_raw(return_groups=True)
    subjects = pd.read_csv(cfg.MERGED_CSV, usecols=[cfg.SUBJECT_COLUMN])[
        cfg.SUBJECT_COLUMN
    ].astype(str).to_numpy()

    for fold, (fit_idx, test_idx) in enumerate(
        LeaveOneGroupOut().split(eeg_raw, labels, subjects), 1
    ):
        if max_folds is not None and fold > max_folds:
            break
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=cfg.VAL_SIZE, random_state=cfg.RANDOM_SEED + fold
        )
        train_rel, val_rel = next(
            val_splitter.split(fit_idx, labels[fit_idx], subjects[fit_idx])
        )
        train_idx = fit_idx[train_rel]
        val_idx = fit_idx[val_rel]

        es = RobustScaler(); fsc = StandardScaler()
        eeg_tr = es.fit_transform(eeg_raw[train_idx])
        eeg_va = es.transform(eeg_raw[val_idx])
        eeg_te = es.transform(eeg_raw[test_idx])
        fn_tr = fsc.fit_transform(fnirs_raw[train_idx])
        fn_va = fsc.transform(fnirs_raw[val_idx])
        fn_te = fsc.transform(fnirs_raw[test_idx])

        # Domain (subject) index space over the 25 training subjects.
        fit_subjects = subjects[fit_idx]
        uniq = np.unique(fit_subjects)
        dom_map = {s: i for i, s in enumerate(uniq)}
        dom_tr = np.array([dom_map[s] for s in subjects[train_idx]])
        dom_va = np.array([dom_map[s] for s in subjects[val_idx]])

        yield {
            "fold": fold,
            "held": subjects[test_idx][0],
            "n_eeg": eeg_raw.shape[1], "n_fnirs": fnirs_raw.shape[1],
            "n_domains": len(uniq),
            "eeg": (eeg_tr, eeg_va, eeg_te),
            "fnirs": (fn_tr, fn_va, fn_te),
            "y": (labels[train_idx], labels[val_idx], labels[test_idx]),
            "dom": (dom_tr, dom_va),
        }


def _fit_rf(eeg_tr, fn_tr, y_tr, eeg_te, fn_te):
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, random_state=RF_SEED,
        class_weight="balanced", n_jobs=-1,
    ).fit(np.hstack([eeg_tr, fn_tr]), y_tr)
    return rf.predict_proba(np.hstack([eeg_te, fn_te]))


def loso_run_dann(cfg, device, max_folds=None, do_probe=True):
    """One pass over the LOSO folds, training BOTH baseline and DANN per fold."""
    base_folds, dann_folds, probe_rows = [], [], []
    for fd in _fold_data(cfg, max_folds):
        (eeg_tr, eeg_va, eeg_te) = fd["eeg"]
        (fn_tr, fn_va, fn_te) = fd["fnirs"]
        (y_tr, y_va, y_te) = fd["y"]
        (dom_tr, dom_va) = fd["dom"]
        n_eeg, n_fnirs = fd["n_eeg"], fd["n_fnirs"]
        meta = {
            "n_eeg_features": n_eeg, "n_fnirs_features": n_fnirs,
            "class_weights": configured_class_weights(y_tr, cfg),
        }
        te_loader = make_loader(eeg_te, fn_te, y_te, cfg)
        rf_probs = _fit_rf(eeg_tr, fn_tr, y_tr, eeg_te, fn_te)

        # ── A) baseline hybrid (locked recipe) ──
        set_seed(cfg.RANDOM_SEED)
        base = build_model("hybrid_cnn_lstm", n_eeg, n_fnirs, cfg)
        base, _ = train_model(
            base, make_loader(eeg_tr, fn_tr, y_tr, cfg, shuffle=True, drop_last=True),
            make_loader(eeg_va, fn_va, y_va, cfg), meta, cfg, verbose=False,
        )
        b_preds, b_probs, y = predict(base.to(device), te_loader, device)
        base_folds.append({"subject": fd["held"], "labels": y, "preds": b_preds,
                           "probs": b_probs, "rf_probs": rf_probs})

        # ── B) DANN hybrid (same recipe + adversarial subject head) ──
        set_seed(cfg.RANDOM_SEED)
        dann = build_model("hybrid_cnn_lstm", n_eeg, n_fnirs, cfg)
        dann, _ = train_dann(
            dann,
            make_dann_loader(eeg_tr, fn_tr, y_tr, dom_tr, cfg, shuffle=True, drop_last=True),
            make_dann_loader(eeg_va, fn_va, y_va, dom_va, cfg),
            meta, cfg, device, n_domains=fd["n_domains"], verbose=False,
        )
        d_preds, d_probs, _ = predict(dann.to(device), te_loader, device)
        dann_folds.append({"subject": fd["held"], "labels": y, "preds": d_preds,
                           "probs": d_probs, "rf_probs": rf_probs})

        # ── subject-invariance probe on training-subject embeddings ──
        if do_probe:
            pb = subject_probe(base, eeg_tr, fn_tr, dom_tr, device)
            pd_ = subject_probe(dann, eeg_tr, fn_tr, dom_tr, device)
            probe_rows.append({"subject": fd["held"],
                               "baseline_probe_acc": pb["probe_acc"],
                               "dann_probe_acc": pd_["probe_acc"],
                               "chance": pb["chance"]})

        bm = metrics_from(y, b_preds, b_probs, cfg.NUM_CLASSES)
        dm = metrics_from(y, d_preds, d_probs, cfg.NUM_CLASSES)
        pr = f" probe base={probe_rows[-1]['baseline_probe_acc']:.2f} " \
             f"dann={probe_rows[-1]['dann_probe_acc']:.2f}" if do_probe else ""
        print(f"  fold {fd['fold']:2d} subj {fd['held']:>2}: "
              f"base f1={bm['f1_macro']:.3f} | dann f1={dm['f1_macro']:.3f}{pr}")
    return base_folds, dann_folds, probe_rows


def _ensemble(folds):
    out = []
    for f in folds:
        ens = 0.5 * (f["probs"] + f["rf_probs"])
        out.append({"subject": f["subject"], "labels": f["labels"],
                    "preds": ens.argmax(1), "probs": ens})
    return out


def _f1_array(folds, ensemble=False):
    arr = []
    for f in folds:
        probs = 0.5 * (f["probs"] + f["rf_probs"]) if ensemble else f["probs"]
        arr.append(metrics_from(f["labels"], probs.argmax(1), probs)["f1_macro"])
    return np.array(arr)


def _verdict(sig):
    d, tp, wp = sig["mean_diff"], sig["t_p"], sig["wilcoxon_p"]
    if d > 0 and tp < 0.05 and wp < 0.05:
        return "DANN SIGNIFICANTLY HELPS (both tests)"
    if d > 0 and (tp < 0.05 or wp < 0.05):
        return "borderline — one test significant"
    if d < 0 and (tp < 0.05 or wp < 0.05):
        return "DANN SIGNIFICANTLY HURTS"
    return "no significant difference"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override EPOCHS and DANN_EPOCHS (smoke testing)")
    ap.add_argument("--lambda-sweep", type=float, nargs="+", default=None,
                    help="run a lambda_max sweep (mean DANN F1 per value) then exit")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--write-results", action="store_true",
                    help="append a 'Subject-Invariant Learning (DANN)' section to RESULTS.md "
                         "(use only for the full run, not smoke tests)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overrides = {"PER_SUBJECT_NORMALIZE": True, "PER_SUBJECT_NORMALIZE_STAGE": "before_log"}
    if args.epochs is not None:
        overrides["EPOCHS"] = args.epochs
        overrides["DANN_EPOCHS"] = args.epochs
        overrides["EARLY_STOP"] = max(args.epochs, 5)
    cfg = clone_config(overrides)

    # ── optional lambda sweep ──
    if args.lambda_sweep is not None:
        sweep = {}
        for lm in args.lambda_sweep:
            cfg_lm = clone_config({**overrides, "DANN_LAMBDA_MAX": lm})
            _, dfolds, _ = loso_run_dann(cfg_lm, device, args.max_folds, do_probe=False)
            s = summarize(dfolds)
            sweep[f"{lm:g}"] = {"f1_mean": s["f1_mean"], "f1_std": s["f1_std"],
                                "auc_mean": s["auc_mean"]}
            print(f"lambda_max={lm:g}: DANN hybrid F1={s['f1_mean']:.4f} +/- {s['f1_std']:.4f}")
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/dann_lambda_sweep.json", "w", encoding="utf-8") as fh:
            json.dump(sweep, fh, indent=2)
        print("\nSaved outputs/dann_lambda_sweep.json")
        return

    print(f"DANN ablation | lambda_max={cfg.DANN_LAMBDA_MAX} gamma={cfg.DANN_GAMMA}")
    base_folds, dann_folds, probe_rows = loso_run_dann(
        cfg, device, args.max_folds, do_probe=not args.no_probe)

    res = {}
    for tag, folds in (("baseline", base_folds), ("dann", dann_folds)):
        hy = summarize(folds)
        en = summarize(_ensemble(folds))
        res[tag] = {
            "hybrid": {k: hy[k] for k in ("f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean")},
            "ensemble": {k: en[k] for k in ("f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean")},
        }

    sig = {
        "hybrid_dann_vs_base": paired_tests(
            _f1_array(dann_folds), _f1_array(base_folds), "Hybrid DANN vs baseline (macro F1)"),
        "ensemble_dann_vs_base": paired_tests(
            _f1_array(dann_folds, True), _f1_array(base_folds, True),
            "Hybrid+RF DANN vs baseline (macro F1)"),
    }
    res["significance_dann_vs_baseline"] = sig
    res["conclusion"] = {"hybrid": _verdict(sig["hybrid_dann_vs_base"]),
                         "ensemble": _verdict(sig["ensemble_dann_vs_base"])}

    if probe_rows:
        bp = np.array([r["baseline_probe_acc"] for r in probe_rows], float)
        dp = np.array([r["dann_probe_acc"] for r in probe_rows], float)
        res["subject_invariance_probe"] = {
            "baseline_probe_acc_mean": float(np.nanmean(bp)),
            "dann_probe_acc_mean": float(np.nanmean(dp)),
            "chance_mean": float(np.nanmean([r["chance"] for r in probe_rows])),
            "interpretation": "lower probe accuracy = more subject-invariant; "
                              "DANN should push subject-id recovery toward chance.",
            "per_fold": probe_rows,
        }

    print("\n" + "=" * 68)
    for tag in ("baseline", "dann"):
        h = res[tag]["hybrid"]; e = res[tag]["ensemble"]
        print(f"{tag:9s} hybrid F1={h['f1_mean']:.4f} auc={h['auc_mean']:.4f} | "
              f"ensemble F1={e['f1_mean']:.4f} auc={e['auc_mean']:.4f}")
    for k, s in sig.items():
        print(f"{k}: diff={s['mean_diff']:+.4f} t_p={s['t_p']:.3f} wilcoxon_p={s['wilcoxon_p']:.3f}")
    print("conclusion:", res["conclusion"])
    if "subject_invariance_probe" in res:
        p = res["subject_invariance_probe"]
        print(f"probe: baseline={p['baseline_probe_acc_mean']:.3f} "
              f"dann={p['dann_probe_acc_mean']:.3f} chance={p['chance_mean']:.3f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/dann_ablation.json", "w", encoding="utf-8") as fh:
        json.dump(_clean(res), fh, indent=2)
    print("\nSaved outputs/dann_ablation.json")

    if args.write_results:
        _write_results_md(res)
        print("Appended 'Subject-Invariant Learning (DANN)' to RESULTS.md")


def _decision(sig_ens):
    """Adopt only on a clear, significant ensemble lift (both tests) — the same
    gate discipline as the HbR and classical-sweep sections."""
    d, tp, wp = sig_ens["mean_diff"], sig_ens["t_p"], sig_ens["wilcoxon_p"]
    if d > 0 and tp < 0.05 and wp < 0.05:
        return ("ADOPTED",
                "DANN significantly improves the served ensemble on both paired tests; "
                "re-export serving via export_for_serving.py on a DANN-trained checkpoint.")
    return ("NOT ADOPTED",
            "DANN did not deliver a clear, significant ensemble-level improvement under "
            "the decision gate; the locked model is retained. Reported as an honest, "
            "well-analyzed result (see the subject-invariance probe for mechanism evidence).")


def _write_results_md(res):
    from datetime import datetime
    b, d = res["baseline"], res["dann"]
    sh = res["significance_dann_vs_baseline"]["hybrid_dann_vs_base"]
    se = res["significance_dann_vs_baseline"]["ensemble_dann_vs_base"]
    verdict, reason = _decision(se)
    lines = [
        "",
        "## Subject-Invariant Learning (DANN)",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "Scripts: `dann.py`, `dann_ablation.py`. Artifact: `outputs/dann_ablation.json`. "
        "Writeup: `NOVELTY_DANN.md`.",
        "",
        "Domain-adversarial training (gradient-reversal subject discriminator) directly "
        "attacks this project's documented ceiling — subject variability — by making the "
        "fused representation predict cognitive LOAD but not WHO. Same 26-fold LOSO "
        "protocol, same per-subject-normalized features and per-fold train-only scalers; "
        "baseline and DANN trained on identical folds for a paired comparison.",
        "",
        "**LOSO results (26 folds):**",
        "",
        "| Config | Hybrid Macro F1 | Hybrid AUC | Ensemble Macro F1 | Ensemble AUC |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline (locked) | {b['hybrid']['f1_mean']:.4f} | {b['hybrid']['auc_mean']:.4f} "
        f"| {b['ensemble']['f1_mean']:.4f} | {b['ensemble']['auc_mean']:.4f} |",
        f"| DANN | {d['hybrid']['f1_mean']:.4f} | {d['hybrid']['auc_mean']:.4f} "
        f"| {d['ensemble']['f1_mean']:.4f} | {d['ensemble']['auc_mean']:.4f} |",
        "",
        "**Paired significance (DANN vs baseline, macro F1):**",
        "",
        "| Comparison | mean diff | paired t p | Wilcoxon p |",
        "|---|---:|---:|---:|",
        f"| Hybrid-alone | {sh['mean_diff']:+.4f} | {sh['t_p']:.4f} | {sh['wilcoxon_p']:.4f} |",
        f"| Ensemble (served) | {se['mean_diff']:+.4f} | {se['t_p']:.4f} | {se['wilcoxon_p']:.4f} |",
        "",
    ]
    if "subject_invariance_probe" in res:
        p = res["subject_invariance_probe"]
        lines += [
            "**Subject-invariance probe** (can a fresh classifier still recover subject "
            "identity from the learned features? lower = more invariant):",
            "",
            f"- Baseline probe accuracy: {p['baseline_probe_acc_mean']:.3f}",
            f"- DANN probe accuracy: {p['dann_probe_acc_mean']:.3f}",
            f"- Chance: {p['chance_mean']:.3f}",
            "",
        ]
    lines += [
        f"**Decision: {verdict}.** {reason}",
        "",
    ]
    with open("RESULTS.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


if __name__ == "__main__":
    main()
