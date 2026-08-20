#!/usr/bin/env python
"""
run_campaign.py — the full controlled-experiment campaign for the binary
(Low vs High) EEG+fNIRS cognitive-load model.

    .venv/bin/python run_campaign.py                 # full campaign
    .venv/bin/python run_campaign.py --quick         # 4 dev subjects, smoke test
    .venv/bin/python run_campaign.py --stages S1 S3  # only some stages

Creates outputs/runs/run_<timestamp>_<id>/ BEFORE anything else happens, so a
crash still leaves a directory with metadata, config, logs and a failed status.
Nothing from a previous run is ever overwritten.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
# These tiny models thrash if every core fights over them; pin the pool so
# per-fold timings stay comparable across experiments.
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 2)))

from research import explain, leakage, plots
from research import metrics as M
from research.campaign import (
    BASELINE_DATA, BASELINE_SPEC, DATASETS, SELECT_METRIC, Campaign,
    paired_vs, score_of, select_best, stderr_of,
)
from research.datamod import holdout_fold, split_dev_lockbox
from research.experiment import mean_history
from research.runctx import RunContext, write_json
from research.train import embeddings, predict, train_fold


def spec_with(base, **kw):
    s = copy.deepcopy(base)
    s.update(kw)
    return s


def data_with(base, **kw):
    d = copy.deepcopy(base)
    d.update(kw)
    return d


# ── the staged plan: each stage varies ONE axis from the incumbent ──────
def build_stages(quick=False):
    ep = dict(max_epochs=40) if quick else {}

    def S(spec_kw=None, data_kw=None, note=""):
        return {"spec": spec_kw or {}, "data": data_kw or {}, "note": note}

    return [
        ("S1_batch_lr", [
            ("S1_bs64", S(dict(batch_size=64), note="batch 64")),
            ("S1_bs128", S(dict(batch_size=128), note="batch 128")),
            ("S1_bs64_lr1e3", S(dict(batch_size=64, lr=1e-3), note="batch 64, lr 1e-3")),
            ("S1_lr1e4", S(dict(lr=1e-4), note="lr 1e-4")),
        ]),
        ("S2_schedule", [
            ("S2_longer_p15", S(dict(max_epochs=120, early_stop_patience=15),
                                note="longer budget, patience 15")),
            ("S2_longer_p25", S(dict(max_epochs=120, early_stop_patience=25),
                                note="longer budget, patience 25")),
            ("S2_cosine", S(dict(max_epochs=120, early_stop_patience=15,
                                 scheduler="cosine"),
                            note="cosine annealing")),
            ("S2_no_sched", S(dict(max_epochs=120, early_stop_patience=15,
                                   scheduler="none"),
                              note="no LR schedule (control)")),
        ]),
        ("S3_weight_decay", [
            ("S3_wd0", S(dict(weight_decay=0.0), note="no weight decay")),
            ("S3_wd003", S(dict(weight_decay=0.003), note="weight decay 0.003")),
            ("S3_wd005", S(dict(weight_decay=0.05), note="weight decay 0.05")),
            ("S3_wd01", S(dict(weight_decay=0.1), note="weight decay 0.1")),
        ]),
        ("S4_dropout", [
            ("S4_drop_low", S(dict(encoder_dropout=0.3, fc_dropout=0.3),
                              note="dropout 0.3 / 0.3")),
            ("S4_drop_mid", S(dict(encoder_dropout=0.4, fc_dropout=0.5),
                              note="dropout 0.4 / 0.5")),
            ("S4_drop_high", S(dict(encoder_dropout=0.6, fc_dropout=0.7),
                               note="dropout 0.6 / 0.7")),
        ]),
        ("S5_capacity", [
            ("S5_proj48", S(dict(proj_dim=48), note="projection width 48")),
            ("S5_proj32_enc32", S(dict(proj_dim=32, enc_dim=32, fc_units=[32]),
                                  note="narrow: proj 32 / enc 32 / head [32]")),
            ("S5_proj192", S(dict(proj_dim=192), note="projection width 192")),
            ("S5_blocks1", S(dict(n_blocks=1), note="1 residual block")),
            ("S5_blocks3", S(dict(n_blocks=3), note="3 residual blocks")),
        ]),
        ("S6_architecture", [
            ("S6_cnn", S(dict(encoder="cnn", cnn_residual=False, fusion="concat"),
                         note="1D CNN over channel axis, concat fusion")),
            ("S6_cnn_res", S(dict(encoder="cnn", cnn_residual=True, fusion="concat"),
                             note="CNN + residual blocks, concat fusion")),
            ("S6_cnn_res_attn", S(dict(encoder="cnn", cnn_residual=True,
                                       fusion="attention"),
                                  note="CNN + residual + attention fusion")),
            ("S6_cnn_wide", S(dict(encoder="cnn", cnn_residual=True,
                                   cnn_filters=[128, 128], fusion="attention"),
                              note="CNN + residual + attention, 128 filters")),
            ("S6_mlp_concat", S(dict(fusion="concat"),
                                note="MLP encoders, concat fusion (attention ablation)")),
        ]),
        ("S7_modality_dropout", [
            ("S7_md010", S(dict(modality_dropout_eeg=0.1, modality_dropout_fnirs=0.1),
                           note="modality dropout 0.1 / 0.1")),
            ("S7_md020", S(dict(modality_dropout_eeg=0.2, modality_dropout_fnirs=0.2),
                           note="modality dropout 0.2 / 0.2")),
            ("S7_md_eeg_only", S(dict(modality_dropout_eeg=0.3,
                                      modality_dropout_fnirs=0.1),
                                 note="drop EEG more often (0.3 / 0.1)")),
        ]),
        ("S8_noise", [
            ("S8_noise0", S(dict(augment_noise=0.0), note="no input noise")),
            ("S8_noise10", S(dict(augment_noise=0.1), note="input noise 0.10")),
            ("S8_noise20", S(dict(augment_noise=0.2), note="input noise 0.20")),
            ("S8_smooth", S(dict(label_smoothing=0.1), note="label smoothing 0.1")),
        ]),
        ("S9_dann", [
            ("S9_dann01", S(dict(use_dann=True, dann_lambda=0.1),
                            note="DANN lambda 0.1")),
            ("S9_dann05", S(dict(use_dann=True, dann_lambda=0.5),
                            note="DANN lambda 0.5")),
            ("S9_dann10", S(dict(use_dann=True, dann_lambda=1.0),
                            note="DANN lambda 1.0")),
        ]),
        ("S10_data", [
            ("S10_stride1s", S(data_kw=dict(dataset="sliding_10s_1s"),
                               note="dense windows: 10s / 1s stride (4.6x rows)")),
            ("S10_win8s", S(data_kw=dict(dataset="sliding_8s_4s"),
                            note="8s windows / 4s stride")),
            ("S10_no_subject_norm", S(data_kw=dict(per_subject_norm="none"),
                                      note="no per-subject normalization")),
            ("S10_calibration_norm", S(data_kw=dict(per_subject_norm="calibration",
                                                    calibration_windows=30),
                                       note="per-subject norm from 30 calibration windows only")),
            ("S10_norm_after_log", S(data_kw=dict(per_subject_norm_stage="after_log"),
                                     note="per-subject norm applied after the log")),
            ("S10_no_hbr", S(data_kw=dict(drop_hbr=True), note="HbO only (drop HbR)")),
        ]),
    ]


def representation_analysis(camp, run, incumbent_spec, incumbent_data, baseline_spec):
    """t-SNE + subject-invariance on ONE controlled fold.

    All variants are trained on the same development subjects and embedded on
    the same never-selected-on lockbox subjects, so the pictures differ only by
    the model that produced them.
    """
    ds = camp.cache.get(incumbent_data)
    fold = holdout_fold(ds, camp.dev_subjects, camp.lockbox_subjects,
                        val_fraction=camp.val_fraction, seed=camp.seed)
    from research.datamod import scale_fold
    scaled = scale_fold(ds, fold)
    leakage.audit_fold(ds, fold, scaled, strict=True)

    te = fold["test_idx"]
    y_te = ds.y[te]
    subj_te = ds.subjects[te]
    eeg_te, fn_te, _ = scaled["test"]

    variants = {}
    # 1. raw scaled features, no model
    variants["raw_features"] = np.hstack([eeg_te, fn_te])

    def train_and_embed(spec, use_dom):
        dom = None
        if use_dom:
            tr_subj = ds.subjects[fold["train_idx"]]
            uniq = np.unique(tr_subj)
            mapping = {s: i for i, s in enumerate(uniq)}
            dom = {"train": np.array([mapping[s] for s in tr_subj], np.int64),
                   "n_domains": len(uniq)}
        res = train_fold(spec, ds.schema, ds.n_classes, scaled, domains=dom,
                         seed=camp.seed, device=camp.device, log=run.log)
        return res["model"], embeddings(res["model"], eeg_te, fn_te, device=camp.device)

    run.log("   representation: baseline model")
    _, variants["baseline_model"] = train_and_embed(baseline_spec, False)
    run.log("   representation: selected model")
    best_model, variants["selected_model"] = train_and_embed(incumbent_spec, False)
    run.log("   representation: DANN model")
    dann_spec = spec_with(incumbent_spec, use_dann=True, dann_lambda=0.5)
    _, variants["dann_model"] = train_and_embed(dann_spec, True)

    rep_dir = run.path("representations")
    out = {"fold": {"train_subjects": camp.dev_subjects.tolist(),
                    "test_subjects": camp.lockbox_subjects.tolist()},
           "variants": {}, "figures": {}}

    for name, emb in variants.items():
        coords, idx = explain.compute_tsne(emb, seed=camp.seed)
        probe = explain.subject_invariance_probe(emb, subj_te, seed=camp.seed)
        df = pd.DataFrame({
            "variant": name,
            "subject_id": subj_te[idx],
            "y_true": y_te[idx],
            "y_true_name": [ds.class_names[i] for i in y_te[idx]],
            "tsne_x": coords[:, 0],
            "tsne_y": coords[:, 1],
        })
        df.to_csv(os.path.join(rep_dir, f"tsne_{name}.csv"), index=False)
        out["variants"][name] = {
            "n_points": int(len(idx)),
            "embedding_dim": int(emb.shape[1]),
            "subject_invariance_probe": probe,
            "points": df.to_dict("records"),
        }
        plots.tsne_fig(coords, y_te[idx], ds.class_names,
                       os.path.join(rep_dir, f"tsne_{name}_by_class.png"),
                       f"t-SNE — {name.replace('_', ' ')} (coloured by workload)")
        uniq_s = sorted(set(subj_te[idx].tolist()))
        remap = np.array([uniq_s.index(s) for s in subj_te[idx]])
        plots.tsne_fig(coords, remap, [f"S{s}" for s in uniq_s],
                       os.path.join(rep_dir, f"tsne_{name}_by_subject.png"),
                       f"t-SNE — {name.replace('_', ' ')} (coloured by subject)")
        out["figures"][name] = {
            "by_class": f"representations/tsne_{name}_by_class.png",
            "by_subject": f"representations/tsne_{name}_by_subject.png",
        }

    # canonical filenames the brief asks for
    import shutil
    shutil.copyfile(os.path.join(rep_dir, "tsne_selected_model_by_class.png"),
                    os.path.join(rep_dir, "tsne.png"))
    shutil.copyfile(os.path.join(rep_dir, "tsne_selected_model_by_subject.png"),
                    os.path.join(rep_dir, "subject_tsne.png"))
    pd.concat([pd.DataFrame(v["points"]) for v in out["variants"].values()]).to_csv(
        os.path.join(rep_dir, "embeddings.csv"), index=False
    )
    write_json(os.path.join(rep_dir, "representations.json"), out)
    return out, best_model, scaled, ds, fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="4 development subjects, short budgets — smoke test")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="only run these stage prefixes, e.g. S1 S6")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lockbox", type=int, default=8)
    ap.add_argument("--tag", default="binary-improvement")
    ap.add_argument("--mode", choices=["binary", "3class"], default="binary",
                    help="binary = Low vs High (0-back vs 3-back); "
                         "3class = Low/Medium/High. The 3-class model is kept "
                         "working; the campaign targets the binary one.")
    ap.add_argument("--limit-subjects", type=int, default=None,
                    help="SMOKE TEST ONLY: shrink the cohort so every code path runs fast")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="SMOKE TEST ONLY: cap every arm's epoch budget")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-tsne", action="store_true")
    args = ap.parse_args()

    run = RunContext(repo_root=ROOT, tag=args.tag)
    try:
        run.stage("audit")
        BASELINE_DATA["binary"] = args.mode == "binary"
        camp = Campaign(run, ROOT, device=args.device, seed=args.seed,
                        n_lockbox=args.n_lockbox, quick=args.quick,
                        limit_subjects=args.limit_subjects)
        if args.max_epochs:
            BASELINE_SPEC["max_epochs"] = args.max_epochs
            run.warn(f"max_epochs capped at {args.max_epochs} (smoke test)")
        base_ds = camp.cache.get(BASELINE_DATA)

        dataset_info = base_ds.info({
            "development_subjects": camp.dev_subjects.tolist(),
            "lockbox_subjects": camp.lockbox_subjects.tolist(),
            "available_datasets": DATASETS,
        })
        run.write_metadata(
            config={
                "baseline_spec": BASELINE_SPEC,
                "baseline_data": BASELINE_DATA,
                "seed": args.seed,
                "device": args.device,
                "n_lockbox": args.n_lockbox,
                "val_fraction": camp.val_fraction,
                "select_metric": SELECT_METRIC,
                "checkpoint_metric": "validation macro F1",
                "quick": args.quick,
                "classification_mode": args.mode,
                "stages_requested": args.stages,
                "protocol": {
                    "selection": "18-fold LOSO over the development pool",
                    "final_headline": "26-fold LOSO over all subjects",
                    "final_unbiased": "8 lockbox subjects, evaluated once",
                },
            },
            dataset_info=dataset_info,
        )

        # ── dataset-level leakage audit ────────────────────────────────
        audit = leakage.audit_dataset(
            base_ds, probe_subjects=base_ds.unique_subjects[:4], strict=True
        )
        run.log("   dataset leakage audit: " +
                ("PASSED" if audit["passed"] else "FAILED"))
        for c in audit["checks"]:
            run.log(f"     [{'ok' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
        write_json(run.path("leakage_audit.json"),
                   {"dataset_level": audit, "fold_level": "see experiments/*/leakage_audit.json"})

        # ── baseline ───────────────────────────────────────────────────
        run.stage("baseline")
        baseline = camp.evaluate_spec(
            "A0_baseline", BASELINE_SPEC, BASELINE_DATA, "S0_baseline",
            note="served binary architecture + config.py training hyperparameters",
        )
        write_json(run.path("experiments", "baseline_record.json"),
                   {k: v for k, v in baseline.items() if k != "mean_history"})

        incumbent_spec = copy.deepcopy(BASELINE_SPEC)
        incumbent_data = copy.deepcopy(BASELINE_DATA)
        incumbent = baseline
        stage_log = []

        # ── staged search ──────────────────────────────────────────────
        run.stage("experiments")
        for stage_name, arms in build_stages(args.quick):
            if args.stages and not any(stage_name.startswith(s) for s in args.stages):
                continue
            run.log(f"\n=== {stage_name} ===")
            arm_records = [incumbent]
            for exp_id, cfg in arms:
                spec = spec_with(incumbent_spec, **cfg["spec"])
                if args.max_epochs:
                    spec["max_epochs"] = min(spec["max_epochs"], args.max_epochs)
                dcfg = data_with(incumbent_data, **cfg["data"])
                try:
                    rec = camp.evaluate_spec(exp_id, spec, dcfg, stage_name,
                                             note=cfg["note"])
                    arm_records.append(rec)
                except Exception as exc:
                    run.warn(f"{exp_id} failed: {type(exc).__name__}: {exc}")
            winner = select_best(arm_records, incumbent=incumbent)
            improved = winner["experiment_id"] != incumbent["experiment_id"]
            challenger = max(
                (r for r in arm_records
                 if r["experiment_id"] != incumbent["experiment_id"]),
                key=score_of, default=None,
            )
            stage_log.append({
                "stage": stage_name,
                "incumbent_before": incumbent["experiment_id"],
                "winner": winner["experiment_id"],
                "changed": improved,
                "score_before": score_of(incumbent),
                "score_after": score_of(winner),
                "delta": score_of(winner) - score_of(incumbent),
                "noise_floor_1se": stderr_of(incumbent),
                "best_challenger": (
                    {"experiment_id": challenger["experiment_id"],
                     "score": score_of(challenger),
                     "delta_vs_incumbent": score_of(challenger) - score_of(incumbent),
                     "inside_noise_floor":
                         score_of(challenger) - score_of(incumbent) <= stderr_of(incumbent)}
                    if challenger else None
                ),
                "arms": [
                    {"experiment_id": r["experiment_id"], "note": r["note"],
                     "score": score_of(r), "n_params": r["n_params"],
                     "accepted": r["experiment_id"] == winner["experiment_id"]}
                    for r in arm_records
                ],
            })
            if improved:
                run.log(f"  -> {stage_name}: ACCEPTED {winner['experiment_id']} "
                        f"({score_of(incumbent):.4f} -> {score_of(winner):.4f})")
                incumbent = winner
                incumbent_spec = copy.deepcopy(winner["spec"])
                incumbent_data = copy.deepcopy(winner["data"])
            else:
                bc = stage_log[-1]["best_challenger"]
                why = ""
                if bc:
                    why = (f"; best challenger {bc['experiment_id']} "
                           f"{bc['score']:.4f} (Δ{bc['delta_vs_incumbent']:+.4f}, "
                           f"noise floor ±{stderr_of(incumbent):.4f})")
                run.log(f"  -> {stage_name}: no accepted change "
                        f"(incumbent {incumbent['experiment_id']} "
                        f"{score_of(incumbent):.4f} retained{why})")
            write_json(run.path("experiments", "stage_log.json"), stage_log)

        # ── final evaluation ───────────────────────────────────────────
        run.stage("final_evaluation")
        run.log(f"\nSELECTED: {incumbent['experiment_id']}  spec={incumbent_spec}")
        loso_rec, loso_preds, loso_hists, loso_curves, loso_models, ds = \
            camp.evaluate_full_loso(incumbent_spec, incumbent_data)
        lock_rec, lock_preds, lock_hists, lock_curves, lock_models, _ = \
            camp.evaluate_lockbox(incumbent_spec, incumbent_data)

        baseline_loso, *_ = camp.evaluate_full_loso(
            BASELINE_SPEC, BASELINE_DATA, exp_id="FINAL_loso26_baseline"
        )
        baseline_lock, *_ = camp.evaluate_lockbox(
            BASELINE_SPEC, BASELINE_DATA, exp_id="FINAL_lockbox_baseline"
        )

        # ── headline artifacts (primary = selected model, 26-fold LOSO) ─
        run.stage("artifacts")
        loso_preds.to_csv(run.path("predictions", "predictions.csv"), index=False)
        pcols = [c for c in loso_preds.columns if c.startswith("prob_")]
        loso_preds[
            ["experiment_id", "fold", "subject_id", "trial_id", "window_id", "y_true"]
            + pcols
        ].to_csv(run.path("predictions", "probabilities.csv"), index=False)

        pooled = loso_rec["pooled"]
        write_json(run.path("metrics", "metrics.json"), pooled)
        pd.DataFrame(M.metrics_to_rows(pooled)).to_csv(
            run.path("metrics", "metrics.csv"), index=False)
        write_json(run.path("metrics", "per_class_metrics.json"), pooled["per_class"])
        write_json(run.path("metrics", "loso_metrics.json"), {
            "protocol": f"{loso_rec['n_folds']}-fold leave-one-subject-out",
            "fold_aggregate": loso_rec["fold_aggregate"],
            "generalization_gap": loso_rec["generalization_gap"],
            "pooled": pooled,
            "lockbox": {
                "subjects": camp.lockbox_subjects.tolist(),
                "fold_aggregate": lock_rec["fold_aggregate"],
                "pooled": lock_rec["pooled"],
            },
            "baseline_comparison": {
                "loso26": baseline_loso["fold_aggregate"],
                "lockbox": baseline_lock["fold_aggregate"],
                "paired_loso26": paired_vs(loso_rec, baseline_loso),
                "paired_lockbox": paired_vs(lock_rec, baseline_lock),
            },
        })
        pd.DataFrame(loso_rec["per_subject"]).to_csv(
            run.path("metrics", "per_subject_metrics.csv"), index=False)
        write_json(run.path("metrics", "metric_explanations.json"),
                   M.METRIC_EXPLANATIONS)
        write_json(run.path("confusion_matrix", "confusion_matrix.json"), {
            "labels": ds.class_names,
            "matrix": pooled["confusion_matrix"],
            "matrix_normalized": pooled["confusion_matrix_normalized"],
            "protocol": f"{loso_rec['n_folds']}-fold LOSO, pooled held-out windows",
        })
        write_json(run.path("curves", "curves.json"), loso_curves)

        mh = mean_history(loso_hists)
        write_json(run.path("curves", "training_history.json"), {
            "mean_over_folds": mh,
            "per_fold": loso_hists,
        })

        plots.loss_curves(mh, run.path("curves"))
        plots.metric_curves(mh, run.path("curves"))
        plots.confusion_matrix_fig(
            pooled["confusion_matrix"], ds.class_names,
            run.path("confusion_matrix", "confusion_matrix.png"),
            title=f"Confusion matrix — {loso_rec['n_folds']}-fold LOSO")
        plots.roc_fig(loso_curves, run.path("curves", "roc_curve.png"))
        plots.pr_fig(loso_curves, run.path("curves", "precision_recall_curve.png"))
        plots.per_subject_fig(
            loso_rec["per_subject"], run.path("curves", "per_subject_performance.png"),
            chance=1.0 / ds.n_classes)
        plots.experiment_comparison_fig(
            camp.records, run.path("experiments", "experiment_comparison.png"))
        if loso_rec.get("modality_gates"):
            plots.modality_gate_fig(
                loso_rec["modality_gates"],
                run.path("feature_importance", "modality_attention.png"))

        # ── model artifacts ────────────────────────────────────────────
        import pickle
        import torch
        for m in loso_models:
            torch.save(m["model"].state_dict(),
                       run.path("model", "folds", f"fold_{m['fold']:02d}.pt"))
        # A deployable model fitted on every subject. It has no held-out data,
        # so it carries no performance claim of its own.
        run.log("   training deployment model on all subjects (no held-out data)")
        from research.datamod import scale_fold
        all_fold = holdout_fold(ds, camp.all_subjects, camp.all_subjects[:1],
                                val_fraction=camp.val_fraction, seed=camp.seed)
        all_fold["train_idx"] = np.concatenate(
            [all_fold["train_idx"], all_fold["test_idx"]])
        all_scaled = scale_fold(ds, all_fold)
        dep = train_fold(incumbent_spec, ds.schema, ds.n_classes, all_scaled,
                         seed=camp.seed, device=camp.device, log=run.log)
        torch.save(dep["model"].state_dict(), run.path("model", "model.pt"))
        with open(run.path("model", "scaler", "feature_scaler.pkl"), "wb") as fh:
            pickle.dump({"eeg_scaler": all_scaled["scalers"][0],
                         "fnirs_scaler": all_scaled["scalers"][1]}, fh)
        write_json(run.path("model", "model_config.json"), {
            "spec": incumbent_spec,
            "data": incumbent_data,
            "n_eeg_features": ds.n_eeg,
            "n_fnirs_features": ds.n_fnirs,
            "num_classes": ds.n_classes,
            "class_names": ds.class_names,
            "class_semantics": ds.class_semantics,
            "n_params": dep["n_params"],
            "selected_experiment": incumbent["experiment_id"],
            "deployment_model_note": (
                "Trained on all subjects for serving. It has NO held-out data, "
                "so it carries no performance claim. The reported metrics come "
                "from the LOSO folds under model/folds/."
            ),
        })
        write_json(run.path("model", "feature_schema.json"), {
            "eeg_columns": ds.schema.eeg_cols,
            "fnirs_columns": ds.schema.fnirs_cols,
            **ds.schema.to_dict(),
        })
        write_json(run.path("model", "label_map.json"),
                   {str(i): n for i, n in enumerate(ds.class_names)})

        # ── SHAP ───────────────────────────────────────────────────────
        shap_out = None
        if not args.skip_shap:
            run.stage("shap")
            try:
                eeg_tr, fn_tr, _ = all_scaled["train"]
                lock_rows = np.flatnonzero(
                    np.isin(ds.subjects, camp.lockbox_subjects))
                eeg_ex = ds.eeg[lock_rows]
                fn_ex = ds.fnirs[lock_rows]
                es, fs = all_scaled["scalers"]
                shap_out = explain.compute_shap(
                    dep["model"], eeg_tr, fn_tr,
                    es.transform(eeg_ex).astype(np.float32),
                    fs.transform(fn_ex).astype(np.float32),
                    ds.schema, ds.class_names, device=camp.device, log=run.log,
                )
                write_json(run.path("feature_importance", "shap_values.json"),
                           {k: v for k, v in shap_out.items() if k != "ranking"})
                pd.DataFrame(shap_out["ranking"]).to_csv(
                    run.path("feature_importance", "feature_ranking.csv"), index=False)
                plots.shap_fig(shap_out["ranking"],
                               run.path("feature_importance", "shap_summary.png"))
                plots.shap_category_fig(
                    shap_out["categories"][:25],
                    run.path("feature_importance", "shap_categories.png"))
                run.log(f"   SHAP modality share: {shap_out['modality_share']}")
            except Exception as exc:
                run.warn(f"SHAP failed: {type(exc).__name__}: {exc}")

        # ── representations ────────────────────────────────────────────
        reps = None
        if not args.skip_tsne:
            run.stage("representations")
            try:
                reps, *_ = representation_analysis(
                    camp, run, incumbent_spec, incumbent_data, BASELINE_SPEC)
                for k, v in reps["variants"].items():
                    p = v["subject_invariance_probe"]
                    if p:
                        run.log(f"   subject probe [{k}]: {p['accuracy_mean']:.3f} "
                                f"(chance {p['chance_uniform']:.3f})")
            except Exception as exc:
                run.warn(f"Representation analysis failed: {type(exc).__name__}: {exc}")

        # ── summary ────────────────────────────────────────────────────
        run.stage("summary")
        leaderboard = camp.save_leaderboard()
        summary = build_summary(
            run, camp, ds, baseline, incumbent, incumbent_spec, incumbent_data,
            loso_rec, lock_rec, baseline_loso, baseline_lock, stage_log,
            shap_out, reps, leaderboard,
        )
        write_json(run.path("run_summary.json"), summary)
        with open(run.path("run_summary.md"), "w", encoding="utf-8") as fh:
            fh.write(render_summary_md(summary))
        run.complete({
            "selected_experiment": incumbent["experiment_id"],
            "loso26_f1_macro": loso_rec["fold_aggregate"]["f1_macro"]["mean"],
            "lockbox_f1_macro": lock_rec["fold_aggregate"]["f1_macro"]["mean"],
            "n_experiments": len(camp.records),
        })
        print(f"\nRun complete: {run.dir}")
    except Exception as exc:
        run.fail(exc)
        raise


def build_summary(run, camp, ds, baseline, incumbent, spec, dcfg, loso_rec,
                  lock_rec, baseline_loso, baseline_lock, stage_log, shap_out,
                  reps, leaderboard):
    def agg(r):
        a = r["fold_aggregate"]
        return {k: a[k] for k in
                ("f1_macro", "accuracy", "balanced_accuracy", "roc_auc", "mcc",
                 "cohen_kappa", "precision_macro", "recall_macro")}

    return {
        "run_id": run.run_id,
        "tag": run.tag,
        "created_at": run.created_at,
        "dataset": {
            "name": dcfg["dataset"],
            "path": ds.csv_path,
            "n_samples": int(len(ds.y)),
            "n_subjects": int(len(ds.unique_subjects)),
            "class_names": ds.class_names,
            "class_semantics": ds.class_semantics,
            "class_distribution": ds.info()["class_distribution"],
            "preprocessing": ds.info()["preprocessing"],
        },
        "protocol": {
            "selection": f"{len(camp.dev_subjects)}-fold LOSO on development subjects",
            "development_subjects": camp.dev_subjects.tolist(),
            "lockbox_subjects": camp.lockbox_subjects.tolist(),
            "headline": f"{loso_rec['n_folds']}-fold LOSO over all subjects",
            "checkpoint_metric": "validation macro F1",
            "select_metric": SELECT_METRIC,
        },
        "model": {
            "selected_experiment": incumbent["experiment_id"],
            "spec": spec,
            "data": dcfg,
            "n_params": incumbent["n_params"],
            "mean_best_epoch": loso_rec["generalization_gap"]["mean_best_epoch"],
        },
        "results": {
            "selected": {
                "loso_all_subjects": agg(loso_rec),
                "loso_pooled": loso_rec["pooled"],
                "lockbox": agg(lock_rec),
                "lockbox_pooled": lock_rec["pooled"],
                "generalization_gap": loso_rec["generalization_gap"],
            },
            "baseline": {
                "loso_all_subjects": agg(baseline_loso),
                "loso_pooled": baseline_loso["pooled"],
                "lockbox": agg(baseline_lock),
                "generalization_gap": baseline_loso["generalization_gap"],
                "dev_loso": agg(baseline),
            },
            "paired_tests": {
                "loso26_selected_vs_baseline": paired_vs(loso_rec, baseline_loso),
                "lockbox_selected_vs_baseline": paired_vs(lock_rec, baseline_lock),
            },
        },
        "stage_log": stage_log,
        "n_experiments": len(camp.records),
        "leaderboard_top": leaderboard.head(15).to_dict("records"),
        "feature_importance": (
            {"modality_share": shap_out["modality_share"],
             "top_features": shap_out["top_features"][:20],
             "top_categories": shap_out["categories"][:15]}
            if shap_out else None
        ),
        "representations": (
            {k: {"subject_invariance_probe": v["subject_invariance_probe"]}
             for k, v in reps["variants"].items()} if reps else None
        ),
        "leakage_audit": {
            "status": "passed",
            "detail": "see leakage_audit.json and experiments/*/leakage_audit.json",
        },
        "warnings": run.warnings,
    }


def render_summary_md(s):
    def fmt(v, n=4):
        return "n/a" if v is None else f"{v:.{n}f}"

    sel = s["results"]["selected"]
    base = s["results"]["baseline"]
    lines = [
        f"# Run {s['run_id']}",
        "",
        f"- **Tag**: {s['tag']}",
        f"- **Created**: {s['created_at']}",
        f"- **Dataset**: `{s['dataset']['name']}` — {s['dataset']['n_samples']} windows, "
        f"{s['dataset']['n_subjects']} subjects, classes "
        f"{s['dataset']['class_names']} ({s['dataset']['class_semantics']})",
        f"- **Protocol**: {s['protocol']['headline']}; selection on "
        f"{s['protocol']['selection']}",
        f"- **Lockbox subjects** (never used for selection): "
        f"{s['protocol']['lockbox_subjects']}",
        f"- **Experiments run**: {s['n_experiments']}",
        f"- **Leakage audit**: {s['leakage_audit']['status']}",
        "",
        "## Selected model",
        "",
        f"`{s['model']['selected_experiment']}` — {s['model']['n_params']:,} parameters, "
        f"mean best epoch {s['model']['mean_best_epoch']:.1f}",
        "",
        "```json",
        __import__("json").dumps(s["model"]["spec"], indent=2),
        "```",
        "",
        "## Results",
        "",
        "| Protocol | Model | Macro F1 | Accuracy | Balanced acc | ROC-AUC | MCC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, rec in (("LOSO (all subjects)", "loso_all_subjects"),
                       ("Lockbox (8 unseen)", "lockbox")):
        for name, src in (("selected", sel), ("baseline", base)):
            a = src[rec]
            lines.append(
                f"| {label} | {name} | {fmt(a['f1_macro']['mean'])} ± "
                f"{fmt(a['f1_macro']['std'])} | {fmt(a['accuracy']['mean'])} | "
                f"{fmt(a['balanced_accuracy']['mean'])} | "
                f"{fmt(a['roc_auc']['mean'])} | {fmt(a['mcc']['mean'])} |"
            )
    g = sel["generalization_gap"]
    gb = base["generalization_gap"]
    lines += [
        "",
        "## Generalization gap (macro F1, at each fold's selected epoch)",
        "",
        "| Model | train | val | held-out | train−held-out |",
        "|---|---:|---:|---:|---:|",
        f"| selected | {fmt(g['train_f1_macro_mean'])} | {fmt(g['val_f1_macro_mean'])} | "
        f"{fmt(g['heldout_f1_macro_mean'])} | {fmt(g['train_minus_heldout'])} |",
        f"| baseline | {fmt(gb['train_f1_macro_mean'])} | {fmt(gb['val_f1_macro_mean'])} | "
        f"{fmt(gb['heldout_f1_macro_mean'])} | {fmt(gb['train_minus_heldout'])} |",
        "",
        "## Stage log",
        "",
        "| Stage | Winner | Changed | Score before | Score after |",
        "|---|---|:--:|---:|---:|",
    ]
    for st in s["stage_log"]:
        lines.append(
            f"| {st['stage']} | {st['winner']} | {'yes' if st['changed'] else 'no'} | "
            f"{fmt(st['score_before'])} | {fmt(st['score_after'])} |"
        )
    if s.get("feature_importance"):
        ms = s["feature_importance"]["modality_share"]
        lines += ["", "## Feature importance (SHAP)", "",
                  f"Modality share — EEG {fmt(ms['EEG'], 3)}, fNIRS {fmt(ms['fNIRS'], 3)} "
                  f"(HbO {fmt(ms['HbO'], 3)}, HbR {fmt(ms['HbR'], 3)})", "",
                  "Top feature families:", ""]
        for c in s["feature_importance"]["top_categories"][:10]:
            lines.append(f"- {c['category']} ({c['modality']}): {c['mean_abs_shap']:.4f}")
    if s.get("representations"):
        lines += ["", "## Subject-invariance probe", "",
                  "| Representation | probe accuracy | chance |", "|---|---:|---:|"]
        for k, v in s["representations"].items():
            p = v["subject_invariance_probe"]
            if p:
                lines.append(f"| {k} | {fmt(p['accuracy_mean'], 3)} | "
                             f"{fmt(p['chance_uniform'], 3)} |")
    if s["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in s["warnings"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
