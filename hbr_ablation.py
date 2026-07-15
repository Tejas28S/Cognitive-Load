"""
HbR ablation — is dropping HbR (deoxy-hemoglobin) from the locked model justified?

The Locked Model uses EEG (180) + HbO (468); HbR (468) is excluded. Earlier
project iterations used all 936 (HbO+HbR), and an RF feature-importance check
found HbR total importance (0.4331) slightly ABOVE HbO (0.3998) — so the drop
was never empirically justified on cross-subject (LOSO) performance.

This runs the decisive comparison on the SAME 26-fold LOSO protocol used
everywhere else:
    A) locked      : EEG + HbO        (n_fnirs = 468)
    B) HbO + HbR   : EEG + HbO + HbR  (n_fnirs = 936)
for the Hybrid network alone AND the Hybrid+RF ensemble (the served config),
with paired significance (B vs A). The Hybrid architecture is unchanged; only
the fNIRS feature set differs.

Outputs: outputs/hbr_ablation.json
Usage:   python hbr_ablation.py [--max-folds N]
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

import config as config_mod
from part1_rigor import clone_config, loso_run, metrics_from, paired_tests, summarize


def ensemble_summary(folds, num_classes=3):
    """0.5/0.5 Hybrid+RF ensemble per fold -> LOSO summary."""
    ens_folds = []
    for f in folds:
        ens = 0.5 * (f["probs"] + f["rf_probs"])
        ens_folds.append({"labels": f["labels"], "preds": ens.argmax(1), "probs": ens})
    return summarize(ens_folds, num_classes)


def f1_array(folds, num_classes=3, ensemble=False):
    out = []
    for f in folds:
        probs = 0.5 * (f["probs"] + f["rf_probs"]) if ensemble else f["probs"]
        out.append(metrics_from(f["labels"], probs.argmax(1), probs, num_classes)["f1_macro"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cpu")

    cfg_hbo = clone_config()  # default = EEG + HbO (468)
    cfg_both = clone_config({
        "MODALITY2_COLUMNS": list(config_mod._hbo_cols) + list(config_mod._hbr_cols),
    })
    print(f"A) locked EEG+HbO : n_fnirs={len(cfg_hbo.MODALITY2_COLUMNS)}")
    print(f"B) EEG+HbO+HbR    : n_fnirs={len(cfg_both.MODALITY2_COLUMNS)}")

    print("\n=== A) EEG + HbO (468, locked) ===")
    fa = loso_run(cfg_hbo, "hybrid", device, args.max_folds, want_rf=True)
    print("\n=== B) EEG + HbO + HbR (936) ===")
    fb = loso_run(cfg_both, "hybrid", device, args.max_folds, want_rf=True)

    res = {}
    for tag, folds in (("hbo_468", fa), ("hbo_hbr_936", fb)):
        hy = summarize(folds)
        en = ensemble_summary(folds)
        res[tag] = {
            "hybrid": {k: hy[k] for k in ("f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean")},
            "ensemble": {k: en[k] for k in ("f1_mean", "f1_std", "acc_mean", "auc_mean", "recall_mean")},
        }

    sig = {
        "hybrid_936_vs_468": paired_tests(
            f1_array(fb), f1_array(fa), "Hybrid HbO+HbR vs HbO (macro F1)"),
        "ensemble_936_vs_468": paired_tests(
            f1_array(fb, ensemble=True), f1_array(fa, ensemble=True),
            "Hybrid+RF HbO+HbR vs HbO (macro F1)"),
    }
    res["significance_936_vs_468"] = sig

    def verdict(s):
        d = s["mean_diff"]; p = s["wilcoxon_p"]
        if p < 0.05 and d > 0:
            return "HbR SIGNIFICANTLY HELPS"
        if p < 0.05 and d < 0:
            return "HbR SIGNIFICANTLY HURTS"
        return "no significant difference (drop justified by parsimony)"
    res["conclusion"] = {
        "hybrid": verdict(sig["hybrid_936_vs_468"]),
        "ensemble": verdict(sig["ensemble_936_vs_468"]),
    }

    print("\n" + "=" * 68)
    for tag in ("hbo_468", "hbo_hbr_936"):
        h = res[tag]["hybrid"]; e = res[tag]["ensemble"]
        print(f"{tag:12s} hybrid F1={h['f1_mean']:.4f} auc={h['auc_mean']:.4f} | "
              f"ensemble F1={e['f1_mean']:.4f} auc={e['auc_mean']:.4f}")
    for k, s in sig.items():
        print(f"{k}: diff={s['mean_diff']:+.4f} t_p={s['t_p']:.3f} wilcoxon_p={s['wilcoxon_p']:.3f}")
    print("conclusion:", res["conclusion"])

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hbr_ablation.json", "w", encoding="utf-8") as f:
        json.dump(_clean(res), f, indent=2)
    print("\nSaved outputs/hbr_ablation.json")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


if __name__ == "__main__":
    main()
