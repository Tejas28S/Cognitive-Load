# ============================================================
#  main.py  –  Full pipeline: load → train → evaluate → compare
# ============================================================
#
#  Usage:
#      python main.py                  # default: hybrid model + ablation
#      python main.py --cv             # also run 5-fold CV
#      python main.py --no-ablation    # skip unimodal baselines
#
# ============================================================

import os, sys, argparse, random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from config              import Config
from data_loader         import CognitiveLoadDataLoader
from models.architectures import build_model, HybridCNNLSTM
from trainer             import train_model, cross_validate
from evaluate            import evaluate_model, save_evaluation_dashboard


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv",           action="store_true",  help="Run cross-validation")
    p.add_argument("--no-ablation",  action="store_true",  help="Skip unimodal baselines")
    p.add_argument("--fusion",       default=None,
                   choices=["concat", "attention", "gated", "cross_attention"],
                   help="Override fusion type from config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    if args.fusion:
        cfg.FUSION_TYPE = args.fusion
        print(f"  ⚙  Fusion overridden → {cfg.FUSION_TYPE}")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print("\n" + "═"*60)
    print("  Multimodal Cognitive Load Prediction")
    print("  EEG + fNIRS  |  Hybrid CNN-LSTM  |  3-Class")
    print("═"*60)

    # ── 1. Load data ────────────────────────────────────────
    dl   = CognitiveLoadDataLoader(cfg)
    train_loader, val_loader, test_loader, meta = dl.prepare()
    n_eeg   = meta["n_eeg_features"]
    n_fnirs = meta["n_fnirs_features"]

    print(f"\n  EEG features  : {n_eeg}")
    print(f"  fNIRS features: {n_fnirs}")

    # ── 2. Train hybrid model ───────────────────────────────
    print("\n  ── [1/3]  Training Hybrid CNN-LSTM ──")
    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "hybrid_best.pt")
    hybrid_model   = build_model("hybrid_cnn_lstm", n_eeg, n_fnirs, cfg)
    hybrid_model, hybrid_hist = train_model(
        hybrid_model, train_loader, val_loader, meta, cfg
    )

    all_histories = {"hybrid_cnn_lstm": hybrid_hist}
    all_results   = {}

    # ── 3. Evaluate hybrid ──────────────────────────────────
    all_results["Hybrid\nCNN-LSTM"] = evaluate_model(
        hybrid_model, test_loader, cfg, model_name="Hybrid CNN-LSTM"
    )

    # ── 4. Ablation: unimodal baselines ─────────────────────
    if not args.no_ablation:
        print("\n  ── [2/3]  Unimodal Baseline – EEG only ──")
        cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "eeg_only_best.pt")
        eeg_model      = build_model("unimodal_eeg", n_eeg, n_fnirs, cfg)
        eeg_model, _   = train_model(eeg_model, train_loader, val_loader,
                                     meta, cfg, is_unimodal=True, modality="eeg")
        all_results["EEG\nOnly"] = evaluate_model(
            eeg_model, test_loader, cfg,
            model_name="EEG Only", is_unimodal=True, modality="eeg"
        )

        print("\n  ── [3/3]  Unimodal Baseline – fNIRS only ──")
        cfg.MODEL_PATH  = os.path.join(cfg.OUTPUT_DIR, "fnirs_only_best.pt")
        fnirs_model     = build_model("unimodal_fnirs", n_eeg, n_fnirs, cfg)
        fnirs_model, _  = train_model(fnirs_model, train_loader, val_loader,
                                      meta, cfg, is_unimodal=True, modality="fnirs")
        all_results["fNIRS\nOnly"] = evaluate_model(
            fnirs_model, test_loader, cfg,
            model_name="fNIRS Only", is_unimodal=True, modality="fnirs"
        )

    # ── 5. Summary table ────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  {'Model':<22} {'Acc':>8} {'Macro F1':>10} {'AUC':>8}")
    print(f"{'─'*55}")
    for name, r in all_results.items():
        clean = name.replace('\n', ' ')
        print(f"  {clean:<22} {r['acc']:>7.2%}  {r['f1_macro']:>8.4f}  {r['auc']:>7.4f}")
    print(f"{'═'*55}")

    # ── 6. Save dashboard ────────────────────────────────────
    save_evaluation_dashboard(all_histories, all_results, cfg, cfg.OUTPUT_DIR)

    # ── 7. Optional: Cross-validation ───────────────────────
    if args.cv or cfg.RUN_CV:
        def model_builder(fold_meta):
            return build_model("hybrid_cnn_lstm",
                               fold_meta["n_eeg_features"],
                               fold_meta["n_fnirs_features"], cfg)
        cross_validate(model_builder, dl, cfg)

    print(f"\n  ✅  All outputs saved to  →  {cfg.OUTPUT_DIR}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
