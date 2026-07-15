# ============================================================
#  test_model.py  –  Load saved .pt model and run inference
#
#  Usage:
#    python test_model.py                        # test on held-out test set
#    python test_model.py --new-csv path/to.csv  # test on any new CSV
#    python test_model.py --model outputs/fnirs_only_best.pt  # use different model
# ============================================================

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from config import Config
from data_loader import CognitiveLoadDataLoader
from models.architectures import build_model
from evaluate import evaluate_model


def load_saved_model(model_path, n_eeg, n_fnirs, cfg):
    """Load a .pt model file back into memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("hybrid_cnn_lstm", n_eeg, n_fnirs, cfg)
    state  = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded: {model_path}")
    print(f"   Params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device : {device}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="outputs/hybrid_best.pt",
                        help="Path to .pt model file  (default: outputs/hybrid_best.pt)")
    parser.add_argument("--new-csv", default=None,
                        help="Optional: path to a new CSV to predict on")
    args = parser.parse_args()

    cfg = Config()

    print("\n" + "="*58)
    print("  Cognitive Load Prediction — Model Inference")
    print("="*58)

    # ── List available models ─────────────────────────────────
    print("\n  Available saved models in outputs/:")
    for f in sorted(os.listdir("outputs")):
        if f.endswith(".pt"):
            size = os.path.getsize(f"outputs/{f}") / 1024
            marker = " ← selected" if f"outputs/{f}" == args.model else ""
            print(f"    outputs/{f}  ({size:.0f} KB){marker}")

    if not os.path.exists(args.model):
        print(f"\n❌  Model not found: {args.model}")
        print("    Train first with:  python main.py")
        sys.exit(1)

    # ── Load data (needed for feature dims + scalers) ─────────
    print(f"\n📂 Loading dataset to get feature dimensions...")
    dl = CognitiveLoadDataLoader(cfg)
    train_loader, val_loader, test_loader, meta = dl.prepare()
    n_eeg   = meta["n_eeg_features"]
    n_fnirs = meta["n_fnirs_features"]
    print(f"   EEG features  : {n_eeg}")
    print(f"   fNIRS features: {n_fnirs}")

    # ── Load model ────────────────────────────────────────────
    print()
    model = load_saved_model(args.model, n_eeg, n_fnirs, cfg)

    # ── Option A: predict on a new CSV ───────────────────────
    if args.new_csv:
        print(f"\n📊 Running inference on: {args.new_csv}")
        df = pd.read_csv(args.new_csv)

        eeg_cols   = dl.eeg_cols
        fnirs_cols = dl.fnirs_cols

        # Check columns exist
        missing = [c for c in eeg_cols + fnirs_cols if c not in df.columns]
        if missing:
            print(f"❌  Missing {len(missing)} columns in CSV: {missing[:3]} ...")
            sys.exit(1)

        X_eeg   = df[eeg_cols].values.astype(np.float32)
        X_fnirs = df[fnirs_cols].values.astype(np.float32)

        if cfg.NORMALIZE:
            X_eeg   = dl.eeg_scaler.transform(X_eeg)
            X_fnirs = dl.fnirs_scaler.transform(X_fnirs)

        device = next(model.parameters()).device
        preds, probs_all = [], []

        with torch.no_grad():
            for i in range(len(X_eeg)):
                eeg_t  = torch.tensor(X_eeg[i],   dtype=torch.float32).unsqueeze(0).to(device)
                fn_t   = torch.tensor(X_fnirs[i],  dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(eeg_t, fn_t)
                prob   = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                preds.append(int(prob.argmax()))
                probs_all.append(prob)

        # Print results
        label_names = cfg.CLASS_NAMES
        print(f"\n{'─'*52}")
        print(f"  {'Sample':>6}  {'Prediction':>22}  Confidence")
        print(f"{'─'*52}")
        for i, (p, prob) in enumerate(zip(preds, probs_all)):
            bar = "█" * int(prob[p] * 20)
            print(f"  {i:>6}  {label_names[p]:>22}  {prob[p]:>5.1%}  {bar}")
        print(f"{'─'*52}")

        # Distribution
        from collections import Counter
        dist = Counter(preds)
        print(f"\n  Prediction distribution:")
        for lbl, name in enumerate(label_names):
            n = dist.get(lbl, 0)
            bar = "█" * int(n / len(preds) * 30)
            print(f"    {name:25s}: {n:3d} ({n/len(preds):.1%})  {bar}")

        # Save predictions CSV
        out_path = args.new_csv.replace(".csv", "_predictions.csv")
        df_out = df.copy()
        df_out["predicted_label"] = preds
        df_out["predicted_name"]  = [label_names[p] for p in preds]
        for i, name in enumerate(label_names):
            key = name.split()[0].replace("(", "").replace(")", "")
            df_out[f"prob_{key}"] = [pr[i] for pr in probs_all]
        df_out.to_csv(out_path, index=False)
        print(f"\n  ✅ Predictions saved → {out_path}")

    # ── Option B: evaluate on held-out test set ───────────────
    else:
        print(f"\n📊 Evaluating on held-out test set ({len(test_loader.dataset)} samples)...")
        results = evaluate_model(
            model, test_loader, cfg,
            model_name="Hybrid CNN-LSTM"
        )

        print(f"\n{'='*58}")
        print(f"  FINAL TEST RESULTS")
        print(f"{'─'*58}")
        print(f"  Model file  : {args.model}")
        print(f"  Accuracy    : {results['acc']:.4f}  ({results['acc']:.2%})")
        print(f"  Macro F1    : {results['f1_macro']:.4f}")
        print(f"  Weighted F1 : {results['f1_weighted']:.4f}")
        print(f"  ROC-AUC     : {results['auc']:.4f}")
        print(f"  Random base : 33.33%  (3-class)")
        print(f"  Above chance: +{(results['acc']-0.333)*100:.1f} pp")
        print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
