# ============================================================
#  train_with_pca.py
#  Applies PCA before training to fix curse of dimensionality
#  Run: python train_with_pca.py
#  Run: python train_with_pca.py --n-components 150
# ============================================================

import os, sys, argparse, pickle, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
sys.path.insert(0, "models")
from config import Config
from data_loader import CognitiveLoadDataLoader, CognitiveLoadDataset, configured_class_weights
from models.architectures import build_model
from trainer import train_model
from evaluate import evaluate_model, save_evaluation_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", type=int, default=None,
                        help="Deprecated joint PCA setting; use --eeg-components and --fnirs-components")
    parser.add_argument("--eeg-components", type=int, default=50,
                        help="EEG PCA components (default 50)")
    parser.add_argument("--fnirs-components", type=int, default=100,
                        help="fNIRS PCA components (default 100)")
    parser.add_argument("--variance",     type=float, default=None,
                        help="Auto-select n by variance threshold e.g. 0.90")
    args = parser.parse_args()

    cfg = Config()
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  Cognitive Load â€” Training WITH PCA")
    print("  Fixing: Curse of Dimensionality")
    print("="*60)

    # â”€â”€ 1. Load raw features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dl  = CognitiveLoadDataLoader(cfg)
    eeg_raw, fnirs_raw, labels, groups = dl.load_raw(return_groups=True)

    print(f"\n  Original feature space:")
    print(f"    EEG   : {eeg_raw.shape[1]} features")
    print(f"    fNIRS : {fnirs_raw.shape[1]} features")
    print(f"    Total : {eeg_raw.shape[1]+fnirs_raw.shape[1]} features")
    print(f"    Samples: {len(labels)}")
    print(f"    Ratio  : {len(labels)/(eeg_raw.shape[1]+fnirs_raw.shape[1]):.2f}  "
          f"â† {'âŒ Too low' if len(labels)/(eeg_raw.shape[1]+fnirs_raw.shape[1]) < 1 else 'âš ï¸'}")

    # â”€â”€ 2. Train/val/test split BEFORE fitting PCA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_idx, val_idx, test_idx = dl.split_indices(labels, groups)
    eeg_train, eeg_val, eeg_test = eeg_raw[train_idx], eeg_raw[val_idx], eeg_raw[test_idx]
    fnirs_train, fnirs_val, fnirs_test = fnirs_raw[train_idx], fnirs_raw[val_idx], fnirs_raw[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    print(f"\n  Split:")
    print(f"    Train: {len(y_train)} samples")
    print(f"    Val  : {len(y_val)} samples")
    print(f"    Test : {len(y_test)} samples")
    if groups is not None:
        print(f"    Groups train/val/test: "
              f"{len(np.unique(groups[train_idx]))}/"
              f"{len(np.unique(groups[val_idx]))}/"
              f"{len(np.unique(groups[test_idx]))}")

    # â”€â”€ 3. Normalize each modality (fit on train only) â”€â”€â”€â”€â”€â”€â”€
    eeg_scaler = RobustScaler()
    fnirs_scaler = StandardScaler()
    eeg_train_s = eeg_scaler.fit_transform(eeg_train)
    eeg_val_s = eeg_scaler.transform(eeg_val)
    eeg_test_s = eeg_scaler.transform(eeg_test)
    fnirs_train_s = fnirs_scaler.fit_transform(fnirs_train)
    fnirs_val_s = fnirs_scaler.transform(fnirs_val)
    fnirs_test_s = fnirs_scaler.transform(fnirs_test)

    # â”€â”€ 4. Per-modality PCA (fit on train only) â”€â”€â”€â”€â”€â”€â”€
    if args.variance:
        eeg_pca = PCA(n_components=args.variance, random_state=42)
        fnirs_pca = PCA(n_components=args.variance, random_state=42)
        print(f"\n  PCA: per-modality auto n_components for {args.variance*100:.0f}% variance")
    else:
        if args.n_components is not None:
            args.eeg_components = min(args.eeg_components, args.n_components)
            args.fnirs_components = min(args.fnirs_components, args.n_components)
        eeg_components = min(args.eeg_components, eeg_train_s.shape[0], eeg_train_s.shape[1])
        fnirs_components = min(args.fnirs_components, fnirs_train_s.shape[0], fnirs_train_s.shape[1])
        eeg_pca = PCA(n_components=eeg_components, random_state=42)
        fnirs_pca = PCA(n_components=fnirs_components, random_state=42)
        print(f"\n  PCA: EEG n_components={eeg_components}, fNIRS n_components={fnirs_components}")

    eeg_train_pca = eeg_pca.fit_transform(eeg_train_s)
    eeg_val_pca = eeg_pca.transform(eeg_val_s)
    eeg_test_pca = eeg_pca.transform(eeg_test_s)
    fnirs_train_pca = fnirs_pca.fit_transform(fnirs_train_s)
    fnirs_val_pca = fnirs_pca.transform(fnirs_val_s)
    fnirs_test_pca = fnirs_pca.transform(fnirs_test_s)

    eeg_var = eeg_pca.explained_variance_ratio_.sum() * 100
    fnirs_var = fnirs_pca.explained_variance_ratio_.sum() * 100
    n_eeg_pca = eeg_train_pca.shape[1]
    n_fnirs_pca = fnirs_train_pca.shape[1]
    n_comp = n_eeg_pca + n_fnirs_pca
    ratio_after = len(y_train) / n_comp
    total_features = eeg_raw.shape[1] + fnirs_raw.shape[1]

    print(f"\n  After per-modality PCA:")
    print(f"    EEG components   : {n_eeg_pca} ({eeg_var:.1f}% variance)")
    print(f"    fNIRS components : {n_fnirs_pca} ({fnirs_var:.1f}% variance)")
    print(f"    Total components : {n_comp}")
    print(f"    Sample:feat      : {ratio_after:.1f}  "
          f"{'âœ… Good' if ratio_after >= 3 else 'âš ï¸ Still low'}")
    print(f"    Reduction        : {total_features} â†’ {n_comp} features "
          f"({(1-n_comp/total_features)*100:.0f}% reduction)")

    # Save PCA + scalers for inference
    pca_path = os.path.join(cfg.OUTPUT_DIR, "pca_scaler.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump({
            "eeg_pca": eeg_pca,
            "fnirs_pca": fnirs_pca,
            "eeg_scaler": eeg_scaler,
            "fnirs_scaler": fnirs_scaler,
            "n_eeg_components": n_eeg_pca,
            "n_fnirs_components": n_fnirs_pca,
            "eeg_variance_explained": eeg_var,
            "fnirs_variance_explained": fnirs_var,
        }, f)
    print(f"    Saved          : {pca_path}")

    # â”€â”€ 5. Build DataLoaders, preserving modality boundaries â”€â”€â”€
    def make_loader(eeg_X, fnirs_X, y, shuffle=True, drop=True):
        cw = None
        if shuffle:
            cw = configured_class_weights(y, cfg)
        ds = CognitiveLoadDataset(
            eeg_X.astype(np.float32),
            fnirs_X.astype(np.float32),
            y.astype(np.int64),
        )
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=shuffle, drop_last=drop and shuffle,
                          num_workers=0), cw

    train_loader, cw = make_loader(eeg_train_pca, fnirs_train_pca, y_train)
    val_loader, _ = make_loader(eeg_val_pca, fnirs_val_pca, y_val, shuffle=False, drop=False)
    test_loader, _ = make_loader(eeg_test_pca, fnirs_test_pca, y_test, shuffle=False, drop=False)

    meta = {
        "n_eeg_features"  : n_eeg_pca,
        "n_fnirs_features": n_fnirs_pca,
        "class_weights"   : cw,
        "y_test"          : y_test,
    }

    print(f"\n  PCA split into encoders:")
    print(f"    EEG encoder  : {n_eeg_pca} PCA components")
    print(f"    fNIRS encoder: {n_fnirs_pca} PCA components")

    # â”€â”€ 6. Train all 3 models with PCA features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    histories, results = {}, {}
    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "pca_hybrid_best.pt")

    print(f"\n  â”€â”€ [1/3] Hybrid CNN-LSTM (PCA) â”€â”€")
    hybrid = build_model("hybrid_cnn_lstm", n_eeg_pca, n_fnirs_pca, cfg)
    hybrid, hist = train_model(hybrid, train_loader, val_loader, meta, cfg)
    histories["PCA Hybrid"] = hist
    results["PCA\nHybrid"]  = evaluate_model(hybrid, test_loader, cfg,
                                              model_name="PCA Hybrid CNN-LSTM")

    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "pca_eeg_best.pt")
    print(f"\n  â”€â”€ [2/3] EEG-only (PCA) â”€â”€")
    eeg_m = build_model("unimodal_eeg", n_eeg_pca, n_fnirs_pca, cfg)
    eeg_m, _ = train_model(eeg_m, train_loader, val_loader, meta, cfg,
                            is_unimodal=True, modality="eeg")
    results["PCA\nEEG Only"] = evaluate_model(eeg_m, test_loader, cfg,
                                               model_name="PCA EEG Only",
                                               is_unimodal=True, modality="eeg")

    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "pca_fnirs_best.pt")
    print(f"\n  â”€â”€ [3/3] fNIRS-only (PCA) â”€â”€")
    fn_m = build_model("unimodal_fnirs", n_eeg_pca, n_fnirs_pca, cfg)
    fn_m, _ = train_model(fn_m, train_loader, val_loader, meta, cfg,
                           is_unimodal=True, modality="fnirs")
    results["PCA\nfNIRS Only"] = evaluate_model(fn_m, test_loader, cfg,
                                                 model_name="PCA fNIRS Only",
                                                 is_unimodal=True, modality="fnirs")

    # â”€â”€ 7. Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n{'='*58}")
    print(f"  {'Model':<22} {'Acc':>8} {'Macro F1':>10} {'AUC':>8}")
    print(f"{'â”€'*58}")
    for name, r in results.items():
        clean = name.replace("\n", " ")
        print(f"  {clean:<22} {r['acc']:>7.2%}  {r['f1_macro']:>8.4f}  {r['auc']:>7.4f}")
    print(f"{'='*58}")
    print(f"\n  PCA reduced {total_features} â†’ {n_comp} features")
    print(f"  Sample:feature ratio improved: "
          f"{len(labels)/total_features:.2f} â†’ {ratio_after:.1f}")

    save_evaluation_dashboard(histories, results, cfg, cfg.OUTPUT_DIR)
    print(f"\nâœ… All outputs â†’ {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

