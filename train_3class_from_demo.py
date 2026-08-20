"""Rebuild a servable 3-class model for serving_3class/.

WHY THIS EXISTS
---------------
The original 3-class research checkpoint (outputs/hybrid_best.pt, exported by
export_3class.py) is not in this repository and cannot be regenerated here: the
extracted feature data under data/ is absent. Without model_weights.pt the whole
3-class option in the app is dead.

What IS available is serving_3class/demo_samples.json — 7650 real, fully
preprocessed TU Berlin windows across 6 subjects, each with its 3-class label.
This script trains the exact architecture described by
serving_3class/model_config.json on those windows so the 3-class mode works.

HONESTY / PROVENANCE
--------------------
The model this produces is NOT the research model behind RESULTS.md:
  * it is trained on what were originally the fixed-split TEST windows, because
    that is the only labelled 3-class data present;
  * it sees 6 subjects, not 26.
Its accuracy is therefore not comparable to the published LOSO numbers, and it
is recorded as such in model_config.json["provenance"], which the API exposes so
the UI can label it. Evaluation here is subject-wise (the validation subjects
are never trained on), so the number printed is at least an honest measure of
THIS model — it is simply a different, smaller experiment.

Usage:  python train_3class_from_demo.py [--epochs N] [--val-subjects 18 24]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
# Train the same nn.Module the server loads, so the state_dict keys line up by
# construction rather than by hand-matching two architecture definitions.
from model_service import HybridCNNLSTM  # noqa: E402

BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving_3class")
BINARY_BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving")
SEED = 42


def load_samples():
    with open(os.path.join(BUNDLE, "demo_samples.json"), encoding="utf-8") as f:
        d = json.load(f)
    s = d["samples"]
    X = np.asarray([r["features"] for r in s], dtype=np.float32)
    y = np.asarray([r["true_label"] for r in s], dtype=np.int64)
    g = np.asarray([r["subject_id"] for r in s], dtype=np.int64)
    # A handful of windows carry extreme tails; clip so BatchNorm statistics are
    # not dominated by a few values (the p99 of this feature space is ~3.4).
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
    return X, y, g


def evaluate(model, X, y, device, bs=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i : i + bs], device=device)
            logits = model(xb[:, :540], xb[:, 540:])
            preds.append(logits.argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return accuracy_score(y, p), f1_score(y, p, average="macro"), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-subjects", type=int, nargs="+", default=[18, 24])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    with open(os.path.join(BUNDLE, "model_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    X, y, g = load_samples()
    val_mask = np.isin(g, args.val_subjects)
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]
    print(f"subjects       : {sorted(set(g.tolist()))}")
    print(f"validation on  : {args.val_subjects} (never trained on)")
    print(f"train / val    : {len(Xtr)} / {len(Xva)} windows")
    print(f"class balance  : {np.bincount(ytr).tolist()} (train)")

    model = HybridCNNLSTM(cfg).to(device)
    # Inverse-frequency weights: the split is near balanced, this just keeps a
    # mild imbalance from steering the argmax.
    counts = np.bincount(ytr, minlength=cfg["num_classes"]).astype(np.float32)
    w = torch.tensor((counts.sum() / (len(counts) * counts)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_state, best_epoch = -1.0, None, -1
    n = len(Xtr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            xb = torch.tensor(Xtr[idx], device=device)
            yb = torch.tensor(ytr[idx], device=device)
            opt.zero_grad()
            loss = criterion(model(xb[:, :540], xb[:, 540:]), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        acc, f1, _ = evaluate(model, Xva, yva, device)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss {total/n:.4f}  val_acc {acc:.4f}  val_macroF1 {f1:.4f}")

    model.load_state_dict(best_state)
    acc, f1, preds = evaluate(model, Xva, yva, device)
    print(f"\nbest epoch {best_epoch}: held-out-subject acc {acc:.4f}  macro F1 {f1:.4f} "
          f"(chance ≈ {1/cfg['num_classes']:.3f})")
    print("per-class recall:",
          [round(float(f1_score(yva == c, preds == c)), 3) for c in range(cfg["num_classes"])])

    torch.save(model.state_dict(), os.path.join(BUNDLE, "model_weights.pt"))
    print(f"wrote {BUNDLE}/model_weights.pt")

    # The scaler maps raw uploaded features into this space. It is label-agnostic
    # and both bundles declare byte-identical eeg/fnirs column lists, so the
    # binary bundle's fitted scaler is the correct one for 3-class uploads too.
    with open(os.path.join(BUNDLE, "feature_schema.json"), encoding="utf-8") as f:
        s3 = json.load(f)
    with open(os.path.join(BINARY_BUNDLE, "feature_schema.json"), encoding="utf-8") as f:
        sb = json.load(f)
    if s3["eeg_columns"] != sb["eeg_columns"] or s3["fnirs_columns"] != sb["fnirs_columns"]:
        raise SystemExit("Feature columns differ between bundles — refusing to reuse the scaler.")
    shutil.copyfile(os.path.join(BINARY_BUNDLE, "feature_scaler.pkl"),
                    os.path.join(BUNDLE, "feature_scaler.pkl"))
    print(f"wrote {BUNDLE}/feature_scaler.pkl (copied from serving/, identical feature columns)")

    cfg["frozen_checkpoint"] = None
    cfg["provenance"] = {
        "kind": "retrained_stand_in",
        "trained_by": "train_3class_from_demo.py",
        "trained_on": "serving_3class/demo_samples.json (fixed-split test windows)",
        "n_subjects": int(len(set(g.tolist()))),
        "validation_subjects": [int(v) for v in args.val_subjects],
        "heldout_subject_accuracy": round(float(acc), 4),
        "heldout_subject_macro_f1": round(float(f1), 4),
        "warning": (
            "Not the research checkpoint behind RESULTS.md. Trained on the "
            "fixed-split test windows across 6 subjects because the original "
            "checkpoint and data/ are absent. Its accuracy is NOT comparable to "
            "the published 26-fold LOSO numbers."
        ),
    }
    with open(os.path.join(BUNDLE, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {BUNDLE}/model_config.json (provenance recorded)")


if __name__ == "__main__":
    main()
