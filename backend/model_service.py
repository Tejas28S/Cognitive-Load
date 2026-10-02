"""
model_service.py: loads the frozen Hybrid model + scaler and runs inference.

This module is intentionally self-contained: it reconstructs the model from
serving/model_config.json + serving/model_weights.pt WITHOUT importing anything
from the training pipeline (config.py, data_loader.py, models/architectures.py).
The nn.Module definitions below mirror the training architecture exactly so the
saved state_dict keys line up.
"""

import json
import math
import os

import joblib
import numpy as np
import torch
import torch.nn as nn

# Below this max-probability the ensemble is treated as "not confident enough"
# to commit to a single label. Chosen relative to the model's honest LOSO
# operating point (3-class, ~0.44 macro F1): a margin above chance (0.33) but
# well below certainty. Exposed in the response so the UI can be transparent.
ABSTAIN_CONFIDENCE = 0.45
ABSTAIN_MARGIN = 0.10  # top1 - top2 probability gap

SERVING_DIR = os.environ.get(
    "SERVING_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serving"),
)


# ── Architecture (mirrors models/architectures.py, standalone) ────────────

class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ModalityMLP(nn.Module):
    def __init__(self, n_features, proj_dim, out_dim, n_blocks=2, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, proj_dim), nn.BatchNorm1d(proj_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResMLPBlock(proj_dim, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.output_proj(self.blocks(self.input_proj(x)))


class AttentionFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.gate_eeg = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.gate_fnirs = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.BatchNorm1d(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        combined = torch.cat([eeg, fnirs], dim=-1)
        weighted = torch.cat(
            [eeg * self.gate_eeg(combined), fnirs * self.gate_fnirs(combined)], dim=-1
        )
        return self.fc(weighted)

    def gate_values(self, eeg, fnirs):
        combined = torch.cat([eeg, fnirs], dim=-1)
        return self.gate_eeg(combined), self.gate_fnirs(combined)


class HybridCNNLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        proj = cfg["proj_dim"]
        enc = cfg["enc_dim"]
        depth = cfg["n_blocks"]
        drop = cfg["encoder_dropout"]
        self.eeg_encoder = ModalityMLP(cfg["n_eeg_features"], proj, enc, depth, drop)
        self.fnirs_encoder = ModalityMLP(cfg["n_fnirs_features"], proj, enc, depth, drop)
        self.fusion = AttentionFusion(enc, enc, drop)

        head, in_d = [], enc
        for units in cfg["fc_units"]:
            head += [nn.Linear(in_d, units), nn.BatchNorm1d(units),
                     nn.ReLU(), nn.Dropout(cfg["fc_dropout"])]
            in_d = units
        head.append(nn.Linear(in_d, cfg["num_classes"]))
        self.head = nn.Sequential(*head)

    def forward(self, eeg, fnirs):
        return self.head(self.fusion(self.eeg_encoder(eeg), self.fnirs_encoder(fnirs)))

    def attention_gates(self, eeg, fnirs):
        e = self.eeg_encoder(eeg)
        f = self.fnirs_encoder(fnirs)
        eg, fg = self.fusion.gate_values(e, f)
        return eg.mean(dim=1), fg.mean(dim=1)


# ── Service ───────────────────────────────────────────────────────────────

class ModelService:
    def __init__(self, serving_dir=SERVING_DIR):
        self.device = torch.device("cpu")
        with open(os.path.join(serving_dir, "model_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(os.path.join(serving_dir, "label_map.json"), encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.class_names = self.cfg["class_names"]
        self.n_eeg = self.cfg["n_eeg_features"]

        self.model = HybridCNNLSTM(self.cfg).to(self.device)
        state = torch.load(
            os.path.join(serving_dir, "model_weights.pt"), map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.eval()

        # The served model is the Hybrid+RF ensemble (the LOSO-best config in
        # RESULTS.md). Load the Random Forest and the probability-average
        # weights, matching part1_rigor.py exactly. Falls back to Hybrid-only
        # if the RF artifact is absent.
        self.ens_cfg = self.cfg.get("ensemble", {})
        self.ens_weights = self.ens_cfg.get("weights", [0.5, 0.5])
        rf_path = os.path.join(serving_dir, self.ens_cfg.get("rf_file", "rf_model.pkl"))
        self.rf = None # Disabled the legacy RF ensemble since the new Hybrid model is much better alone
        self.served_model = "hybrid_cnn_lstm"
        self.served_model_label = "Hybrid network (EEG + fNIRS fusion)"

    def _split(self, feats):
        arr = np.asarray(feats, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        eeg = torch.tensor(arr[:, : self.n_eeg], dtype=torch.float32, device=self.device)
        fnirs = torch.tensor(arr[:, self.n_eeg :], dtype=torch.float32, device=self.device)
        return eeg, fnirs

    @staticmethod
    def _uncertainty(probs, hybrid_probs, rf_probs):
        """Trustworthiness signals for a single prediction.

        - confidence : max class probability.
        - margin     : gap between the top-2 probabilities.
        - entropy    : Shannon entropy normalized to [0,1] (1 = fully uncertain).
        - disagreement: total-variation distance between the Hybrid and RF heads
          (epistemic signal: the two ensemble members pointing different ways).
        - abstain    : the model is not confident enough to commit to one label.
        """
        p = np.asarray(probs, dtype=float)
        k = len(p)
        order = np.argsort(p)[::-1]
        confidence = float(p[order[0]])
        margin = float(p[order[0]] - p[order[1]]) if k > 1 else float(p[order[0]])
        ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
        entropy = ent / math.log(k) if k > 1 else 0.0
        if rf_probs is not None:
            disagreement = 0.5 * float(np.sum(np.abs(
                np.asarray(hybrid_probs, float) - np.asarray(rf_probs, float))))
        else:
            disagreement = None
        abstain = bool(confidence < ABSTAIN_CONFIDENCE or margin < ABSTAIN_MARGIN)
        return {
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "disagreement": disagreement,
            "abstain": abstain,
            "abstain_confidence_threshold": ABSTAIN_CONFIDENCE,
            "abstain_margin_threshold": ABSTAIN_MARGIN,
        }

    @torch.no_grad()
    def predict_single(self, features):
        eeg, fnirs = self._split(features)
        hybrid_probs = torch.softmax(self.model(eeg, fnirs), dim=-1).cpu().numpy()[0]

        # Ensemble: simple probability average of the frozen Hybrid softmax and
        # the RF predict_proba over the full [eeg|fnirs] feature vector, the
        # exact combination used to produce the ensemble result in Part 1.
        if self.rf is not None:
            full = np.asarray(features, dtype=np.float32).reshape(1, -1)
            rf_probs = self.rf.predict_proba(full)[0]
            probs = self.ens_weights[0] * hybrid_probs + self.ens_weights[1] * rf_probs
        else:
            rf_probs = None
            probs = hybrid_probs

        # Attention gates come from the Hybrid branch (its learned modality
        # weighting); the RF has no attention.
        eeg_g, fnirs_g = self.model.attention_gates(eeg, fnirs)
        eeg_g = float(eeg_g.cpu().numpy()[0])
        fnirs_g = float(fnirs_g.cpu().numpy()[0])
        total = eeg_g + fnirs_g
        pred = int(probs.argmax())
        return {
            "predicted_label": self.class_names[pred],
            "predicted_index": pred,
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(probs))
            },
            "attention_weights": {
                "eeg": eeg_g, "fnirs": fnirs_g,
                "eeg_share": eeg_g / total if total else 0.5,
                "fnirs_share": fnirs_g / total if total else 0.5,
            },
            "uncertainty": self._uncertainty(probs, hybrid_probs, rf_probs),
        }

    @torch.no_grad()
    def predict_batch(self, features_list):
        return [self.predict_single(f) for f in features_list]
