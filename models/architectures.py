# ============================================================
#  models/architectures.py
#  MLP-based hybrid for PRE-EXTRACTED FEATURE VECTORS
#  Fixed: proj_dim now controlled by config, not auto-scaled
# ============================================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ── Residual MLP Block ────────────────────────────────────────

class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


# ── Per-modality MLP Encoder ──────────────────────────────────

class ModalityMLP(nn.Module):
    """
    Input  : (B, n_features)
    Output : (B, out_dim)

    proj_dim is fixed by config — NOT scaled by n_features
    This prevents huge models when n_features is large (e.g. 468)
    """
    def __init__(self, n_features, proj_dim, out_dim, n_blocks=2, dropout=0.4):
        super().__init__()

        # Input projection: always goes to fixed proj_dim
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(
            *[ResMLPBlock(proj_dim, dropout) for _ in range(n_blocks)]
        )

        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.out_dim = out_dim

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class EEGSequenceEncoder(nn.Module):
    """LSTM encoder for v2 EEG tensors shaped (batch, timesteps, features)."""
    def __init__(self, n_features, out_dim, cfg: Config = None):
        super().__init__()
        cfg = cfg or Config()
        hidden = cfg.LSTM_UNITS[0]
        num_layers = max(1, len(cfg.LSTM_UNITS))
        dropout = cfg.LSTM_DROPOUT if num_layers > 1 else 0.0

        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.print_shape = bool(getattr(cfg, "PRINT_EEG_LSTM_SHAPE", False))
        self._printed_shape = False

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.print_shape and not self._printed_shape:
            print(f"[Model] EEG LSTM input shape: {tuple(x.shape)}")
            self._printed_shape = True
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        return self.output_proj(out[:, -1, :])


def _is_sequence_shape(n_features):
    return isinstance(n_features, (tuple, list)) and len(n_features) == 2


# ── Fusion modules ────────────────────────────────────────────

class ConcatFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        return self.fc(torch.cat([eeg, fnirs], dim=-1))


class AttentionFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.gate_eeg   = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.gate_fnirs = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        combined = torch.cat([eeg, fnirs], dim=-1)
        weighted = torch.cat([eeg * self.gate_eeg(combined),
                               fnirs * self.gate_fnirs(combined)], dim=-1)
        return self.fc(weighted)

    def gate_values(self, eeg, fnirs):
        combined = torch.cat([eeg, fnirs], dim=-1)
        return self.gate_eeg(combined), self.gate_fnirs(combined)


class GatedFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim * 2, 2), nn.Softmax(dim=-1))
        self.fc   = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        g = self.gate(torch.cat([eeg, fnirs], dim=-1))
        return self.fc(g[:, 0:1] * eeg + g[:, 1:2] * fnirs)


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4, num_heads=4):
        super().__init__()
        if in_dim % num_heads != 0:
            num_heads = 1
        self.eeg_to_fnirs = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fnirs_to_eeg = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        eeg_seq = eeg.unsqueeze(1)
        fnirs_seq = fnirs.unsqueeze(1)
        eeg_attended, _ = self.eeg_to_fnirs(
            query=eeg_seq,
            key=fnirs_seq,
            value=fnirs_seq,
            need_weights=False,
        )
        fnirs_attended, _ = self.fnirs_to_eeg(
            query=fnirs_seq,
            key=eeg_seq,
            value=eeg_seq,
            need_weights=False,
        )
        return self.fc(torch.cat([
            eeg_attended.squeeze(1),
            fnirs_attended.squeeze(1),
        ], dim=-1))


FUSION_MAP = {
    "concat": ConcatFusion,
    "attention": AttentionFusion,
    "gated": GatedFusion,
    "cross_attention": CrossAttentionFusion,
}


# ── Domain-adversarial pieces (DANN) ──────────────────────────
#
# A gradient-reversal layer + subject discriminator make the fused
# representation subject-INVARIANT: the encoder is pushed to fool a
# discriminator that tries to recover which subject a sample came from,
# while the task head still predicts cognitive load. Used only when a
# domain head is attached (dann.py); the default forward() is untouched so
# serving/eval and the frozen locked checkpoint are unaffected.

class GradientReversal(torch.autograd.Function):
    """Identity forward; gradient multiplied by -lambda on the backward pass."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Small MLP that classifies the fused embedding into one of n_domains
    (the training subjects). Trained through the reversed gradient."""
    def __init__(self, in_dim, n_domains, hidden=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x):
        return self.net(x)


# ── Main Hybrid Model ─────────────────────────────────────────

class HybridCNNLSTM(nn.Module):
    """
    Parallel MLP encoders for two modalities + attention fusion.

    Both encoders project to the SAME fixed proj_dim (CNN_FILTERS[-1])
    regardless of input size — this keeps param count predictable.

    With CNN_FILTERS=[64,128], FUSION_DIM=64, FC_UNITS=[64,32]:
      ~200K params — appropriate for 654 samples
    """
    def __init__(self, n_eeg, n_fnirs, cfg: Config = None):
        super().__init__()
        cfg     = cfg or Config()
        proj    = cfg.CNN_FILTERS[-1]   # fixed projection dim for BOTH encoders
        enc_dim = cfg.FUSION_DIM        # encoder output dim
        depth   = len(cfg.LSTM_UNITS)   # number of residual blocks

        if _is_sequence_shape(n_eeg):
            self.eeg_encoder = EEGSequenceEncoder(
                n_features = int(n_eeg[1]),
                out_dim    = enc_dim,
                cfg        = cfg,
            )
        else:
            self.eeg_encoder = ModalityMLP(
                n_features = n_eeg,
                proj_dim   = proj,
                out_dim    = enc_dim,
                n_blocks   = depth,
                dropout    = cfg.CNN_DROPOUT,
            )
        self.fnirs_encoder = ModalityMLP(
            n_features = n_fnirs,
            proj_dim   = proj,          # ← FIXED: same proj_dim, not n_fnirs*4
            out_dim    = enc_dim,
            n_blocks   = depth,
            dropout    = cfg.CNN_DROPOUT,
        )

        fusion_cls = FUSION_MAP[cfg.FUSION_TYPE]
        if cfg.FUSION_TYPE == "cross_attention":
            self.fusion = fusion_cls(
                enc_dim,
                enc_dim,
                cfg.CNN_DROPOUT,
                num_heads=getattr(cfg, "CROSS_ATTENTION_HEADS", 4),
            )
        else:
            self.fusion = fusion_cls(enc_dim, enc_dim, cfg.CNN_DROPOUT)

        head = []
        in_d = enc_dim
        for units in cfg.FC_UNITS:
            head += [
                nn.Linear(in_d, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(cfg.FC_DROPOUT),
            ]
            in_d = units
        head.append(nn.Linear(in_d, cfg.NUM_CLASSES))
        self.head = nn.Sequential(*head)

        self.enc_dim = enc_dim
        self.domain_head = None   # attached on demand for DANN training

    def encode(self, eeg, fnirs):
        """Grad-enabled fused embedding (the DANN attack point)."""
        return self.fusion(self.eeg_encoder(eeg), self.fnirs_encoder(fnirs))

    def forward(self, eeg, fnirs):
        return self.head(self.encode(eeg, fnirs))

    # ── DANN hooks (no effect unless a domain head is attached) ──
    def attach_domain_head(self, n_domains, hidden=64, dropout=0.5):
        self.domain_head = DomainDiscriminator(
            self.enc_dim, n_domains, hidden=hidden, dropout=dropout
        )
        return self.domain_head

    def forward_dann(self, eeg, fnirs, lambda_=1.0):
        """Return (task_logits, domain_logits); domain grad is reversed."""
        if self.domain_head is None:
            raise RuntimeError("attach_domain_head() must be called before forward_dann()")
        feat = self.encode(eeg, fnirs)
        task_logits = self.head(feat)
        domain_logits = self.domain_head(grad_reverse(feat, lambda_))
        return task_logits, domain_logits

    def get_embeddings(self, eeg, fnirs):
        with torch.no_grad():
            return self.encode(eeg, fnirs)

    def get_attention_weights(self, eeg, fnirs):
        with torch.no_grad():
            eeg_emb = self.eeg_encoder(eeg)
            fnirs_emb = self.fnirs_encoder(fnirs)
            if not hasattr(self.fusion, "gate_values"):
                raise ValueError("Attention weights are only available for attention fusion")
            eeg_gate, fnirs_gate = self.fusion.gate_values(eeg_emb, fnirs_emb)
            return eeg_gate.mean(dim=1), fnirs_gate.mean(dim=1)


class CrossAttentionHybrid(HybridCNNLSTM):
    """Hybrid model variant that keeps both encoders and swaps in cross-attention fusion."""
    def __init__(self, n_eeg, n_fnirs, cfg: Config = None):
        cfg = copy.copy(cfg or Config())
        cfg.FUSION_TYPE = "cross_attention"
        super().__init__(n_eeg, n_fnirs, cfg)


# ── Unimodal Baseline ─────────────────────────────────────────

class UnimodalCNNLSTM(nn.Module):
    def __init__(self, n_features, cfg: Config = None):
        super().__init__()
        cfg  = cfg or Config()
        proj = cfg.CNN_FILTERS[-1]
        if _is_sequence_shape(n_features):
            self.encoder = EEGSequenceEncoder(
                n_features = int(n_features[1]),
                out_dim    = cfg.FUSION_DIM,
                cfg        = cfg,
            )
        else:
            self.encoder = ModalityMLP(
                n_features = n_features,
                proj_dim   = proj,
                out_dim    = cfg.FUSION_DIM,
                n_blocks   = len(cfg.LSTM_UNITS),
                dropout    = cfg.CNN_DROPOUT,
            )
        head = []
        in_d = cfg.FUSION_DIM
        for units in cfg.FC_UNITS:
            head += [
                nn.Linear(in_d, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(cfg.FC_DROPOUT),
            ]
            in_d = units
        head.append(nn.Linear(in_d, cfg.NUM_CLASSES))
        self.head = nn.Sequential(*head)

    def forward(self, x, _=None):
        return self.head(self.encoder(x))


def build_model(model_type, n_eeg, n_fnirs, cfg=None):
    cfg = cfg or Config()
    if model_type == "hybrid_cnn_lstm":
        return HybridCNNLSTM(n_eeg, n_fnirs, cfg)
    elif model_type == "cross_attention_hybrid":
        return CrossAttentionHybrid(n_eeg, n_fnirs, cfg)
    elif model_type == "unimodal_eeg":
        return UnimodalCNNLSTM(n_eeg, cfg)
    elif model_type == "unimodal_fnirs":
        return UnimodalCNNLSTM(n_fnirs, cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
