"""
archs.py — model zoo.

Every model exposes the same interface so the trainer, the explainability
code and the DANN path are architecture-agnostic:

    forward(eeg, fnirs) -> logits
    encode(eeg, fnirs)  -> fused embedding      (the DANN attack point)
    gates(eeg, fnirs)   -> (eeg_gate, fnirs_gate) or None
    attach_domain_head(n_domains)
    forward_dann(eeg, fnirs, lambda_)

Why a 1D CNN is legitimate on these features
--------------------------------------------
The feature columns are channel-major, so each modality block reshapes exactly
onto a (channel, feature-type) grid:

    EEG   540 = 30 electrodes x 18 feature types (6 bands x {mean,std,slope})
    fNIRS 936 = 36 optodes    x 26 feature types (13 stats x {HbO,HbR})

Convolving along the ELECTRODE / OPTODE axis with the feature types as input
channels shares one filter bank across all channels.  That is a real inductive
bias (the same spectral/haemodynamic pattern is looked for everywhere on the
head) and it cuts the input-stage parameter count by more than an order of
magnitude versus a dense 540->96 projection.  It assumes neighbouring channel
indices are spatially related, which is only approximately true for this
montage — so it is an experiment, not an assumption, and is reported as such.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── shared pieces ──────────────────────────────────────────────────────

class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ModalityMLP(nn.Module):
    """Dense encoder: n_features -> proj_dim -> [residual blocks] -> out_dim."""

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


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ResConvBlock(nn.Module):
    """Two convolutions with an identity (or 1x1-projected) skip."""

    def __init__(self, cin, cout, k=3, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ModalityCNN(nn.Module):
    """1D CNN over the channel axis of a (n_channels, n_feats) grid."""

    def __init__(self, n_channels, n_feats, out_dim, filters=(64, 64),
                 kernel_size=3, dropout=0.3, residual=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_feats = n_feats
        block = ResConvBlock if residual else ConvBlock
        layers, cin = [], n_feats
        for cout in filters:
            layers.append(block(cin, cout, kernel_size, dropout))
            cin = cout
        self.conv = nn.Sequential(*layers)
        # Mean + max pooling over channels: "how strong on average" and
        # "how strong at the most responsive site".
        self.output_proj = nn.Sequential(
            nn.Linear(cin * 2, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, self.n_channels, self.n_feats).transpose(1, 2)  # (B, feats, chans)
        h = self.conv(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.output_proj(pooled)


class FNIRSGridCNN(ModalityCNN):
    """fNIRS block is [HbO(36x13) | HbR(36x13)]; stack them on the feature axis."""

    def __init__(self, n_channels, n_feats_per_chromophore, n_chromophores,
                 out_dim, **kw):
        super().__init__(
            n_channels=n_channels,
            n_feats=n_feats_per_chromophore * n_chromophores,
            out_dim=out_dim,
            **kw,
        )
        self.n_chrom = n_chromophores
        self.n_feats_pc = n_feats_per_chromophore

    def forward(self, x):
        b = x.shape[0]
        parts = x.chunk(self.n_chrom, dim=1)  # HbO block, HbR block
        grid = torch.cat(
            [p.view(b, self.n_channels, self.n_feats_pc) for p in parts], dim=2
        )
        h = self.conv(grid.transpose(1, 2))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.output_proj(pooled)


# ── fusion ─────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.BatchNorm1d(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        return self.fc(torch.cat([eeg, fnirs], dim=-1))

    def gate_values(self, eeg, fnirs):
        return None


class AttentionFusion(nn.Module):
    """Per-modality sigmoid gate computed from both embeddings."""

    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.gate_eeg = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.gate_fnirs = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.BatchNorm1d(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, eeg, fnirs):
        c = torch.cat([eeg, fnirs], dim=-1)
        return self.fc(
            torch.cat([eeg * self.gate_eeg(c), fnirs * self.gate_fnirs(c)], dim=-1)
        )

    def gate_values(self, eeg, fnirs):
        c = torch.cat([eeg, fnirs], dim=-1)
        return self.gate_eeg(c), self.gate_fnirs(c)


FUSIONS = {"concat": ConcatFusion, "attention": AttentionFusion}


# ── DANN ───────────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
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
    def __init__(self, in_dim, n_domains, hidden=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_domains),
        )

    def forward(self, x):
        return self.net(x)


# ── the model ──────────────────────────────────────────────────────────

class DualStreamNet(nn.Module):
    """Two modality encoders + fusion + classification head.

    `encoder` selects the encoder family ("mlp" or "cnn"); everything else
    (fusion type, residual blocks, dropout, modality dropout) is orthogonal,
    so the CNN/MLP/residual/attention comparisons are controlled.
    """

    def __init__(self, spec, schema, n_classes):
        super().__init__()
        self.spec = spec
        enc_dim = spec["enc_dim"]
        kind = spec["encoder"]

        n_eeg = len(schema.eeg_cols)
        n_fnirs = len(schema.fnirs_cols)

        if kind == "mlp":
            self.eeg_encoder = ModalityMLP(
                n_eeg, spec["proj_dim"], enc_dim,
                n_blocks=spec["n_blocks"], dropout=spec["encoder_dropout"],
            )
            self.fnirs_encoder = ModalityMLP(
                n_fnirs, spec["proj_dim"], enc_dim,
                n_blocks=spec["n_blocks"], dropout=spec["encoder_dropout"],
            )
        elif kind == "cnn":
            eeg_ch, eeg_f, _ = schema.eeg_grid
            hbo_ch, hbo_f, _ = schema.hbo_grid
            n_chrom = 2 if schema.hbr_cols else 1
            self.eeg_encoder = ModalityCNN(
                eeg_ch, eeg_f, enc_dim,
                filters=tuple(spec["cnn_filters"]), kernel_size=spec["cnn_kernel"],
                dropout=spec["encoder_dropout"], residual=spec["cnn_residual"],
            )
            self.fnirs_encoder = FNIRSGridCNN(
                hbo_ch, hbo_f, n_chrom, enc_dim,
                filters=tuple(spec["cnn_filters"]), kernel_size=spec["cnn_kernel"],
                dropout=spec["encoder_dropout"], residual=spec["cnn_residual"],
            )
        else:
            raise ValueError(f"unknown encoder '{kind}'")

        self.fusion = FUSIONS[spec["fusion"]](
            enc_dim, enc_dim, spec["encoder_dropout"]
        )

        head, in_d = [], enc_dim
        for units in spec["fc_units"]:
            head += [
                nn.Linear(in_d, units), nn.BatchNorm1d(units),
                nn.ReLU(), nn.Dropout(spec["fc_dropout"]),
            ]
            in_d = units
        head.append(nn.Linear(in_d, n_classes))
        self.head = nn.Sequential(*head)

        self.enc_dim = enc_dim
        self.domain_head = None
        # Probability of dropping each modality during TRAINING only.
        self.p_drop_eeg = float(spec.get("modality_dropout_eeg", 0.0))
        self.p_drop_fnirs = float(spec.get("modality_dropout_fnirs", 0.0))

    # ── modality dropout (training only) ───────────────────────────────
    def _modality_dropout(self, eeg_emb, fnirs_emb):
        """Randomly blank one modality's embedding, per sample.

        Never both at once (that would give the head no signal at all), and
        never outside training — `self.training` gates it, so validation, LOSO
        evaluation and the lockbox always see both modalities.
        """
        if not self.training or (self.p_drop_eeg <= 0 and self.p_drop_fnirs <= 0):
            return eeg_emb, fnirs_emb
        b = eeg_emb.shape[0]
        u = torch.rand(b, 1, device=eeg_emb.device)
        drop_eeg = (u < self.p_drop_eeg).float()
        drop_fnirs = ((u >= self.p_drop_eeg) & (u < self.p_drop_eeg + self.p_drop_fnirs)).float()
        return eeg_emb * (1 - drop_eeg), fnirs_emb * (1 - drop_fnirs)

    def encode(self, eeg, fnirs):
        e = self.eeg_encoder(eeg)
        f = self.fnirs_encoder(fnirs)
        e, f = self._modality_dropout(e, f)
        return self.fusion(e, f)

    def forward(self, eeg, fnirs):
        return self.head(self.encode(eeg, fnirs))

    def gates(self, eeg, fnirs):
        with torch.no_grad():
            e = self.eeg_encoder(eeg)
            f = self.fnirs_encoder(fnirs)
            g = self.fusion.gate_values(e, f)
            if g is None:
                return None
            return g[0].mean(dim=1), g[1].mean(dim=1)

    def embeddings(self, eeg, fnirs):
        with torch.no_grad():
            e = self.eeg_encoder(eeg)
            f = self.fnirs_encoder(fnirs)
            return self.fusion(e, f)

    # ── DANN ───────────────────────────────────────────────────────────
    def attach_domain_head(self, n_domains, hidden=64, dropout=0.5):
        self.domain_head = DomainDiscriminator(self.enc_dim, n_domains, hidden, dropout)
        return self.domain_head

    def forward_dann(self, eeg, fnirs, lambda_=1.0):
        if self.domain_head is None:
            raise RuntimeError("attach_domain_head() first")
        feat = self.encode(eeg, fnirs)
        return self.head(feat), self.domain_head(grad_reverse(feat, lambda_))


DEFAULT_SPEC = {
    "encoder": "mlp",
    "proj_dim": 96,
    "enc_dim": 48,
    "n_blocks": 2,
    "encoder_dropout": 0.5,
    "fusion": "attention",
    "fc_units": [48, 24],
    "fc_dropout": 0.6,
    "cnn_filters": [64, 64],
    "cnn_kernel": 3,
    "cnn_residual": False,
    "modality_dropout_eeg": 0.0,
    "modality_dropout_fnirs": 0.0,
}


def build_model(spec, schema, n_classes):
    full = dict(DEFAULT_SPEC)
    full.update(spec or {})
    return DualStreamNet(full, schema, n_classes)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))
