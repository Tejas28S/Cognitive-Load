"""
projection.py: map a canonical recording's channels onto a model bundle's
expected channel layout.

This is the step where the honest limits of cross-dataset transfer get
quantified instead of hidden. The frozen model consumes a FIXED 30-EEG /
36-fNIRS channel layout in TU Berlin acquisition order. Another dataset has its
own montage. Some positions match exactly, some can be filled by a documented
spatial neighbour, and some cannot be filled at all.

Rather than zero-filling the gaps and returning a confident-looking number, this
module computes per-modality coverage and returns a verdict:

    confident  coverage >= 0.90            emit normally
    degraded   0.70 <= coverage < 0.90     emit, flagged, with the numbers attached
    refuse     coverage < 0.70             do NOT predict; return what is missing

Thresholds live in schema/canonical_schema.yaml (projection.thresholds), not
here, so the policy is editable without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .canonical import AliasIndex, CanonicalRecording, load_canonical_schema


@dataclass
class ChannelAssignment:
    position: str            # ch00 ...
    wanted: str              # canonical electrode the model expects there
    source: str | None       # channel actually used
    how: str                 # exact | neighbour | missing
    detail: str = ""


@dataclass
class ProjectionReport:
    modality: str
    assignments: list[ChannelAssignment] = field(default_factory=list)
    coverage: float = 0.0
    n_exact: int = 0
    n_neighbour: int = 0
    n_missing: int = 0
    verdict: str = "refuse"
    unused_source_channels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "modality": self.modality,
            "coverage": round(self.coverage, 4),
            "verdict": self.verdict,
            "n_exact": self.n_exact,
            "n_neighbour": self.n_neighbour,
            "n_missing": self.n_missing,
            "missing_positions": [
                {"position": a.position, "wanted": a.wanted}
                for a in self.assignments if a.how == "missing"
            ],
            "approximate_positions": [
                {"position": a.position, "wanted": a.wanted, "used": a.source}
                for a in self.assignments if a.how == "neighbour"
            ],
            "unused_source_channels": self.unused_source_channels,
            "warnings": self.warnings,
            "notes": self.notes,
        }


class ChannelProjector:

    def __init__(self, canonical_schema=None, alias_index=None):
        self.schema = canonical_schema or load_canonical_schema()
        self.aliases = alias_index or AliasIndex()
        proj = self.schema.get("projection", {})
        self.thresholds = proj.get("thresholds", {})
        self.substitution = proj.get("substitution", {})
        self.fill_policy = proj.get("fill_policy", {})

    # ── Verdict ─────────────────────────────────────────────────────────────

    def _verdict(self, coverage):
        confident = float(self.thresholds.get("confident_min_coverage", 0.90))
        refuse = float(self.thresholds.get("refuse_below", 0.70))
        if coverage >= confident:
            return "confident"
        if coverage >= refuse:
            return "degraded"
        return "refuse"

    # ── EEG ─────────────────────────────────────────────────────────────────

    def project_eeg(self, rec: CanonicalRecording):
        """Reorder/select EEG channels into the bundle's 30-position layout."""
        order = self.schema["feature_level"]["eeg"]["channel_order"]
        positions = sorted(order.keys())
        report = ProjectionReport(modality="eeg")

        if rec.eeg is None:
            report.warnings.append("Recording carries no EEG data.")
            report.n_missing = len(positions)
            report.verdict = "refuse"
            return None, report

        available = {}
        for i, ch in enumerate(rec.eeg_channels):
            available.setdefault(str(ch), i)

        allow_neighbours = bool(self.substitution.get("allow_spatial_neighbours", True))
        weight = float(self.substitution.get("neighbour_coverage_weight", 0.5))

        n_samples = rec.eeg.shape[0]
        out = np.full((n_samples, len(positions)), np.nan, dtype=np.float32)
        used_sources = set()

        for pi, pos in enumerate(positions):
            wanted = order[pos]
            if wanted in available:
                idx = available[wanted]
                out[:, pi] = rec.eeg[:, idx]
                report.assignments.append(
                    ChannelAssignment(pos, wanted, wanted, "exact"))
                report.n_exact += 1
                used_sources.add(wanted)
                continue

            picked = None
            if allow_neighbours:
                for nb in self.aliases.neighbours(wanted):
                    if nb in available:
                        picked = nb
                        break

            if picked is not None:
                out[:, pi] = rec.eeg[:, available[picked]]
                report.assignments.append(ChannelAssignment(
                    pos, wanted, picked, "neighbour",
                    f"'{wanted}' absent; substituted spatial neighbour '{picked}'"))
                report.n_neighbour += 1
                used_sources.add(picked)
            else:
                # Left as NaN, never 0.0 — after the per-subject z-score, 0.0 is
                # a legitimate value and a zero-fill would be indistinguishable
                # from real data.
                report.assignments.append(
                    ChannelAssignment(pos, wanted, None, "missing"))
                report.n_missing += 1

        report.coverage = (report.n_exact + weight * report.n_neighbour) / len(positions)
        report.verdict = self._verdict(report.coverage)
        report.unused_source_channels = sorted(set(available) - used_sources)
        report.notes = {
            "n_positions": len(positions),
            "n_source_channels": len(available),
            "neighbour_coverage_weight": weight,
        }
        if report.n_neighbour:
            report.warnings.append(
                f"{report.n_neighbour} of {len(positions)} EEG positions were "
                f"filled by spatial-neighbour substitution. These are "
                f"APPROXIMATE — the electrode the model was trained on is not "
                f"present in this dataset."
            )
        if report.n_missing:
            report.warnings.append(
                f"{report.n_missing} EEG position(s) could not be filled at all "
                f"and are NaN."
            )
        return out, report

    # ── fNIRS ───────────────────────────────────────────────────────────────

    def project_fnirs(self, rec: CanonicalRecording, dataset_profile=None):
        """Map fNIRS channels onto the bundle's 36 optode positions.

        Unlike EEG, fNIRS channels carry NO recoverable spatial identity in the
        serving bundle — they are bare optode-pair indices with no montage
        metadata (see docs/tu_berlin_assumptions.md §0). A positional mapping is
        therefore only performed when the dataset profile declares an explicit
        one. Absent that, coverage is 0 and the caller refuses, because inventing
        a correspondence would produce 936 features of fiction.
        """
        n_positions = int(self.schema["feature_level"]["fnirs"]["n_channels"])
        report = ProjectionReport(modality="fnirs")

        if rec.hbo is None:
            report.warnings.append("Recording carries no fNIRS data.")
            report.n_missing = n_positions
            report.verdict = "refuse"
            return None, None, report

        n_src = int(rec.hbo.shape[1])
        report.notes = {"n_positions": n_positions, "n_source_channels": n_src}

        explicit = None
        if dataset_profile:
            explicit = (dataset_profile.get("fnirs") or {}).get("montage_map")

        if explicit is None:
            if n_src == n_positions:
                # Same count AND the dataset is the reference montage: identity
                # is the correct mapping, not an assumption.
                if dataset_profile and dataset_profile.get("is_reference_dataset"):
                    hbo = np.asarray(rec.hbo, dtype=np.float32)
                    hbr = (np.asarray(rec.hbr, dtype=np.float32)
                           if rec.hbr is not None else np.zeros_like(hbo))
                    for i in range(n_positions):
                        report.assignments.append(
                            ChannelAssignment(f"ch{i:02d}", f"ch{i:02d}",
                                              f"ch{i:02d}", "exact"))
                    report.n_exact = n_positions
                    report.coverage = 1.0
                    report.verdict = self._verdict(1.0)
                    return hbo, hbr, report

                report.warnings.append(
                    f"This dataset supplies {n_src} fNIRS channels, matching the "
                    f"model's {n_positions} by COUNT — but channel count is not "
                    f"channel identity. Without a declared montage map the "
                    f"correspondence is unknown, so it is not assumed."
                )
            else:
                report.warnings.append(
                    f"This dataset supplies {n_src} fNIRS channels; the model "
                    f"expects {n_positions}. No montage map is declared in the "
                    f"dataset profile, and optode positions carry no recoverable "
                    f"identity in the serving bundle, so no defensible "
                    f"correspondence exists."
                )
            report.n_missing = n_positions
            report.coverage = 0.0
            report.verdict = "refuse"
            report.notes["reason"] = "no_montage_map"
            return None, None, report

        # Explicit map: {model_position_index: source_channel_index}
        hbo = np.full((rec.hbo.shape[0], n_positions), np.nan, dtype=np.float32)
        hbr = np.full((rec.hbo.shape[0], n_positions), np.nan, dtype=np.float32)
        src_hbr = rec.hbr if rec.hbr is not None else None
        for pos_i in range(n_positions):
            src_i = explicit.get(pos_i, explicit.get(str(pos_i)))
            pos = f"ch{pos_i:02d}"
            if src_i is None or int(src_i) >= n_src:
                report.assignments.append(
                    ChannelAssignment(pos, pos, None, "missing"))
                report.n_missing += 1
                continue
            hbo[:, pos_i] = rec.hbo[:, int(src_i)]
            if src_hbr is not None:
                hbr[:, pos_i] = src_hbr[:, int(src_i)]
            report.assignments.append(
                ChannelAssignment(pos, pos, f"src{int(src_i):02d}", "exact",
                                  "explicit montage map"))
            report.n_exact += 1

        report.coverage = report.n_exact / n_positions
        report.verdict = self._verdict(report.coverage)
        report.notes["reason"] = "explicit_montage_map"
        return hbo, hbr, report

    # ── Resampling ──────────────────────────────────────────────────────────

    @staticmethod
    def resample(data, fs_in, fs_out):
        """Resample (samples, channels) to the bundle's rate.

        The model's band-power features were fit on 200 Hz EEG / 10 Hz fNIRS.
        Feeding a different rate changes the sub-window sample counts and the
        Welch resolution, so rate is matched rather than assumed harmless.
        """
        if data is None or fs_in is None or fs_out is None:
            return data, {"resampled": False}
        if abs(float(fs_in) - float(fs_out)) < 1e-9:
            return data, {"resampled": False, "fs": float(fs_in)}

        from fractions import Fraction
        frac = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
        out = resample_poly(np.nan_to_num(data, nan=0.0), frac.numerator,
                            frac.denominator, axis=0).astype(np.float32)
        # resample_poly cannot carry NaN, so restore all-NaN channels, which are
        # the "position could not be filled" markers from projection.
        if data.ndim == 2:
            all_nan = np.all(np.isnan(data), axis=0)
            if all_nan.any():
                out[:, all_nan] = np.nan
        return out, {
            "resampled": True, "fs_in": float(fs_in), "fs_out": float(fs_out),
            "ratio": f"{frac.numerator}/{frac.denominator}",
        }


def combine_verdicts(*reports):
    """Overall verdict across modalities: the worst one wins.

    The model consumes BOTH branches, so an unusable fNIRS side is not rescued
    by a good EEG side.
    """
    rank = {"confident": 0, "degraded": 1, "refuse": 2}
    worst, name = "confident", None
    for r in reports:
        if r is not None and rank[r.verdict] > rank[worst]:
            worst, name = r.verdict, r.modality
    return worst, name
