"""
normalizer.py: SchemaNormalizer — map whatever was read onto the canonical schema.

Two entry points, one for each thing the parser can produce:

    normalize_features(df, ...)    an already-extracted feature table
    normalize_recording(rec)       a raw continuous recording

Both return `(normalized_thing, MappingReport)`. The report is not decoration —
it is returned alongside every prediction so a caller can see exactly how the
input was interpreted, which columns were ignored, and which canonical fields
had to be filled in. A prediction whose provenance cannot be inspected is not
trustworthy, so the report travels with it.

Matching policy (enforced here, documented in schema/channel_aliases.yaml):
exact normalized match, then a modality-tag-stripped retry, then UNMATCHED.
There is no edit-distance fallback anywhere. If a column cannot be resolved
confidently it is reported, never guessed — a wrong-but-confident mapping
produces a silently wrong prediction, which is the one outcome worth failing to
avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .canonical import (
    AliasIndex,
    CanonicalRecording,
    IngestionError,
    load_canonical_schema,
    normalize_token,
)

# Structured feature-name vocabulary. Feature columns are generated names
# (`eeg_ch00_alpha_mean`), not human-authored ones, so they are parsed
# structurally rather than listed one-by-one in the alias registry.
_EEG_BANDS = {"delta", "theta", "alpha", "beta", "gamma", "engage",
              "engagement", "engag"}
_BAND_CANON = {"engagement": "engage", "engag": "engage"}
_EEG_STATS = {"mean", "std", "slope"}
_FNIRS_STATS = {"mean", "std", "max", "min", "ptp", "skew", "kurt", "kurtosis",
                "slope", "early", "late", "delta", "power", "auc"}
_STAT_CANON = {"kurtosis": "kurt"}
_MODALITIES = {"eeg": "eeg", "hbo": "hbo", "hbr": "hbr",
               "oxy": "hbo", "deoxy": "hbr", "hb": "hbr"}


@dataclass
class MappingReport:
    """How an input was interpreted. Travels with every prediction."""
    dataset: str | None = None
    # original column/channel name -> canonical name, or "UNMATCHED"
    mapping: dict[str, str] = field(default_factory=dict)
    unmatched_input: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    filled_defaults: dict[str, str] = field(default_factory=dict)
    approximate: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self):
        return not self.missing_required

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "mapping": self.mapping,
            "unmatched_input": self.unmatched_input,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "filled_defaults": self.filled_defaults,
            "approximate": self.approximate,
            "warnings": self.warnings,
            "coverage": self.coverage,
            "notes": self.notes,
            "ok": self.ok,
        }


class SchemaNormalizer:

    def __init__(self, alias_index=None, canonical_schema=None):
        self.aliases = alias_index or AliasIndex()
        self.schema = canonical_schema or load_canonical_schema()

    # ── Feature-table path ──────────────────────────────────────────────────

    def _tokens(self, name):
        return [t for t in re.split(r"[\s_.\-/()\[\]]+", str(name).strip().lower()) if t]

    def parse_feature_column(self, name):
        """Parse a generated feature column name into its canonical spelling.

        Tolerates case, separator style, zero-padding, and token order:
        `EEG_CH0_Alpha_Mean`, `eeg.ch00.alpha.mean` and `eeg_ch00_alpha_mean`
        all resolve to `eeg_ch00_alpha_mean`. Returns None if the name is not a
        feature column at all.
        """
        toks = self._tokens(name)
        if not toks:
            return None

        # Pass 1: modality and channel index. Both are unambiguous, and the
        # modality is what disambiguates everything else.
        modality = channel = None
        rest = []
        for t in toks:
            if modality is None and t in _MODALITIES:
                modality = _MODALITIES[t]
                continue
            m = re.fullmatch(r"(?:ch|chan|channel|c)?(\d+)", t)
            if m and channel is None and t not in _EEG_BANDS:
                channel = int(m.group(1))
                continue
            rest.append(t)

        if modality is None or channel is None:
            return None

        # Pass 2: `delta` is BOTH an EEG band and an fNIRS stat, so it can only
        # be read once the modality is known. Classifying it eagerly in one pass
        # silently loses every `{hbo,hbr}_chNN_delta` column.
        if modality == "eeg":
            band = next((_BAND_CANON.get(t, t) for t in rest if t in _EEG_BANDS), None)
            stat = next((_STAT_CANON.get(t, t) for t in rest
                         if t in _EEG_STATS and t not in _EEG_BANDS), None)
            if band is None or stat is None:
                return None
            return f"eeg_ch{channel:02d}_{band}_{stat}"

        stat = next((_STAT_CANON.get(t, t) for t in rest if t in _FNIRS_STATS), None)
        if stat is None:
            return None
        return f"{modality}_ch{channel:02d}_{stat}"

    def normalize_features(self, df, target_columns, label_column="Cognitive_Load",
                           subject_column="subject_id", dataset=None,
                           optional_ignored_prefixes=("hbr_",)):
        """Rename/reorder a raw feature table onto the bundle's exact schema.

        `target_columns` is the bundle's required feature column list (from
        serving/feature_schema.json), so this function stays bundle-driven and
        needs no change when a differently-shaped bundle is added.
        """
        report = MappingReport(dataset=dataset)
        target = list(target_columns)
        target_set = set(target)

        renames, seen = {}, {}
        for col in df.columns:
            col = str(col)

            canon_feat = self.parse_feature_column(col)
            if canon_feat is not None:
                report.mapping[col] = canon_feat
                if canon_feat in seen:
                    report.warnings.append(
                        f"Columns '{seen[canon_feat]}' and '{col}' both resolve to "
                        f"'{canon_feat}'; kept the first."
                    )
                    continue
                seen[canon_feat] = col
                renames[col] = canon_feat
                continue

            match = self.aliases.resolve(col)
            if match.matched and match.kind == "metadata":
                report.mapping[col] = match.canonical
                renames[col] = match.canonical
                continue

            report.mapping[col] = "UNMATCHED"
            report.unmatched_input.append(col)

        out = df.rename(columns=renames)
        # A rename can collide with a column that was already correctly named.
        out = out.loc[:, ~out.columns.duplicated(keep="first")]

        present = set(out.columns)
        missing = [c for c in target if c not in present]
        if missing:
            report.missing_required = missing

        # Optional-but-known columns: report, do not fail.
        for opt, canon in (("label", label_column), ("subject", subject_column)):
            if canon not in present:
                report.missing_optional.append(canon)

        matched_features = sum(1 for c in target if c in present)
        report.coverage["features"] = matched_features / len(target) if target else 0.0
        report.notes["n_target_features"] = len(target)
        report.notes["n_matched_features"] = matched_features
        report.notes["n_input_columns"] = int(df.shape[1])

        if report.unmatched_input:
            ignorable = [
                c for c in report.unmatched_input
                if str(c).lower().startswith(tuple(optional_ignored_prefixes))
            ]
            truly_unknown = [c for c in report.unmatched_input if c not in ignorable]
            if truly_unknown:
                report.warnings.append(
                    f"{len(truly_unknown)} column(s) were not recognized and are "
                    f"ignored: {truly_unknown[:10]}"
                    + (" ..." if len(truly_unknown) > 10 else "")
                )

        # Reorder: canonical features first, then recognized metadata, then the
        # rest (kept, so nothing is silently dropped without a trace).
        ordered = [c for c in target if c in present]
        meta = [c for c in (subject_column, label_column) if c in present]
        rest = [c for c in out.columns if c not in set(ordered) | set(meta)]
        out = out[ordered + meta + rest]

        return out, report

    # ── Raw-recording path ──────────────────────────────────────────────────

    def normalize_recording(self, rec: CanonicalRecording):
        """Resolve a recording's channel names to canonical electrode names."""
        report = MappingReport(dataset=rec.dataset)
        report.warnings.extend(rec.warnings)
        report.notes["provenance"] = rec.provenance

        if rec.eeg is not None:
            canonical_names, unmatched = [], []
            for ch in rec.eeg_channels:
                match = self.aliases.resolve(ch)
                if match.matched and match.kind == "eeg_electrode":
                    report.mapping[ch] = match.canonical
                    canonical_names.append(match.canonical)
                else:
                    report.mapping[ch] = "UNMATCHED"
                    unmatched.append(ch)
                    # Keep the original name so the channel is still addressable;
                    # projection will simply never select it.
                    canonical_names.append(str(ch))
            rec.eeg_channels = canonical_names
            report.unmatched_input.extend(unmatched)
            if unmatched:
                report.warnings.append(
                    f"{len(unmatched)} EEG channel name(s) not in the alias "
                    f"registry, carried through unmapped: {unmatched[:10]}"
                )
            report.notes["n_eeg_channels"] = len(canonical_names)

        if rec.hbo is not None:
            report.notes["n_fnirs_channels"] = int(rec.hbo.shape[1])
            report.notes["fnirs_identity"] = "positional (no montage metadata)"

        if rec.subject_id is None:
            report.missing_optional.append("subject_id")
            report.warnings.append(
                "No subject id was recoverable from the source. The model's "
                "per-subject normalization will treat the whole upload as one "
                "subject, which is a documented degradation."
            )

        return rec, report

    # ── Label mapping ───────────────────────────────────────────────────────

    def map_label_values(self, values, mapping_name):
        """Map a dataset's raw label values onto canonical class indices.

        Unmappable values become -1 rather than a guess; the caller decides
        whether that is fatal.
        """
        table = self.aliases.label_value_map(mapping_name)
        if not table:
            raise IngestionError(
                f"No label value map named '{mapping_name}' in "
                f"schema/channel_aliases.yaml:label_values.",
                {"requested": mapping_name},
            )
        out, unmapped = [], set()
        for v in values:
            key = str(v).strip().lower()
            if key in table:
                out.append(table[key])
            else:
                out.append(-1)
                unmapped.add(str(v))
        return np.asarray(out, dtype=np.int64), sorted(unmapped)
