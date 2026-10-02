"""
canonical.py: schema loading + the name-matching primitives shared by the
parser, the normalizer, and the projection step.

Everything here is driven by the plain-config files under schema/ —
canonical_schema.yaml, channel_aliases.yaml, and schema/datasets/*.yaml. No
channel name, alias, or label mapping is hard-coded in Python, so onboarding a
dataset whose naming is a variation on known aliases is a config edit.

The one thing this module DOES enforce in code is the matching policy: exact
normalized match, then a modality-tag strip retry, then UNMATCHED. There is
deliberately no edit-distance / fuzzy fallback — a wrong-but-confident channel
match silently corrupts a prediction, which is worse than a clear miss.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.environ.get("COGLOAD_SCHEMA_DIR", os.path.join(_ROOT, "schema"))


# ── Schema loading ──────────────────────────────────────────────────────────

def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_canonical_schema(schema_dir=SCHEMA_DIR):
    return _read_yaml(os.path.join(schema_dir, "canonical_schema.yaml"))


def load_alias_registry(schema_dir=SCHEMA_DIR):
    return _read_yaml(os.path.join(schema_dir, "channel_aliases.yaml"))


def load_dataset_profiles(schema_dir=SCHEMA_DIR):
    """Every schema/datasets/*.yaml, keyed by its `name`."""
    d = os.path.join(schema_dir, "datasets")
    profiles = {}
    if not os.path.isdir(d):
        return profiles
    for fn in sorted(os.listdir(d)):
        if fn.endswith((".yaml", ".yml")):
            prof = _read_yaml(os.path.join(d, fn))
            if prof and prof.get("name"):
                profiles[prof["name"]] = prof
    return profiles


def validate_against_bundle(schema_dir=SCHEMA_DIR, serving_dir=None):
    """Assert canonical_schema.yaml still describes the live serving bundle.

    The bundle is the source of truth at prediction time; this file is the
    human-readable contract. If someone re-exports a bundle with a different
    feature count, we want a loud error here rather than a silent shape
    mismatch deep inside the model.
    """
    serving_dir = serving_dir or os.path.join(_ROOT, "serving")
    canon = load_canonical_schema(schema_dir)
    with open(os.path.join(serving_dir, "feature_schema.json"), encoding="utf-8") as f:
        bundle = json.load(f)

    problems = []
    fl = canon["feature_level"]
    if fl["eeg"]["n_features_total"] != len(bundle["eeg_columns"]):
        problems.append(
            f"EEG feature count: schema says {fl['eeg']['n_features_total']}, "
            f"bundle has {len(bundle['eeg_columns'])}"
        )
    if fl["fnirs"]["n_features_total"] != len(bundle["fnirs_columns"]):
        problems.append(
            f"fNIRS feature count: schema says {fl['fnirs']['n_features_total']}, "
            f"bundle has {len(bundle['fnirs_columns'])}"
        )
    if fl["metadata_columns"]["label_column"] != bundle["label_column"]:
        problems.append("label_column mismatch")
    if fl["metadata_columns"]["subject_column"] != bundle["subject_column"]:
        problems.append("subject_column mismatch")

    # The preprocessing recipe must match exactly — the saved scalers were fit
    # under it, so any drift silently invalidates every prediction.
    for key, want in fl["preprocessing"].items():
        got = bundle["preprocessing"].get(key)
        if isinstance(want, float) or isinstance(got, float):
            if got is None or abs(float(got) - float(want)) > 1e-15:
                problems.append(f"preprocessing.{key}: schema {want!r} != bundle {got!r}")
        elif got != want:
            problems.append(f"preprocessing.{key}: schema {want!r} != bundle {got!r}")

    return (len(problems) == 0), problems


# ── Name normalization (matching rules 1 and 3) ─────────────────────────────

_SEPARATORS = re.compile(r"[\s_.\-/()\[\]]+")


def normalize_token(name, unit_suffixes=(), modality_tags=(), strip_tags=False):
    """Collapse a column/channel name to its comparable form.

    Rule 1: lowercase, strip separators, strip a trailing unit suffix.
    Rule 3 (only when strip_tags=True): additionally strip leading/trailing
    modality tags such as `eeg`, `ch`, `channel`.

    `EEG_Fp1`, `eeg.fp1`, `Fp1_uV` and `FP1` all reduce to `fp1`.
    """
    if name is None:
        return ""
    s = _SEPARATORS.sub("", str(name).strip().lower())

    for suf in sorted(unit_suffixes, key=len, reverse=True):
        suf = _SEPARATORS.sub("", str(suf).lower())
        if suf and s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)]
            break

    if strip_tags:
        changed = True
        while changed:
            changed = False
            for tag in sorted(modality_tags, key=len, reverse=True):
                tag = _SEPARATORS.sub("", str(tag).lower())
                if not tag or len(s) <= len(tag):
                    continue
                if s.startswith(tag):
                    s, changed = s[len(tag):], True
                    break
                if s.endswith(tag):
                    s, changed = s[: -len(tag)], True
                    break
    return s


@dataclass
class Match:
    """One resolved (or unresolved) name, carrying how it was resolved."""
    original: str
    canonical: str | None
    kind: str            # eeg_electrode | fnirs_chromophore | metadata | wavelength
    how: str             # exact | tag_stripped | alias | unmatched
    detail: str = ""

    @property
    def matched(self):
        return self.canonical is not None


class AliasIndex:
    """Lookup from any known spelling to its canonical name.

    Built once from channel_aliases.yaml. Every entry records what kind of thing
    it is so the normalizer can tell an electrode from a metadata field without
    a second pass.
    """

    def __init__(self, registry=None):
        self.registry = registry or load_alias_registry()
        self.unit_suffixes = self.registry.get("unit_suffixes", []) or []
        self.modality_tags = self.registry.get("modality_tags", []) or []

        self._index: dict[str, tuple[str, str]] = {}   # token -> (canonical, kind)
        self._build()

    def _add(self, spelling, canonical, kind):
        for strip in (False, True):
            tok = normalize_token(
                spelling, self.unit_suffixes, self.modality_tags, strip_tags=strip
            )
            if tok and tok not in self._index:
                self._index[tok] = (canonical, kind)

    def _build(self):
        for canon, spec in (self.registry.get("eeg_electrodes") or {}).items():
            self._add(canon, canon, "eeg_electrode")
            for a in ((spec or {}).get("aliases") or []):
                self._add(a, canon, "eeg_electrode")

        fnirs = self.registry.get("fnirs") or {}
        for chrom, spec in (fnirs.get("chromophores") or {}).items():
            self._add(chrom, chrom, "fnirs_chromophore")
            for a in ((spec or {}).get("aliases") or []):
                self._add(a, chrom, "fnirs_chromophore")
        for wl, spec in (fnirs.get("wavelengths") or {}).items():
            self._add(wl, wl, "wavelength")
            for a in ((spec or {}).get("aliases") or []):
                self._add(a, wl, "wavelength")

        for fieldname, spec in (self.registry.get("metadata") or {}).items():
            canon = (spec or {}).get("canonical", fieldname)
            self._add(fieldname, canon, "metadata")
            self._add(canon, canon, "metadata")
            for a in ((spec or {}).get("aliases") or []):
                self._add(a, canon, "metadata")

    def resolve(self, name):
        """Resolve one name to a Match. Never guesses."""
        tok = normalize_token(name, self.unit_suffixes, self.modality_tags)
        if tok in self._index:
            canon, kind = self._index[tok]
            how = "exact" if tok == normalize_token(canon, self.unit_suffixes) else "alias"
            return Match(str(name), canon, kind, how)

        tok2 = normalize_token(
            name, self.unit_suffixes, self.modality_tags, strip_tags=True
        )
        if tok2 and tok2 in self._index:
            canon, kind = self._index[tok2]
            return Match(str(name), canon, kind, "tag_stripped")

        return Match(str(name), None, "unknown", "unmatched")

    # Spatial-neighbour candidates, in the preference order the registry declares.
    def neighbours(self, canonical_electrode):
        spec = (self.registry.get("eeg_electrodes") or {}).get(canonical_electrode) or {}
        return list(spec.get("spatial_neighbours") or [])

    def label_value_map(self, mapping_name):
        """Raw label values -> canonical class index, for a named dataset map."""
        maps = self.registry.get("label_values") or {}
        spec = maps.get(mapping_name) or {}
        return {str(k).strip().lower(): int(v) for k, v in (spec.get("map") or {}).items()}


# ── Canonical recording container ───────────────────────────────────────────

@dataclass
class CanonicalRecording:
    """A raw recording normalized to canonical signal-level form.

    This is the single hand-off point between "reading files" and "extracting
    features". Everything downstream consumes only this, so no downstream code
    ever needs to know what file format it came from.
    """
    eeg: np.ndarray | None = None            # (samples, channels), microvolts
    eeg_channels: list[str] = field(default_factory=list)   # canonical electrodes
    eeg_fs: float | None = None

    hbo: np.ndarray | None = None            # (samples, channels), micromolar
    hbr: np.ndarray | None = None
    fnirs_fs: float | None = None

    events: list[dict] = field(default_factory=list)        # {type, latency_sec, label}
    subject_id: str | None = None
    dataset: str | None = None

    # Everything the caller should know about HOW this was produced: assumed
    # constants, conversions applied, alignment caveats.
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self):
        return {
            "dataset": self.dataset,
            "subject_id": self.subject_id,
            "eeg": None if self.eeg is None else {
                "n_samples": int(self.eeg.shape[0]),
                "n_channels": int(self.eeg.shape[1]),
                "sample_rate_hz": self.eeg_fs,
                "channels": self.eeg_channels,
            },
            "fnirs": None if self.hbo is None else {
                "n_samples": int(self.hbo.shape[0]),
                "n_channels": int(self.hbo.shape[1]),
                "sample_rate_hz": self.fnirs_fs,
                "has_hbr": self.hbr is not None,
            },
            "n_events": len(self.events),
            "provenance": self.provenance,
            "warnings": self.warnings,
        }


class IngestionError(Exception):
    """Raised for actionable, user-facing ingestion failures.

    Carries a structured `detail` so the API layer can return it as JSON rather
    than a stack trace.
    """

    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail or {}
