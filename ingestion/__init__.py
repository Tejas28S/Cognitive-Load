"""
ingestion: the dataset-agnostic input layer.

    raw file (any supported format)
        -> DatasetParser      (reads it; format detection)
        -> SchemaNormalizer   (maps names onto the canonical schema)
        -> ChannelProjector   (maps canonical channels onto a model bundle)
        -> existing feature extraction + existing frozen model  [UNCHANGED]

Everything downstream of the canonical form is untouched by this package: the
model weights, architecture, and LOSO evaluation logic are exactly as they were.
"""

from .canonical import (
    AliasIndex,
    CanonicalRecording,
    IngestionError,
    load_alias_registry,
    load_canonical_schema,
    load_dataset_profiles,
    normalize_token,
    validate_against_bundle,
)
from .parser import DatasetParser, ParseResult, beer_lambert

__all__ = [
    "AliasIndex",
    "CanonicalRecording",
    "DatasetParser",
    "IngestionError",
    "ParseResult",
    "beer_lambert",
    "load_alias_registry",
    "load_canonical_schema",
    "load_dataset_profiles",
    "normalize_token",
    "validate_against_bundle",
]
