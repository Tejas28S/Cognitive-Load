"""
research/ — leakage-audited experiment framework for the binary
(and 3-class) EEG+fNIRS cognitive-load models.

This package is additive. It does not modify the frozen serving path
(serving/, serving_3class/, backend/model_service.py) and does not import
the legacy training scripts, so the existing app keeps working unchanged.

Every number this package produces is computed from actual model
predictions saved to disk under outputs/runs/<run_id>/.
"""

__all__ = [
    "runctx",
    "datamod",
    "leakage",
    "archs",
    "train",
    "metrics",
    "plots",
    "explain",
    "experiment",
    "registry",
]
