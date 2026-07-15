"""Pydantic request/response models for the cognitive-load demo API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class Health(BaseModel):
    status: str


class PredictRequest(BaseModel):
    # Provide EITHER a demo pointer (subject_id + window_index) OR raw features.
    subject_id: Optional[int] = None
    window_index: Optional[int] = None
    features: Optional[List[float]] = None


class SessionRequest(BaseModel):
    subject_id: int
    trial_id: int = 0


class AttentionWeights(BaseModel):
    eeg: float
    fnirs: float
    eeg_share: float
    fnirs_share: float


class Uncertainty(BaseModel):
    confidence: float
    margin: float
    entropy: float
    disagreement: Optional[float] = None
    abstain: bool
    abstain_confidence_threshold: float
    abstain_margin_threshold: float


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_index: int
    probabilities: Dict[str, float]
    attention_weights: AttentionWeights
    uncertainty: Optional[Uncertainty] = None
    true_label: Optional[int] = None
    true_label_name: Optional[str] = None
    window_index: Optional[int] = None


class ConfusionMatrix(BaseModel):
    model: str
    labels: List[str]
    matrix: List[List[int]]
