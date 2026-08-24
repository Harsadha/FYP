"""
Data model matching /schemas/impact_estimate.json exactly (frozen Day 1, 9:00).

    {
      "change_event_id": "string",
      "artifact_id": "string",
      "artifact_type": "CHUNK | EMBEDDING",
      "impact_score": "float [0,1]",
      "estimator": "TRACE | SIMILARITY",
      "confidence": "float [0,1]"
    }

Do not add/remove fields without a 3-way sync (per Day 0 ground rule) --
P1 and P3 build against this exact shape.
"""
from dataclasses import dataclass, asdict
import json

VALID_ARTIFACT_TYPES = ("CHUNK", "EMBEDDING")
VALID_ESTIMATORS = ("TRACE", "SIMILARITY")


@dataclass
class ImpactEstimate:
    change_event_id: str
    artifact_id: str
    artifact_type: str
    impact_score: float
    estimator: str
    confidence: float

    def __post_init__(self):
        if self.artifact_type not in VALID_ARTIFACT_TYPES:
            raise ValueError(
                f"artifact_type must be one of {VALID_ARTIFACT_TYPES}, got {self.artifact_type!r}"
            )
        if self.estimator not in VALID_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {VALID_ESTIMATORS}, got {self.estimator!r}"
            )
        if not (0.0 <= float(self.impact_score) <= 1.0):
            raise ValueError(f"impact_score must be in [0,1], got {self.impact_score}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
