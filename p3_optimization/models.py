"""
P3 models.

Reuses P2's ImpactEstimate exactly as defined in p2_estimation/models.py
-- it already matches /schemas/impact_estimate.json field-for-field, so
it is imported and re-exported here rather than redefined. Redefining
it would risk a silent drift between two "ImpactEstimate" classes with
the same name but different validation -- exactly what the Day-0 3-way
schema-sync rule exists to prevent.

MaintenancePlan is new here, matching /schemas/maintenance_plan.json:

    {
      "artifact_id": "string",
      "action": "update | invalidate | retain",
      "reason": "string"
    }

Deliberately plain dataclasses, matching p2_estimation/models.py's own
convention. Pydantic is not used anywhere in the existing repo, so
introducing it here would add a project dependency with no precedent
and no functional need -- the validation Pydantic would give us is
already done by hand in __post_init__, same as P2's ImpactEstimate.

KNOWN SCHEMA GAP (documented, not silently patched): earlier planning
discussion referenced a 5-action space (update/re-embed/invalidate/
retain/rebuild) and a GENERATED artifact type. The CURRENT, frozen
/schemas/maintenance_plan.json and /schemas/impact_estimate.json only
define 3 actions and 2 artifact types. This file implements strictly
to the current frozen schema. Extending it is a 3-way schema change,
not a P3-only decision.
"""
from dataclasses import dataclass, asdict, field
from typing import List
import json

# Re-exported, not redefined -- see module docstring.
from p2_estimation.models import ImpactEstimate  # noqa: F401

VALID_ACTIONS = ("update", "invalidate", "retain")


@dataclass
class MaintenancePlan:
    artifact_id: str
    action: str
    reason: str

    def __post_init__(self):
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {VALID_ACTIONS}, got {self.action!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AggregatedEstimate:
    """
    P3-internal type, NOT part of the shared schema. When multiple
    estimators (TRACE, SIMILARITY, ...) each produce an ImpactEstimate
    for the same artifact, the optimizer needs one combined view to
    decide an action from. This holds that combined view plus the raw
    inputs it was built from, so the aggregation is auditable rather
    than opaque.

    The combination strategy used to produce this (see optimizer.py's
    aggregate_estimates()) is a documented prototype heuristic --
    confidence-weighted average of impact_score, max of confidence --
    not a scientifically validated fusion method.
    """
    change_event_id: str
    artifact_id: str
    artifact_type: str
    combined_score: float
    combined_confidence: float
    source_estimators: List[str] = field(default_factory=list)
    raw_estimates: List[ImpactEstimate] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_estimates"] = [e.to_dict() for e in self.raw_estimates]
        return d