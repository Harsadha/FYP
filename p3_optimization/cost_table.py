"""
P3 - Configurable maintenance-action cost table.

These are prototype defaults only, NOT experimentally measured. Real
costs (compute/latency/$-equivalent) should replace these once the
executor (P1) reports actual timings back through the feedback loop --
tracked as a TODO, not implemented here (see README).

Scoped strictly to the CURRENT, frozen schemas:
  artifact_type: CHUNK | EMBEDDING   (per /schemas/impact_estimate.json)
  action:        update | invalidate | retain
                 (per /schemas/maintenance_plan.json)

A GENERATED artifact type and re-embed/rebuild actions were discussed
in earlier architecture drafts but are NOT part of the current frozen
schema, so they are intentionally absent here -- see models.py's
"KNOWN SCHEMA GAP" note.
"""
from dataclasses import dataclass, field
from typing import Dict

DEFAULT_COSTS: Dict[str, Dict[str, float]] = {
    "CHUNK": {"update": 3.0, "invalidate": 1.0, "retain": 0.0},
    "EMBEDDING": {"update": 1.0, "invalidate": 0.5, "retain": 0.0},
}


@dataclass
class CostTable:
    """Mutable, configurable cost table -- pass a custom `costs` dict
    to override defaults (e.g. from a config file or calibration run)."""

    costs: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_COSTS.items()}
    )

    def cost(self, artifact_type: str, action: str) -> float:
        try:
            return self.costs[artifact_type][action]
        except KeyError as exc:
            raise ValueError(
                f"No cost configured for artifact_type={artifact_type!r}, "
                f"action={action!r}. Known artifact types: {list(self.costs)}"
            ) from exc

    def set_cost(self, artifact_type: str, action: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"cost must be >= 0, got {value}")
        self.costs.setdefault(artifact_type, {})[action] = value

    def known_artifact_types(self) -> list:
        return list(self.costs.keys())