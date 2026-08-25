"""
P3 - Cost-quality constrained maintenance optimizer.

This is the main research component: given ImpactEstimate records
(possibly several per artifact, from different estimators), decide one
MaintenancePlan action per artifact that minimizes total maintenance
cost while keeping a quality-recovery proxy above a configurable
threshold.

Two implementations are provided, deliberately kept separate:

1. threshold_based_plan() -- a simple, fast baseline. Exists so the
   full pipeline (estimator -> plan -> executor -> oracle) works
   end-to-end from day one, and so the MIP has something to be
   compared against, not just something to replace.

2. optimize_maintenance_plan() -- the actual MIP (OR-Tools CP-SAT).
   This is the reformulation the project's contribution rests on:
   turning "which artifacts are impacted" into "which impacted
   artifacts actually need action, under a budget, without violating
   a quality floor."

IMPORTANT / honest scope note: the quality constraint below uses
impact_score itself (aggregated across estimators) as a PROXY for
"how much correctness is at risk if this artifact is left unfixed."
This is NOT the same as a real, measured RAG-correctness delta -- that
requires actually running the RAG pipeline and Oracle (oracle.py),
which is too expensive to do per-candidate inside the solver's inner
loop. The `current_quality` / `current_consistency` parameters below
are the documented interface point where oracle.py's real, periodically
-refreshed measurements are meant to be wired in later (mirrors the
Q_base/C_base pattern from the architecture spec) -- until that wiring
exists, the constraint is evaluated against the impact-score proxy
only. This is stated explicitly here so it is never silently confused
with a validated correctness guarantee.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

from ortools.sat.python import cp_model

from p2_estimation.models import ImpactEstimate
from p3_optimization.models import MaintenancePlan, AggregatedEstimate
from p3_optimization.cost_table import CostTable

# Quality-recovery efficacy per action: how much of an artifact's risk
# is resolved by taking this action. Prototype priors, not learned --
# same status as the cost table (see cost_table.py docstring).
DEFAULT_EFFICACY: Dict[str, float] = {
    "update": 1.0,
    "invalidate": 0.6,   # removes stale content from circulation without correcting it
    "retain": 0.0,
}

# Integer scaling factor for CP-SAT, which requires integer coefficients.
_SCALE = 1000


@dataclass
class OptimizationResult:
    """Wraps the plan with metadata about how it was produced, so
    callers (and tests) can tell a solved-optimal plan apart from a
    greedy fallback rather than treating them as equivalent."""

    plan: List[MaintenancePlan]
    solution_quality: str  # "optimal" | "feasible" | "greedy_fallback" | "empty"
    total_cost: float


def aggregate_estimates(estimates: List[ImpactEstimate]) -> List[AggregatedEstimate]:
    """
    Combines multiple ImpactEstimate records for the same artifact_id
    (e.g. one from TRACE, one from SIMILARITY) into a single
    AggregatedEstimate.

    Documented strategy (prototype, not scientifically validated):
      combined_score      = confidence-weighted average of impact_score
                             (falls back to plain average if all
                             confidences are 0)
      combined_confidence = max confidence across contributing estimators
                             (most-confident estimator sets the
                             confidence in the combined view)

    Deterministic: same input list always produces the same output,
    regardless of input order (grouping is by artifact_id, and the
    weighted-average formula is order-independent).
    """
    by_artifact: Dict[str, List[ImpactEstimate]] = defaultdict(list)
    for est in estimates:
        by_artifact[est.artifact_id].append(est)

    aggregated: List[AggregatedEstimate] = []
    for artifact_id, group in by_artifact.items():
        total_weight = sum(e.confidence for e in group)
        if total_weight > 0:
            combined_score = sum(e.impact_score * e.confidence for e in group) / total_weight
        else:
            combined_score = sum(e.impact_score for e in group) / len(group)
        combined_confidence = max(e.confidence for e in group)

        aggregated.append(
            AggregatedEstimate(
                change_event_id=group[0].change_event_id,
                artifact_id=artifact_id,
                artifact_type=group[0].artifact_type,
                combined_score=max(0.0, min(1.0, combined_score)),
                combined_confidence=combined_confidence,
                source_estimators=sorted({e.estimator for e in group}),
                raw_estimates=list(group),
            )
        )

    # Sort for deterministic ordering regardless of input/dict iteration order.
    aggregated.sort(key=lambda a: a.artifact_id)
    return aggregated


def threshold_based_plan(
    estimates: List[ImpactEstimate],
    update_threshold: float = 0.8,
    invalidate_threshold: float = 0.4,
) -> List[MaintenancePlan]:
    """
    Baseline: no optimization, just thresholds on the aggregated score.
      combined_score >= update_threshold      -> update
      combined_score >= invalidate_threshold   -> invalidate
      otherwise                                -> retain

    Exists so the pipeline works end-to-end before the MIP is trusted,
    and as a documented, always-available fallback (see
    optimize_maintenance_plan's solver-failure handling).
    """
    if not (0.0 <= invalidate_threshold <= update_threshold <= 1.0):
        raise ValueError(
            "require 0 <= invalidate_threshold <= update_threshold <= 1, "
            f"got invalidate_threshold={invalidate_threshold}, update_threshold={update_threshold}"
        )

    aggregated = aggregate_estimates(estimates)
    plan: List[MaintenancePlan] = []
    for agg in aggregated:
        estimator_list = "+".join(agg.source_estimators)
        if agg.combined_score >= update_threshold:
            action, reason = "update", (
                f"combined_score={agg.combined_score:.3f} >= update_threshold="
                f"{update_threshold} (estimators: {estimator_list})"
            )
        elif agg.combined_score >= invalidate_threshold:
            action, reason = "invalidate", (
                f"combined_score={agg.combined_score:.3f} >= invalidate_threshold="
                f"{invalidate_threshold} (estimators: {estimator_list})"
            )
        else:
            action, reason = "retain", (
                f"combined_score={agg.combined_score:.3f} below invalidate_threshold="
                f"{invalidate_threshold} (estimators: {estimator_list})"
            )
        plan.append(MaintenancePlan(artifact_id=agg.artifact_id, action=action, reason=reason))
    return plan


def optimize_maintenance_plan(
    estimates: List[ImpactEstimate],
    cost_table: Optional[CostTable] = None,
    quality_threshold: float = 0.8,
    consistency_threshold: float = 0.8,  # accepted for interface completeness; see TODO below
    efficacy: Optional[Dict[str, float]] = None,
    current_quality: Optional[float] = None,
    current_consistency: Optional[float] = None,
    solver_time_limit_seconds: float = 5.0,
) -> OptimizationResult:
    """
    CP-SAT MIP formulation.

    Variables: x[i, a] in {0, 1} for each artifact i and action a in
               {update, invalidate, retain}.
    Constraint: sum_a x[i, a] == 1 for all i (exactly one action per artifact).
    Objective:  minimize sum_i sum_a cost(type_i, a) * x[i, a].
    Quality constraint (proxy, see module docstring):
        sum_i impact_i * efficacy(a) * x[i, a]  >=  quality_threshold * sum_i impact_i

    consistency_threshold / current_quality / current_consistency are
    accepted now so the calling interface will not need to change once
    oracle.py's real periodic measurements are wired in (mirrors the
    architecture spec's Q_base/C_base refresh pattern), but they are
    NOT YET used as an active solver constraint -- doing so correctly
    requires real, per-artifact consistency-recovery data from the
    Oracle, which does not exist until an actual RAG run has happened.
    TODO(P3, post-Review-1): once oracle.py has scored at least one
    real before/after scenario, extend this constraint using measured
    contradiction-rate deltas instead of the impact-score proxy alone.

    Failure handling: if the solver does not reach OPTIMAL/FEASIBLE
    within solver_time_limit_seconds, falls back to
    threshold_based_plan() and reports solution_quality="greedy_fallback"
    -- this fallback is always logged in the result, never silently
    presented as an optimal solution.

    Deterministic: fixes num_search_workers=1 and a constant random
    seed so repeated calls on identical input produce identical output
    (required for reproducible tests).
    """
    if cost_table is None:
        cost_table = CostTable()
    if efficacy is None:
        efficacy = dict(DEFAULT_EFFICACY)
    if not (0.0 <= quality_threshold <= 1.0):
        raise ValueError(f"quality_threshold must be in [0,1], got {quality_threshold}")

    aggregated = aggregate_estimates(estimates)
    if not aggregated:
        return OptimizationResult(plan=[], solution_quality="empty", total_cost=0.0)

    actions = ("update", "invalidate", "retain")
    model = cp_model.CpModel()

    x: Dict[tuple, cp_model.IntVar] = {}
    for agg in aggregated:
        for action in actions:
            x[(agg.artifact_id, action)] = model.NewBoolVar(f"x_{agg.artifact_id}_{action}")

    # Exactly one action per artifact.
    for agg in aggregated:
        model.Add(sum(x[(agg.artifact_id, a)] for a in actions) == 1)

    # Objective: minimize total (integer-scaled) cost.
    cost_terms = []
    for agg in aggregated:
        for action in actions:
            scaled_cost = round(cost_table.cost(agg.artifact_type, action) * _SCALE)
            cost_terms.append(scaled_cost * x[(agg.artifact_id, action)])
    model.Minimize(sum(cost_terms))

    # Quality-recovery proxy constraint (see docstring for scope caveat).
    # LHS: sum impact_int * efficacy_int * x   (scale = _SCALE^2)
    # RHS: quality_threshold_int * sum impact_int   (scale = _SCALE^2)
    impact_ints = {agg.artifact_id: round(agg.combined_score * _SCALE) for agg in aggregated}
    total_impact_int = sum(impact_ints.values())
    if total_impact_int > 0:
        quality_terms = []
        for agg in aggregated:
            for action in actions:
                efficacy_int = round(efficacy.get(action, 0.0) * _SCALE)
                quality_terms.append(impact_ints[agg.artifact_id] * efficacy_int * x[(agg.artifact_id, action)])
        threshold_int = round(quality_threshold * _SCALE)
        model.Add(sum(quality_terms) >= threshold_int * total_impact_int)
    # If total_impact_int == 0 (all estimates score 0), there is nothing
    # at risk, so no quality constraint is needed -- retain-everything
    # trivially satisfies "preserve quality."

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan: List[MaintenancePlan] = []
        total_cost = 0.0
        for agg in aggregated:
            chosen_action = next(a for a in actions if solver.Value(x[(agg.artifact_id, a)]) == 1)
            total_cost += cost_table.cost(agg.artifact_type, chosen_action)
            estimator_list = "+".join(agg.source_estimators)
            reason = (
                f"CP-SAT: combined_score={agg.combined_score:.3f}, "
                f"estimators={estimator_list}, quality_threshold={quality_threshold}"
            )
            plan.append(MaintenancePlan(artifact_id=agg.artifact_id, action=chosen_action, reason=reason))
        quality_label = "optimal" if status == cp_model.OPTIMAL else "feasible"
        return OptimizationResult(plan=plan, solution_quality=quality_label, total_cost=total_cost)

    # Solver failed to find a feasible solution in time -> documented fallback.
    fallback_plan = threshold_based_plan(estimates)
    fallback_plan = [
        MaintenancePlan(
            artifact_id=p.artifact_id,
            action=p.action,
            reason=f"[greedy_fallback: CP-SAT status={solver.StatusName(status)}] {p.reason}",
        )
        for p in fallback_plan
    ]
    total_cost = sum(
        cost_table.cost(
            next(a.artifact_type for a in aggregated if a.artifact_id == p.artifact_id),
            p.action,
        )
        for p in fallback_plan
    )
    return OptimizationResult(plan=fallback_plan, solution_quality="greedy_fallback", total_cost=total_cost)