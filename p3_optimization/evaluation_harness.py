"""
P3 - Evaluation harness.

Runs the full change -> estimate -> plan -> (simulated) execute ->
oracle loop against a mocked corpus/RAG setup, so it is fully testable
without a live Neo4j instance, Kafka, or a real LLM/RAG stack. This is
what makes P3's tests runnable standalone (see project rule: "initial
version may operate on mocked RAG responses").

Two things here are explicitly mocked, and this is documented rather
than hidden:

  1. The "RAG answer" for a question is just the current text of its
     relevant artifact(s) after a plan has been applied (or None if
     invalidated). oracle.py's F1/EM scoring is tolerant of this being
     a full sentence rather than a terse answer -- token overlap still
     works.

  2. Similarity-estimator embeddings are produced by a tiny local
     bag-of-words vectorizer, NOT P2's real sentence-transformer
     embedder (p2_estimation/embedder.py). This harness must run
     offline with no model downloads. Swap in P2's real embedder for
     a genuine evaluation run -- see build_expense_demo_scenario()'s
     docstring.

Compares 6 strategies over the same scenario: no_maintenance,
update_everything, trace_only, similarity_only, combined (threshold
baseline), optimized (CP-SAT).
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from p2_estimation.graph_client import MockGraphClient
from p2_estimation.similarity_estimator import estimate_impact as similarity_estimate_impact
from p3_optimization.dependency_tracer import trace_impact
from p3_optimization.models import MaintenancePlan
from p3_optimization.cost_table import CostTable
from p3_optimization.optimizer import threshold_based_plan, optimize_maintenance_plan
from p3_optimization import oracle as oracle_mod

STRATEGIES = (
    "no_maintenance",
    "update_everything",
    "trace_only",
    "similarity_only",
    "combined",
    "optimized",
)


@dataclass
class QAItem:
    question: str
    relevant_artifact_ids: List[str]  # >1 => this question also gets a consistency check
    gold_answer: str


@dataclass
class Scenario:
    name: str
    corpus: Dict[str, str]              # artifact_id -> current (pre-fix) text
    duplicates: Dict[str, List[str]]    # artifact_id -> artifact_ids reachable from it (graph edges)
    change_event: dict                  # ChangeEvent-shaped dict, matches /schemas/change_event.json
    updated_content: Dict[str, str]     # artifact_id -> correct text after proper propagation
    questions: List[QAItem]
    artifact_types: Dict[str, str] = field(default_factory=dict)  # defaults to CHUNK if absent


@dataclass
class QuestionResult:
    question: str
    quality: Optional[oracle_mod.QualityResult]
    consistency: Optional[oracle_mod.ConsistencyResult]


@dataclass
class EvaluationResult:
    strategy: str
    plan: List[MaintenancePlan]
    total_cost: float
    avg_f1: float
    exact_match_rate: float
    consistency_rate: float
    question_results: List[QuestionResult]


# --- tiny offline vectorizer, mock-only, see module docstring ---------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for text in texts:
        for tok in _tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _vectorize(text: str, vocab: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=float)
    for tok in _tokenize(text):
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    return vec


# --- corpus state / plan application -----------------------------------

def _apply_plan_to_corpus(scenario: Scenario, plan: List[MaintenancePlan]) -> Dict[str, Optional[str]]:
    """
    The source artifact itself always reflects the change (that's the
    definition of a ChangeEvent -- the source has already changed).
    Maintenance actions only govern what happens to OTHER (e.g.
    duplicate) artifacts:
      update      -> replaced with scenario.updated_content[id] if known
      invalidate  -> excluded from retrieval (None)
      retain      -> left as its current (possibly stale) text
    """
    state: Dict[str, Optional[str]] = dict(scenario.corpus)
    changed_id = scenario.change_event["source_artifact_id"]
    if changed_id in scenario.updated_content:
        state[changed_id] = scenario.updated_content[changed_id]

    for p in plan:
        if p.artifact_id == changed_id:
            continue  # source already handled above
        if p.action == "update":
            state[p.artifact_id] = scenario.updated_content.get(p.artifact_id, state.get(p.artifact_id))
        elif p.action == "invalidate":
            state[p.artifact_id] = None
        # "retain" -> no change to state
    return state


def _score_questions(scenario: Scenario, state: Dict[str, Optional[str]]) -> List[QuestionResult]:
    results: List[QuestionResult] = []
    for qa in scenario.questions:
        answers = {aid: state[aid] for aid in qa.relevant_artifact_ids if state.get(aid) is not None}

        quality = None
        if len(qa.relevant_artifact_ids) == 1:
            aid = qa.relevant_artifact_ids[0]
            prediction = state.get(aid) or ""
            quality = oracle_mod.score_quality(qa.question, qa.gold_answer, prediction)

        consistency = None
        if len(qa.relevant_artifact_ids) > 1:
            consistency = oracle_mod.consistency_check(answers)

        results.append(QuestionResult(question=qa.question, quality=quality, consistency=consistency))
    return results


def _cost_of_plan(scenario: Scenario, plan: List[MaintenancePlan], cost_table: CostTable) -> float:
    total = 0.0
    for p in plan:
        artifact_type = scenario.artifact_types.get(p.artifact_id, "CHUNK")
        total += cost_table.cost(artifact_type, p.action)
    return total


def _summarize(strategy: str, scenario: Scenario, plan: List[MaintenancePlan], cost_table: CostTable) -> EvaluationResult:
    state = _apply_plan_to_corpus(scenario, plan)
    question_results = _score_questions(scenario, state)

    quality_results = [qr.quality for qr in question_results if qr.quality is not None]
    consistency_results = [qr.consistency for qr in question_results if qr.consistency is not None]

    avg_f1 = sum(q.f1 for q in quality_results) / len(quality_results) if quality_results else 1.0
    exact_match_rate = (
        sum(1 for q in quality_results if q.exact_match) / len(quality_results) if quality_results else 1.0
    )
    consistency_rate = (
        sum(1 for c in consistency_results if c.consistent) / len(consistency_results)
        if consistency_results
        else 1.0
    )

    return EvaluationResult(
        strategy=strategy,
        plan=plan,
        total_cost=_cost_of_plan(scenario, plan, cost_table),
        avg_f1=avg_f1,
        exact_match_rate=exact_match_rate,
        consistency_rate=consistency_rate,
        question_results=question_results,
    )


def _get_estimates(scenario: Scenario, estimator: str):
    """estimator in {'trace', 'similarity'}."""
    changed_id = scenario.change_event["source_artifact_id"]
    event_id = scenario.change_event["event_id"]

    if estimator == "trace":
        client = MockGraphClient(neighbors=scenario.duplicates, embeddings={})
        return trace_impact(change_event_id=event_id, changed_artifact_id=changed_id, client=client)

    if estimator == "similarity":
        candidate_ids = scenario.duplicates.get(changed_id, [])
        texts_for_vocab = [scenario.updated_content.get(changed_id, scenario.corpus[changed_id])]
        texts_for_vocab += [scenario.corpus[cid] for cid in candidate_ids]
        vocab = _build_vocab(texts_for_vocab)

        changed_embedding = _vectorize(
            scenario.updated_content.get(changed_id, scenario.corpus[changed_id]), vocab
        )
        candidates = [(cid, _vectorize(scenario.corpus[cid], vocab)) for cid in candidate_ids]
        return similarity_estimate_impact(
            change_event_id=event_id, changed_embedding=changed_embedding, candidates=candidates
        )

    raise ValueError(f"unknown estimator: {estimator!r}")


def run_strategy(scenario: Scenario, strategy: str, cost_table: Optional[CostTable] = None) -> EvaluationResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    if cost_table is None:
        cost_table = CostTable()

    changed_id = scenario.change_event["source_artifact_id"]

    if strategy == "no_maintenance":
        plan: List[MaintenancePlan] = []

    elif strategy == "update_everything":
        plan = [
            MaintenancePlan(artifact_id=aid, action="update", reason="full-reindex baseline")
            for aid in scenario.updated_content
            if aid != changed_id
        ]

    elif strategy == "trace_only":
        plan = threshold_based_plan(_get_estimates(scenario, "trace"))

    elif strategy == "similarity_only":
        plan = threshold_based_plan(_get_estimates(scenario, "similarity"))

    elif strategy == "combined":
        estimates = _get_estimates(scenario, "trace") + _get_estimates(scenario, "similarity")
        plan = threshold_based_plan(estimates)

    else:  # "optimized"
        estimates = _get_estimates(scenario, "trace") + _get_estimates(scenario, "similarity")
        plan = optimize_maintenance_plan(estimates, cost_table=cost_table).plan

    return _summarize(strategy, scenario, plan, cost_table)


def compare_strategies(scenario: Scenario, cost_table: Optional[CostTable] = None) -> Dict[str, EvaluationResult]:
    return {s: run_strategy(scenario, s, cost_table) for s in STRATEGIES}


# --- built-in demo scenario ---------------------------------------------

def build_expense_demo_scenario() -> Scenario:
    """
    The expense-report / $500 -> $750 scenario used throughout this
    project's planning. Self-contained -- does not read from corpus/
    on disk, so it stays stable regardless of edits to those files.
    """
    corpus = {
        "chunk_exp_01": (
            "Employees submit expense reports through the finance portal within "
            "30 days of incurring the expense. Each line item requires a receipt "
            "image and a cost-center code."
        ),
        "chunk_exp_02": "Manager approval is required for any single line item exceeding $500.",
        "chunk_exp_03": (
            "Reports are reconciled monthly, and reimbursements are issued via "
            "direct deposit within two pay cycles of final approval."
        ),
        "chunk_faq_17": (
            "Q: Do I need manager approval for an expense? "
            "A: Yes -- any single expense line item over $500 requires manager sign-off."
        ),
    }

    updated_content = {
        "chunk_exp_02": "Manager approval is required for any single line item exceeding $750.",
        "chunk_faq_17": (
            "Q: Do I need manager approval for an expense? "
            "A: Yes -- any single expense line item over $750 requires manager sign-off."
        ),
    }

    duplicates = {
        "chunk_exp_02": ["chunk_faq_17"],
    }

    change_event = {
        "event_id": "evt_a1b2c3",
        "source_artifact_id": "chunk_exp_02",
        "change_type": "UPDATE",
        "new_content_hash": "sha256:9f8a...c21",
        "detected_at": "2026-08-24T10:15:00Z",
    }

    questions = [
        QAItem(
            question="What is the expense approval threshold?",
            relevant_artifact_ids=["chunk_exp_02"],
            gold_answer="$750",
        ),
        QAItem(
            question="Do the official policy and the FAQ agree on the expense approval threshold?",
            relevant_artifact_ids=["chunk_exp_02", "chunk_faq_17"],
            gold_answer="",  # unused for consistency-only questions
        ),
    ]

    return Scenario(
        name="expense_report_threshold_change",
        corpus=corpus,
        duplicates=duplicates,
        change_event=change_event,
        updated_content=updated_content,
        questions=questions,
        artifact_types={aid: "CHUNK" for aid in corpus},
    )