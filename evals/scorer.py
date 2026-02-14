"""Aggregate scoring and comparison utilities for eval results."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .runner import SuiteResult


@dataclass
class DimensionScores:
    """Per-dimension pass rates across a suite."""
    action_accuracy: float = 0.0
    score_in_range: float = 0.0
    violation_recall: float = 0.0
    regex_accuracy: float = 0.0
    smt_accuracy: float = 0.0
    judge_in_range: float = 0.0
    coverage_met: float = 0.0
    determinism_rate: Optional[float] = None


def compute_dimension_scores(result: SuiteResult) -> DimensionScores:
    """Compute per-dimension pass rates from a suite result."""
    if not result.results:
        return DimensionScores()

    n = len(result.results)
    action_ok = 0
    score_ok = 0
    violation_ok = 0
    regex_ok = 0
    smt_ok = 0
    judge_ok = 0
    coverage_ok = 0
    determinism_ok = 0
    determinism_count = 0

    for r in result.results:
        fails = set(r.failures)
        has_action_fail = any("action:" in f for f in fails)
        has_score_fail = any("score_min:" in f or "score_max:" in f for f in fails)
        has_violation_fail = any("violation:" in f for f in fails)
        has_regex_fail = any("regex" in f for f in fails)
        has_smt_fail = any("smt" in f for f in fails)
        has_judge_fail = any("judge" in f for f in fails)
        has_coverage_fail = any("coverage" in f for f in fails)
        has_determinism_fail = any("determinism:" in f for f in fails)

        if not has_action_fail:
            action_ok += 1
        if not has_score_fail:
            score_ok += 1
        if not has_violation_fail:
            violation_ok += 1
        if not has_regex_fail:
            regex_ok += 1
        if not has_smt_fail:
            smt_ok += 1
        if not has_judge_fail:
            judge_ok += 1
        if not has_coverage_fail:
            coverage_ok += 1
        if r.determinism_consistent is not None:
            determinism_count += 1
            if not has_determinism_fail:
                determinism_ok += 1

    return DimensionScores(
        action_accuracy=action_ok / n,
        score_in_range=score_ok / n,
        violation_recall=violation_ok / n,
        regex_accuracy=regex_ok / n,
        smt_accuracy=smt_ok / n,
        judge_in_range=judge_ok / n,
        coverage_met=coverage_ok / n,
        determinism_rate=determinism_ok / determinism_count if determinism_count > 0 else None,
    )


@dataclass
class ComparisonEntry:
    """Comparison of two provider/model runs on the same suite."""
    suite_name: str
    provider_a: str
    model_a: str
    pass_rate_a: float
    provider_b: str
    model_b: str
    pass_rate_b: float
    delta: float
    per_scenario: List[Dict[str, Any]] = field(default_factory=list)


def compare_runs(a: SuiteResult, b: SuiteResult) -> ComparisonEntry:
    """Compare two suite results scenario-by-scenario."""
    a_map = {r.scenario_id: r for r in a.results}
    b_map = {r.scenario_id: r for r in b.results}
    all_ids = sorted(set(a_map) | set(b_map))

    per_scenario = []
    for sid in all_ids:
        ra = a_map.get(sid)
        rb = b_map.get(sid)
        per_scenario.append({
            "scenario_id": sid,
            "a_passed": ra.passed if ra else None,
            "b_passed": rb.passed if rb else None,
            "a_action": ra.decision.get("action") if ra and ra.decision else None,
            "b_action": rb.decision.get("action") if rb and rb.decision else None,
            "a_score": ra.decision.get("score") if ra and ra.decision else None,
            "b_score": rb.decision.get("score") if rb and rb.decision else None,
        })

    rate_a = a.passed / a.total if a.total else 0
    rate_b = b.passed / b.total if b.total else 0

    return ComparisonEntry(
        suite_name=a.suite_name,
        provider_a=a.provider,
        model_a=a.model,
        pass_rate_a=rate_a,
        provider_b=b.provider,
        model_b=b.model,
        pass_rate_b=rate_b,
        delta=rate_b - rate_a,
        per_scenario=per_scenario,
    )
