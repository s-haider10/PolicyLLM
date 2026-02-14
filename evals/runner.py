"""Core evaluation runner — runs scenarios through the Enforcement pipeline."""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from Enforcement import load_bundle, enforce, ComplianceDecision, ComplianceAction
from Enforcement.orchestrator import EnforcementConfig

from .scenarios.schema import EvalScenario, EvalSuite


@dataclass
class ScenarioResult:
    """Result of a single scenario evaluation."""
    scenario_id: str
    scenario_name: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    decision: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    determinism_consistent: Optional[bool] = None


@dataclass
class SuiteResult:
    """Aggregate result of running a full suite."""
    suite_name: str
    provider: str
    model: str
    results: List[ScenarioResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    duration_ms: float = 0.0


def _check_scenario(scenario: EvalScenario, decision: ComplianceDecision) -> List[str]:
    """Check a decision against scenario expectations. Returns list of failure messages."""
    failures = []
    actual_action = decision.action.value if isinstance(decision.action, ComplianceAction) else str(decision.action)

    # Action check
    if actual_action != scenario.expected_action:
        failures.append(f"action: expected={scenario.expected_action}, got={actual_action}")

    # Score range check
    if scenario.expected_score_min is not None and decision.score < scenario.expected_score_min:
        failures.append(f"score_min: expected>={scenario.expected_score_min}, got={decision.score:.4f}")
    if scenario.expected_score_max is not None and decision.score > scenario.expected_score_max:
        failures.append(f"score_max: expected<={scenario.expected_score_max}, got={decision.score:.4f}")

    # Violation substring check
    for expected_v in scenario.expected_violations:
        found = any(expected_v.lower() in v.lower() for v in decision.violations)
        if not found:
            failures.append(f"violation: expected substring '{expected_v}' not found in {decision.violations}")

    # Fine-grained regex check
    if scenario.expected_regex:
        evidence = decision.evidence or {}
        audit_scores = (decision.audit_trail or {}).get("scores", {})
        regex_score = audit_scores.get("regex", 1.0)
        regex_passed = regex_score >= 1.0
        if regex_passed != scenario.expected_regex.should_pass:
            failures.append(f"regex.passed: expected={scenario.expected_regex.should_pass}, got={regex_passed}")
        flagged = evidence.get("regex_flags", [])
        for should_flag in scenario.expected_regex.should_flag:
            if not any(should_flag.lower() in f.lower() for f in flagged):
                failures.append(f"regex: expected flag '{should_flag}' not found in {flagged}")

    # Fine-grained SMT check
    if scenario.expected_smt:
        evidence = decision.evidence or {}
        smt_violations = evidence.get("smt_violations", [])
        smt_passed = len(smt_violations) == 0
        if smt_passed != scenario.expected_smt.should_pass:
            failures.append(f"smt.passed: expected={scenario.expected_smt.should_pass}, got={smt_passed}")

    # Fine-grained judge check
    if scenario.expected_judge:
        audit_scores = (decision.audit_trail or {}).get("scores", {})
        judge_score = audit_scores.get("judge", 0.5)
        if judge_score < scenario.expected_judge.min_score:
            failures.append(f"judge.score: expected>={scenario.expected_judge.min_score}, got={judge_score:.4f}")
        if judge_score > scenario.expected_judge.max_score:
            failures.append(f"judge.score: expected<={scenario.expected_judge.max_score}, got={judge_score:.4f}")

    # Fine-grained coverage check
    if scenario.expected_coverage:
        audit_scores = (decision.audit_trail or {}).get("scores", {})
        coverage_score = audit_scores.get("coverage", 0.0)
        if coverage_score < scenario.expected_coverage.min_score:
            failures.append(f"coverage.score: expected>={scenario.expected_coverage.min_score}, got={coverage_score:.4f}")

    return failures


def run_scenario(
    scenario: EvalScenario,
    bundle_path: str,
    llm_client: Any,
    judge_llm_client: Optional[Any] = None,
    config: Optional[EnforcementConfig] = None,
) -> ScenarioResult:
    """Run a single scenario and return the result."""
    config = config or EnforcementConfig()
    bundle, index = load_bundle(bundle_path)

    generate_fn: Optional[Callable] = None
    if scenario.response is not None:
        generate_fn = lambda _prompt: scenario.response

    all_decisions: List[ComplianceDecision] = []
    start = time.perf_counter()

    for _ in range(max(scenario.determinism_runs, 1)):
        decision = enforce(
            query=scenario.query,
            bundle=bundle,
            bundle_index=index,
            llm_client=llm_client,
            judge_llm_client=judge_llm_client,
            config=config,
            generate_fn=generate_fn,
        )
        all_decisions.append(decision)

    duration_ms = (time.perf_counter() - start) * 1000
    primary = all_decisions[0]
    failures = _check_scenario(scenario, primary)

    # Determinism check
    determinism_consistent = None
    if scenario.determinism_runs > 1:
        actions = {d.action for d in all_decisions}
        scores = [d.score for d in all_decisions]
        determinism_consistent = len(actions) == 1 and (max(scores) - min(scores)) < 0.05
        if not determinism_consistent:
            failures.append(
                f"determinism: actions={[d.action.value for d in all_decisions]}, "
                f"scores={[round(d.score, 4) for d in all_decisions]}"
            )

    return ScenarioResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        passed=len(failures) == 0,
        failures=failures,
        decision=primary.model_dump(),
        duration_ms=duration_ms,
        determinism_consistent=determinism_consistent,
    )


def run_suite(
    suite: EvalSuite,
    llm_client: Any,
    judge_llm_client: Optional[Any] = None,
    config: Optional[EnforcementConfig] = None,
    provider: str = "unknown",
    model: str = "unknown",
) -> SuiteResult:
    """Run all scenarios in a suite and return aggregate results."""
    start = time.perf_counter()
    results = []
    for scenario in suite.scenarios:
        result = run_scenario(
            scenario=scenario,
            bundle_path=suite.bundle_path,
            llm_client=llm_client,
            judge_llm_client=judge_llm_client,
            config=config,
        )
        results.append(result)

    duration_ms = (time.perf_counter() - start) * 1000
    passed = sum(1 for r in results if r.passed)
    return SuiteResult(
        suite_name=suite.name,
        provider=provider,
        model=model,
        results=results,
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        duration_ms=duration_ms,
    )
