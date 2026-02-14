"""Report generation for eval results — terminal summary and JSON output."""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .runner import SuiteResult
from .scorer import compute_dimension_scores


def to_json_report(result: SuiteResult) -> dict:
    """Convert a SuiteResult to a JSON-serializable dict."""
    scores = compute_dimension_scores(result)
    return {
        "suite_name": result.suite_name,
        "provider": result.provider,
        "model": result.model,
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "pass_rate": round(result.passed / result.total, 4) if result.total else 0,
        "duration_ms": round(result.duration_ms, 1),
        "dimension_scores": asdict(scores),
        "scenarios": [
            {
                "id": r.scenario_id,
                "name": r.scenario_name,
                "passed": r.passed,
                "failures": r.failures,
                "duration_ms": round(r.duration_ms, 1),
                "determinism_consistent": r.determinism_consistent,
                "decision": r.decision,
            }
            for r in result.results
        ],
    }


def write_json_report(result: SuiteResult, path: str) -> None:
    """Write a JSON report to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report = to_json_report(result)
    Path(path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def print_summary(result: SuiteResult) -> None:
    """Print a human-readable summary to the terminal."""
    sep = "=" * 60
    print(sep)
    print(f"Eval Suite: {result.suite_name}")
    print(f"Provider: {result.provider}  Model: {result.model}")
    print(sep)

    pass_rate = (result.passed / result.total * 100) if result.total else 0
    print(f"Total: {result.total}  Passed: {result.passed}  Failed: {result.failed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"Duration: {result.duration_ms:.0f}ms")

    scores = compute_dimension_scores(result)
    print(f"\nDimension Scores:")
    print(f"  Action Accuracy:  {scores.action_accuracy:.1%}")
    print(f"  Score In Range:   {scores.score_in_range:.1%}")
    print(f"  Violation Recall: {scores.violation_recall:.1%}")
    print(f"  Regex Accuracy:   {scores.regex_accuracy:.1%}")
    print(f"  SMT Accuracy:     {scores.smt_accuracy:.1%}")
    print(f"  Judge In Range:   {scores.judge_in_range:.1%}")
    print(f"  Coverage Met:     {scores.coverage_met:.1%}")
    if scores.determinism_rate is not None:
        print(f"  Determinism:      {scores.determinism_rate:.1%}")

    print("-" * 60)
    for r in result.results:
        status = "PASS" if r.passed else "FAIL"
        line = f"  [{status}] {r.scenario_id}: {r.scenario_name} ({r.duration_ms:.0f}ms)"
        print(line)
        if not r.passed:
            for fail in r.failures:
                print(f"         -> {fail}")
    print(sep)
