"""Compliance score computation and action routing."""
import re
from typing import List

from .schemas import (
    ComplianceAction,
    ComplianceDecision,
    CoverageResult,
    EnforcementContext,
    PostGenReport,
    VariableSchema,
)

# Weights
W_SMT = 0.60
W_JUDGE = 0.30
W_COVERAGE = 0.10
# Regex is now a Hard-Gate (Weight 0, but can trigger ESCALATE)

# Thresholds
THRESHOLD_PASS = 0.95
THRESHOLD_AUTO_CORRECT = 0.85
THRESHOLD_REGENERATE = 0.70


def compute_compliance_score(report: PostGenReport) -> float:
    """Compute the weighted compliance score.

    S = W_SMT * Z + W_JUDGE * L + W_COVERAGE * C

    Note: Regex is no longer weighted in the score but acts as a hard-gate.
    """
    return (
        W_SMT * report.smt_result.score
        + W_JUDGE * report.judge_result.score
        + W_COVERAGE * report.coverage_result.score
    )


def determine_action(score: float, report: PostGenReport) -> ComplianceAction:
    """Determine enforcement action from score with Safety-First hard-gate.

    Regex failure triggers immediate escalation regardless of other metrics,
    ensuring safety constraints cannot be overridden by high compliance scores.
    """
    # Safety hard-gate: regex failure always escalates
    if not report.regex_result.passed:
        return ComplianceAction.ESCALATE

    # Score-based thresholds for all other cases
    if score >= THRESHOLD_PASS:
        return ComplianceAction.PASS
    if score >= THRESHOLD_AUTO_CORRECT:
        return ComplianceAction.AUTO_CORRECT
    if score >= THRESHOLD_REGENERATE:
        return ComplianceAction.REGENERATE
    return ComplianceAction.ESCALATE


def compute_coverage(
    context: EnforcementContext,
    response_text: str,
) -> CoverageResult:
    """Measure what fraction of required decision nodes are addressed in the response."""
    nodes_required: List[str] = []
    for path in context.applicable_paths:
        for step in path.path:
            if step.var not in nodes_required:
                nodes_required.append(step.var)

    if not nodes_required:
        return CoverageResult(score=1.0, nodes_required=[], nodes_covered=[])

    lower = response_text.lower()
    nodes_covered: List[str] = []
    for node in nodes_required:
        readable = node.replace("_", " ")
        if readable in lower or node in lower:
            nodes_covered.append(node)

    # Base coverage: fraction of nodes mentioned
    base_score = len(nodes_covered) / len(nodes_required) if nodes_required else 1.0

    # Penalty if not all required nodes covered (incomplete path)
    if len(nodes_covered) < len(nodes_required):
        score = base_score * 0.8  # 20% penalty for incomplete coverage
    else:
        score = base_score

    return CoverageResult(
        score=score,
        nodes_required=nodes_required,
        nodes_covered=nodes_covered
    )


def build_compliance_decision(
    report: PostGenReport,
    llm_response: str,
    context: EnforcementContext,
) -> ComplianceDecision:
    """Full scoring pipeline: compute score, determine action, build decision."""
    score = compute_compliance_score(report)
    action = determine_action(score, report)

    # Collect violations
    violations: List[str] = []
    violations.extend(report.regex_result.flags)
    for v in report.smt_result.violations:
        violations.append(f"SMT: {v.get('policy_id', '?')} — {v.get('constraint', v.get('violation_type', '?'))}")
    violations.extend(f"Judge: {issue}" for issue in report.judge_result.issues)

    evidence = {
        "smt_violations": report.smt_result.violations,
        "regex_flags": report.regex_result.flags,
        "judge_issues": report.judge_result.issues,
        "judge_explanation": report.judge_result.explanation,
        "coverage": {
            "required": report.coverage_result.nodes_required,
            "covered": report.coverage_result.nodes_covered,
        },
    }

    audit_trail = {
        "scores": {
            "smt": report.smt_result.score,
            "judge": report.judge_result.score,
            "regex": report.regex_result.score,
            "coverage": report.coverage_result.score,
            "final": score,
        },
        "weights": {"smt": W_SMT, "judge": W_JUDGE, "coverage": W_COVERAGE, "regex_hard_gate": True},
    }

    return ComplianceDecision(
        score=score,
        action=action,
        violations=violations,
        evidence=evidence,
        audit_trail=audit_trail,
        llm_response=llm_response,
    )
