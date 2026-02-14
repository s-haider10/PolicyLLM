"""Tests for scoring.py — compliance score computation and action routing."""
import os
import sys

import pytest

from Enforcement.bundle_loader import load_bundle
from Enforcement.pregen import build_context
from Enforcement.schemas import (
    ComplianceAction,
    CoverageResult,
    JudgeResult,
    PostGenReport,
    RegexResult,
    SMTResult,
)
from Enforcement.scoring import (
    THRESHOLD_AUTO_CORRECT,
    THRESHOLD_PASS,
    THRESHOLD_REGENERATE,
    W_COVERAGE,
    W_JUDGE,
    W_SMT,
    build_compliance_decision,
    compute_compliance_score,
    compute_coverage,
    determine_action,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")


@pytest.fixture
def llm_client():
    """Real LLM client for classification tests using credentials from .env"""
    sys.path.insert(0, ".")
    from Extractor.src.llm.client import LLMClient

    # Use environment variables for provider and model
    provider = os.getenv("LLM_PROVIDER", "chatgpt")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    return LLMClient(
        provider=provider,
        model_id=model,
        temperature=0.0,
        max_tokens=512,
    )


def _make_report(smt=1.0, judge=1.0, regex=1.0, coverage=1.0, regex_passed=True):
    return PostGenReport(
        regex_result=RegexResult(passed=regex_passed, flags=[] if regex_passed else ["flag"], score=regex),
        smt_result=SMTResult(passed=smt == 1.0, violations=[], score=smt),
        judge_result=JudgeResult(score=judge, issues=[], explanation=""),
        coverage_result=CoverageResult(score=coverage, nodes_required=[], nodes_covered=[]),
    )


# --- compute_compliance_score ---


class TestComputeComplianceScore:
    def test_perfect_score(self):
        report = _make_report(1.0, 1.0, 1.0, 1.0)
        score = compute_compliance_score(report)
        assert abs(score - 1.0) < 1e-6

    def test_zero_score(self):
        report = _make_report(0.0, 0.0, 0.0, 0.0)
        score = compute_compliance_score(report)
        assert abs(score) < 1e-6

    def test_weights_add_to_one(self):
        # Regex is now a hard-gate, not weighted
        assert abs(W_SMT + W_JUDGE + W_COVERAGE - 1.0) < 1e-6

    def test_smt_dominant_weight(self):
        """SMT has highest weight (0.60) — failing SMT alone should drop score significantly."""
        report = _make_report(smt=0.0, judge=1.0, regex=1.0, coverage=1.0)
        score = compute_compliance_score(report)
        assert score < 0.5  # 0.40 (0.30 judge + 0.10 coverage)

    def test_judge_weight(self):
        report = _make_report(smt=1.0, judge=0.0, regex=1.0, coverage=1.0)
        score = compute_compliance_score(report)
        # 0.60 (SMT) + 0.00 (judge) + 0.10 (coverage) = 0.70
        assert abs(score - 0.70) < 1e-6

    def test_partial_scores(self):
        report = _make_report(smt=1.0, judge=0.8, regex=1.0, coverage=0.5)
        score = compute_compliance_score(report)
        # Regex no longer contributes to weighted score
        expected = W_SMT * 1.0 + W_JUDGE * 0.8 + W_COVERAGE * 0.5
        assert abs(score - expected) < 1e-6


# --- determine_action ---


class TestDetermineAction:
    def test_pass_threshold(self):
        report = _make_report(regex_passed=True)
        assert determine_action(0.95, report) == ComplianceAction.PASS
        assert determine_action(1.0, report) == ComplianceAction.PASS

    def test_auto_correct_threshold(self):
        report = _make_report(regex_passed=True)
        assert determine_action(0.90, report) == ComplianceAction.AUTO_CORRECT
        assert determine_action(0.85, report) == ComplianceAction.AUTO_CORRECT

    def test_regenerate_threshold(self):
        report = _make_report(regex_passed=True)
        assert determine_action(0.75, report) == ComplianceAction.REGENERATE
        assert determine_action(0.70, report) == ComplianceAction.REGENERATE

    def test_escalate_threshold(self):
        report = _make_report(regex_passed=True)
        assert determine_action(0.69, report) == ComplianceAction.ESCALATE
        assert determine_action(0.0, report) == ComplianceAction.ESCALATE

    def test_regex_failure_overrides(self):
        """Regex failure always escalates, regardless of score (Safety Hard-Gate)."""
        report_fail = _make_report(regex_passed=False)
        assert determine_action(1.0, report_fail) == ComplianceAction.ESCALATE
        assert determine_action(0.99, report_fail) == ComplianceAction.ESCALATE

    def test_boundary_values(self):
        report = _make_report(regex_passed=True)
        # Exactly at threshold
        assert determine_action(THRESHOLD_PASS, report) == ComplianceAction.PASS
        assert determine_action(THRESHOLD_PASS - 0.001, report) == ComplianceAction.AUTO_CORRECT
        assert determine_action(THRESHOLD_AUTO_CORRECT, report) == ComplianceAction.AUTO_CORRECT
        assert determine_action(THRESHOLD_AUTO_CORRECT - 0.001, report) == ComplianceAction.REGENERATE
        assert determine_action(THRESHOLD_REGENERATE, report) == ComplianceAction.REGENERATE
        assert determine_action(THRESHOLD_REGENERATE - 0.001, report) == ComplianceAction.ESCALATE


# --- compute_coverage ---


class TestComputeCoverage:
    @pytest.fixture
    def refund_context(self, llm_client):
        bundle, index = load_bundle(FIXTURE_PATH)
        return build_context("I want to return my laptop", bundle, index, session_id="cov-test", llm_client=llm_client)

    def test_full_coverage(self, refund_context):
        response = (
            "The customer has receipt for this electronics product. "
            "days since purchase is 10. The refund amount is $200."
        )
        result = compute_coverage(refund_context, response)
        assert result.score > 0
        assert len(result.nodes_covered) > 0

    def test_empty_response_zero_coverage(self, refund_context):
        result = compute_coverage(refund_context, "")
        if result.nodes_required:
            assert result.score == 0.0
        else:
            assert result.score == 1.0  # No nodes required

    def test_partial_coverage(self, refund_context):
        response = "The customer has receipt."
        result = compute_coverage(refund_context, response)
        # Should cover has_receipt but not all nodes
        if result.nodes_required:
            assert 0 < result.score <= 1.0


# --- build_compliance_decision ---


class TestBuildComplianceDecision:
    @pytest.fixture
    def refund_context(self, llm_client):
        bundle, index = load_bundle(FIXTURE_PATH)
        return build_context("refund laptop", bundle, index, session_id="bcd-test", llm_client=llm_client)

    def test_pass_decision(self, refund_context):
        report = _make_report(1.0, 1.0, 1.0, 1.0)
        decision = build_compliance_decision(report, "Approved full refund.", refund_context)
        assert decision.action == ComplianceAction.PASS
        assert decision.score >= THRESHOLD_PASS
        assert decision.violations == []

    def test_escalate_decision(self, refund_context):
        report = _make_report(0.0, 0.0, 0.0, 0.0)
        decision = build_compliance_decision(report, "", refund_context)
        assert decision.action == ComplianceAction.ESCALATE
        assert decision.score < THRESHOLD_REGENERATE

    def test_regex_failure_escalates(self, refund_context):
        report = _make_report(1.0, 1.0, 0.0, 1.0, regex_passed=False)
        decision = build_compliance_decision(report, "SSN: 123-45-6789", refund_context)
        assert decision.action == ComplianceAction.ESCALATE

    def test_evidence_populated(self, refund_context):
        report = _make_report(1.0, 0.8, 1.0, 0.9)
        decision = build_compliance_decision(report, "test", refund_context)
        assert "smt_violations" in decision.evidence
        assert "scores" in decision.audit_trail

    def test_audit_trail_scores(self, refund_context):
        report = _make_report(1.0, 0.8, 1.0, 0.5)
        decision = build_compliance_decision(report, "test", refund_context)
        scores = decision.audit_trail["scores"]
        assert scores["smt"] == 1.0
        assert scores["judge"] == 0.8
        assert scores["regex"] == 1.0
        assert scores["coverage"] == 0.5

    def test_determinism(self, refund_context):
        """Same input produces identical decisions."""
        report = _make_report(1.0, 0.9, 1.0, 0.8)
        d1 = build_compliance_decision(report, "test", refund_context)
        d2 = build_compliance_decision(report, "test", refund_context)
        assert d1.score == d2.score
        assert d1.action == d2.action
        assert d1.violations == d2.violations
