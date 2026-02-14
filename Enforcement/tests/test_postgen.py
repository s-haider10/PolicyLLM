"""Tests for postgen modules — regex, SMT, judge."""
import os
import sys

import pytest

from Enforcement.bundle_loader import load_bundle
from Enforcement.postgen.regex import (
    compile_constraint_patterns,
    run_regex_check,
)
from Enforcement.postgen.smt import (
    extract_facts_from_response,
    run_smt_check,
    verify_facts_against_rules,
)
from Enforcement.postgen.judge import build_judge_prompt, run_judge_check
from Enforcement.pregen import build_context
from Enforcement.schemas import Constraint, RuleMetadata, VariableSchema

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")


@pytest.fixture
def bundle_and_index():
    return load_bundle(FIXTURE_PATH)


@pytest.fixture
def bundle(bundle_and_index):
    return bundle_and_index[0]


@pytest.fixture
def index(bundle_and_index):
    return bundle_and_index[1]


@pytest.fixture
def llm_client():
    """Real LLM client for classification tests using credentials from .env"""
    sys.path.insert(0, ".")
    from Extractor.src.llm.client import LLMClient

    provider = os.getenv("LLM_PROVIDER", "chatgpt")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    return LLMClient(
        provider=provider,
        model_id=model,
        temperature=0.0,
        max_tokens=512,
    )


@pytest.fixture
def refund_context(bundle, index):
    """Context without LLM - for tests that don't need classification."""
    return build_context("I want to return my laptop", bundle, index, session_id="test-pg")


@pytest.fixture
def refund_context_with_llm(bundle, index, llm_client):
    """Context with LLM classification - for tests that need populated rules/paths."""
    return build_context("I want to return my laptop", bundle, index, session_id="test-pg", llm_client=llm_client)


# ==========================================================================
# Regex tests
# ==========================================================================


class TestRegex:
    def test_clean_text_passes(self):
        result = run_regex_check(
            "Your refund request has been processed for a full refund.",
            [],
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.flags == []

    def test_ssn_detected(self):
        result = run_regex_check(
            "Your SSN is 123-45-6789.",
            [],
        )
        assert result.passed is False
        assert result.score == 0.0
        assert any("ssn" in f for f in result.flags)

    def test_email_detected(self):
        result = run_regex_check(
            "Send it to customer@example.com for processing.",
            [],
        )
        assert result.passed is False
        assert any("email" in f for f in result.flags)

    def test_credit_card_detected(self):
        result = run_regex_check(
            "Your card 4111-1111-1111-1111 was charged.",
            [],
        )
        assert result.passed is False
        assert any("credit_card" in f for f in result.flags)

    def test_guarantee_promise_detected(self):
        result = run_regex_check(
            "I guarantee you will get a full refund.",
            [],
        )
        assert result.passed is False
        assert any("guarantee" in f for f in result.flags)

    def test_password_disclosure_detected(self):
        result = run_regex_check(
            "Your password: MyS3cret!",
            [],
        )
        assert result.passed is False
        assert any("password" in f for f in result.flags)

    def test_constraint_pattern_compilation(self):
        c = Constraint(
            policy_id="test",
            constraint="NOT(promise_before_verified)",
            scope="always",
            metadata=RuleMetadata(),
        )
        patterns = compile_constraint_patterns([c])
        assert len(patterns) > 0

    def test_constraint_pattern_matches(self):
        c = Constraint(
            policy_id="test",
            constraint="NOT(promise_before_verified)",
            scope="always",
            metadata=RuleMetadata(),
        )
        result = run_regex_check(
            "We will promise before verified the details.",
            [c],
        )
        assert result.passed is False

    def test_pii_constraint_uses_defaults(self):
        """PII constraints should rely on default PII patterns, not add new ones."""
        c = Constraint(
            policy_id="test",
            constraint="NOT(disclose_pii)",
            scope="always",
            metadata=RuleMetadata(),
        )
        patterns = compile_constraint_patterns([c])
        # PII is handled by defaults, so constraint shouldn't add extra
        assert len(patterns) == 0


# ==========================================================================
# SMT tests
# ==========================================================================


class TestSMT:
    def test_extract_bool_facts(self, bundle):
        response = "The customer has receipt and the product is verified."
        facts = extract_facts_from_response(response, bundle.variables)
        assert "has_receipt" in facts
        assert facts["has_receipt"] is True

    def test_extract_int_facts(self, bundle):
        response = "The days since purchase is 10 days for this electronics item."
        facts = extract_facts_from_response(response, bundle.variables)
        assert "days_since_purchase" in facts
        assert facts["days_since_purchase"] == 10

    def test_extract_enum_facts(self, bundle):
        response = "This is an electronics product with a receipt."
        facts = extract_facts_from_response(response, bundle.variables)
        assert "product_category" in facts
        assert facts["product_category"] == "electronics"

    def test_extract_float_facts(self, bundle):
        response = "The refund amount is $149.99 for this item."
        facts = extract_facts_from_response(response, bundle.variables)
        assert "refund_amount" in facts
        assert facts["refund_amount"] == 149.99

    def test_no_facts_extracted(self, bundle):
        response = "Hello, how can I help you today?"
        facts = extract_facts_from_response(response, bundle.variables)
        # Might extract nothing or very little
        assert isinstance(facts, dict)

    def test_verify_clean_facts_pass(self, bundle, refund_context):
        """Facts consistent with a rule should pass."""
        facts = {
            "has_receipt": True,
            "product_category": "electronics",
            "days_since_purchase": 10,
        }
        result = verify_facts_against_rules(
            facts,
            refund_context.applicable_rules,
            refund_context.applicable_paths,
            refund_context.applicable_constraints,
            bundle.variables,
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_run_smt_check_clean_response(self, refund_context, bundle):
        response = (
            "Based on your receipt, this electronics product was purchased 10 days ago. "
            "You are eligible for a full refund per electronics_refund_v2."
        )
        result = run_smt_check(response, refund_context, bundle)
        assert result.passed is True

    def test_empty_facts_pass(self, refund_context, bundle):
        result = run_smt_check("Hello there!", refund_context, bundle)
        assert result.passed is True  # No facts → can't verify → pass


# ==========================================================================
# Judge tests
# ==========================================================================


class TestJudge:
    def test_build_judge_prompt_has_sections(self, refund_context_with_llm):
        prompt = build_judge_prompt("Your refund is approved.", refund_context_with_llm)
        assert "POLICY RULES IN SCOPE" in prompt
        assert "CONSTRAINTS" in prompt
        assert "USER QUERY" in prompt
        assert "AI RESPONSE TO EVALUATE" in prompt

    def test_build_judge_prompt_includes_rules(self, refund_context_with_llm):
        prompt = build_judge_prompt("test", refund_context_with_llm)
        # Should contain at least one policy ID
        assert any(
            r.policy_id in prompt
            for r in refund_context_with_llm.applicable_rules
        )

    def test_judge_with_stub_llm_returns_fallback(self, refund_context_with_llm):
        """When LLM fails, judge should return score=0.5."""

        class BrokenLLM:
            def invoke_json(self, prompt, schema=None):
                raise RuntimeError("LLM unavailable")

        result = run_judge_check("test response", refund_context_with_llm, BrokenLLM())
        assert result.score == 0.5
        assert "judge_llm_unavailable" in result.issues

    def test_judge_with_mock_llm(self, refund_context_with_llm):
        """Judge with a mock LLM that returns a valid score."""

        class MockLLM:
            def invoke_json(self, prompt, schema=None):
                return {"score": 0.9, "issues": [], "explanation": "Compliant response"}

        result = run_judge_check("Your full refund has been approved.", refund_context_with_llm, MockLLM())
        assert result.score == 0.9
        assert result.issues == []
        assert result.explanation == "Compliant response"

    def test_judge_clamps_score(self, refund_context_with_llm):
        """Scores outside 0-1 should be clamped."""

        class OverScoreLLM:
            def invoke_json(self, prompt, schema=None):
                return {"score": 1.5, "issues": [], "explanation": ""}

        result = run_judge_check("test", refund_context_with_llm, OverScoreLLM())
        assert result.score == 1.0

        class UnderScoreLLM:
            def invoke_json(self, prompt, schema=None):
                return {"score": -0.5, "issues": [], "explanation": ""}

        result = run_judge_check("test", refund_context_with_llm, UnderScoreLLM())
        assert result.score == 0.0


# ==========================================================================
# Path Verification tests
# ==========================================================================


class TestSMTPathVerification:
    """Test that SMT verification checks path traversal."""

    def test_response_on_valid_path(self, refund_context_with_llm):
        """Response that follows a valid DAG path should pass."""
        response = (
            "Customer has receipt and it is verified. "
            "The product category is electronics. "
            "Days since purchase is 10. "
            "Full refund approved."
        )
        bundle, _ = load_bundle(FIXTURE_PATH)

        result = run_smt_check(response, refund_context_with_llm, bundle)

        # Should have high score because facts match a valid path
        assert result.passed or result.score >= 0.8

    def test_response_outside_paths(self, refund_context_with_llm):
        """Response with facts not on any DAG path should score low."""
        response = (
            "Customer has receipt confirmed. "
            "The product category is furniture. "  # Not in any path!
            "Days since purchase is 10."
        )
        bundle, _ = load_bundle(FIXTURE_PATH)

        result = run_smt_check(response, refund_context_with_llm, bundle)

        # Should flag as uncovered case
        assert not result.passed or result.score < 0.7
        assert any("uncovered_case" in str(v) for v in result.violations)
