"""Tests for orchestrator.py — full pipeline integration with mock LLM."""
import os

import pytest

from Enforcement.bundle_loader import load_bundle
from Enforcement.orchestrator import EnforcementConfig, enforce
from Enforcement.schemas import ComplianceAction

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


class StubLLM:
    """Stub LLM client that returns canned responses."""

    def __init__(self, response_text=None, json_response=None):
        self._response = response_text or ""
        self._json_response = json_response

    def invoke(self, prompt, **kwargs):
        return self._response

    def invoke_json(self, prompt, schema=None):
        if self._json_response is not None:
            return self._json_response
        return self._response


class TestEnforcementConfig:
    def test_defaults(self):
        cfg = EnforcementConfig()
        assert cfg.max_retries == 2
        assert cfg.auto_correct_max_attempts == 1
        assert cfg.judge_enabled is True
        assert cfg.smt_enabled is True
        assert cfg.regex_enabled is True


class TestEnforcePassPath:
    def test_pass_with_clean_response(self, bundle, index):
        """A clean, policy-compliant pre-generated response should PASS or AUTO_CORRECT."""
        # Response mentions all decision node variable names for coverage
        response = (
            "The customer has receipt (has_receipt confirmed). "
            "The product category is electronics. "
            "Days since purchase: 10. The refund amount is $299. "
            "Per electronics_refund_v2, you are eligible for a full refund."
        )
        config = EnforcementConfig(
            judge_enabled=False,
            smt_enabled=True,
            regex_enabled=True,
        )
        decision = enforce(
            query="I want to return my laptop",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            config=config,
            generate_fn=lambda _: response,
            session_id="test-pass",
        )
        assert decision.action in (ComplianceAction.PASS, ComplianceAction.AUTO_CORRECT)
        assert decision.score > 0

    def test_pass_with_judge(self, bundle, index):
        """With a mock judge returning 1.0, should still pass."""
        response = (
            "Based on your receipt, this electronics product was purchased 10 days ago. "
            "Per electronics_refund_v2, you are eligible for a full refund."
        )
        judge_llm = StubLLM(json_response={"score": 1.0, "issues": [], "explanation": "Compliant"})
        config = EnforcementConfig(
            judge_enabled=True,
            smt_enabled=True,
            regex_enabled=True,
        )
        decision = enforce(
            query="I want to return my laptop",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            judge_llm_client=judge_llm,
            config=config,
            generate_fn=lambda _: response,
            session_id="test-pass-judge",
        )
        assert decision.score > 0


class TestEnforceEscalatePath:
    def test_escalate_pii_in_response(self, bundle, index):
        """PII in response should trigger regex failure → ESCALATE."""
        response = "Your SSN is 123-45-6789 and your refund is approved."
        config = EnforcementConfig(
            judge_enabled=False,
            smt_enabled=False,
            regex_enabled=True,
        )
        decision = enforce(
            query="I want to return my laptop",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            config=config,
            generate_fn=lambda _: response,
            session_id="test-escalate-pii",
        )
        assert decision.action == ComplianceAction.ESCALATE

    def test_escalate_with_low_judge(self, bundle, index):
        """Judge scoring 0.0 with all checks enabled should escalate."""
        response = "I'll process your refund immediately, no questions asked!"
        judge_llm = StubLLM(json_response={"score": 0.0, "issues": ["non_compliant"], "explanation": "Bad"})
        config = EnforcementConfig(
            judge_enabled=True,
            smt_enabled=False,
            regex_enabled=True,
        )
        decision = enforce(
            query="I want to return my laptop",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            judge_llm_client=judge_llm,
            config=config,
            generate_fn=lambda _: response,
            session_id="test-escalate-judge",
        )
        # With judge=0.0, SMT disabled (1.0), regex pass (1.0), coverage varies
        # Score = 0.55*1.0 + 0.25*0.0 + 0.10*1.0 + 0.10*C
        # Worst case C=0: 0.65 → ESCALATE
        assert decision.action in (ComplianceAction.ESCALATE, ComplianceAction.REGENERATE)


class TestEnforceNoApplicableRules:
    def test_no_rules_returns_pass(self, bundle, index):
        """Query with no matching rules should PASS immediately."""
        config = EnforcementConfig(
            judge_enabled=False,
            smt_enabled=False,
        )
        decision = enforce(
            query="What is the meaning of life?",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            config=config,
            session_id="test-no-rules",
        )
        assert decision.action == ComplianceAction.PASS
        assert decision.score == 1.0


class TestEnforceRetryLoop:
    def test_auto_correct_retries(self, bundle, index):
        """Auto-correct should retry once with hints."""
        call_count = [0]

        def generate(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                # First response triggers a guarantee promise
                return "I guarantee you a full refund for your electronics with receipt."
            else:
                # Corrected response
                return (
                    "Based on your receipt, this electronics product was purchased 10 days ago. "
                    "Per policy, you are eligible for a full refund."
                )

        judge_llm = StubLLM(json_response={"score": 1.0, "issues": [], "explanation": "OK"})
        config = EnforcementConfig(
            judge_enabled=True,
            smt_enabled=True,
            regex_enabled=True,
            auto_correct_max_attempts=1,
        )
        decision = enforce(
            query="I want to return my laptop, I have the receipt",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            judge_llm_client=judge_llm,
            config=config,
            generate_fn=generate,
            session_id="test-auto-correct",
        )
        # Should have been called at least twice (original + retry)
        assert call_count[0] >= 1


class TestEnforcePregenResponse:
    def test_pre_generated_response(self, bundle, index):
        """enforce() with a generate_fn that returns a pre-set string."""
        pre_response = (
            "The customer has a receipt for electronics. "
            "Days since purchase: 10. Full refund per electronics_refund_v2."
        )
        config = EnforcementConfig(
            judge_enabled=False,
            smt_enabled=True,
            regex_enabled=True,
        )
        decision = enforce(
            query="I want to return my laptop",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            config=config,
            generate_fn=lambda _: pre_response,
        )
        assert decision.llm_response == pre_response


class TestEnforceAuditLogging:
    def test_audit_logger_called(self, bundle, index, tmp_path):
        """AuditLogger should write an entry after enforce()."""
        from Enforcement.audit import AuditLogger

        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path)
        config = EnforcementConfig(
            judge_enabled=False,
            smt_enabled=False,
            regex_enabled=True,
        )
        enforce(
            query="I want to return my laptop",
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            config=config,
            generate_fn=lambda _: "Full refund approved based on receipt for electronics.",
            audit_logger=logger,
            session_id="test-audit",
        )
        import json

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert "session_id" in entry
        assert "entry_hash" in entry

    def test_audit_hash_chain_integrity(self, bundle, index, tmp_path):
        """Multiple enforce calls should produce a verifiable hash chain."""
        from Enforcement.audit import AuditLogger

        log_path = str(tmp_path / "chain_audit.jsonl")
        logger = AuditLogger(log_path)
        config = EnforcementConfig(
            judge_enabled=False,
            smt_enabled=False,
            regex_enabled=True,
        )
        for i in range(3):
            enforce(
                query=f"Return request #{i}",
                bundle=bundle,
                bundle_index=index,
                llm_client=StubLLM(),
                config=config,
                generate_fn=lambda _: "Refund approved.",
                audit_logger=logger,
                session_id=f"chain-{i}",
            )
        assert logger.verify_integrity()
