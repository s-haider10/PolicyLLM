"""End-to-end tests: full pipeline from bundle load to ComplianceDecision."""
import json
import os

import pytest

from Enforcement.audit import AuditLogger
from Enforcement.bundle_loader import load_bundle
from Enforcement.orchestrator import EnforcementConfig, enforce
from Enforcement.schemas import ComplianceAction, ComplianceDecision

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")


class StubLLM:
    def invoke(self, prompt, **kwargs):
        return ""

    def invoke_json(self, prompt, schema=None):
        return ""


class MockJudgeLLM:
    def __init__(self, score=0.9):
        self._score = score

    def invoke_json(self, prompt, schema=None):
        return {"score": self._score, "issues": [], "explanation": "Auto-scored"}


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
def all_checks_config():
    return EnforcementConfig(
        judge_enabled=True,
        smt_enabled=True,
        regex_enabled=True,
    )


@pytest.fixture
def no_judge_config():
    return EnforcementConfig(
        judge_enabled=False,
        smt_enabled=True,
        regex_enabled=True,
    )


# ==========================================================================
# Scenario 1: Clean refund response — should PASS
# ==========================================================================


def test_e2e_clean_refund_pass(bundle, index, all_checks_config):
    response = (
        "The customer has receipt for this electronics product. "
        "Days since purchase: 10. Per electronics_refund_v2 (source: refund_policy_2024.pdf, "
        "eff_date: 2024-01-01), you are eligible for a full refund."
    )
    decision = enforce(
        query="I want to return my laptop, I have the receipt",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        judge_llm_client=MockJudgeLLM(0.95),
        config=all_checks_config,
        generate_fn=lambda _: response,
        session_id="e2e-001",
    )
    assert isinstance(decision, ComplianceDecision)
    assert decision.action in (ComplianceAction.PASS, ComplianceAction.AUTO_CORRECT)
    assert decision.llm_response == response
    assert "duration_ms" in decision.audit_trail


# ==========================================================================
# Scenario 2: PII leak — should ESCALATE
# ==========================================================================


def test_e2e_pii_escalation(bundle, index, no_judge_config):
    response = (
        "Based on your receipt, your SSN 123-45-6789 shows you purchased electronics. "
        "Full refund approved."
    )
    decision = enforce(
        query="refund for laptop",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-002",
    )
    assert decision.action == ComplianceAction.ESCALATE
    assert any("ssn" in v.lower() for v in decision.violations)


# ==========================================================================
# Scenario 3: Guarantee/promise violation
# ==========================================================================


def test_e2e_guarantee_violation(bundle, index, no_judge_config):
    response = "I guarantee you that your full refund will be processed for the electronics."
    decision = enforce(
        query="I want to return my laptop",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-003",
    )
    assert decision.action == ComplianceAction.ESCALATE
    assert any("guarantee" in v.lower() for v in decision.violations)


# ==========================================================================
# Scenario 4: Unknown domain query — no policies — PASS
# ==========================================================================


def test_e2e_unknown_domain_pass(bundle, index, all_checks_config):
    decision = enforce(
        query="What is the weather forecast for tomorrow?",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        judge_llm_client=MockJudgeLLM(1.0),
        config=all_checks_config,
        session_id="e2e-004",
    )
    assert decision.action == ComplianceAction.PASS
    assert decision.score == 1.0


# ==========================================================================
# Scenario 5: Privacy query
# ==========================================================================


def test_e2e_privacy_query(bundle, index, no_judge_config):
    response = (
        "We take personal data protection seriously under GDPR. "
        "No PII will be disclosed per privacy_pii_v1."
    )
    decision = enforce(
        query="What personal data do you store under GDPR?",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-005",
    )
    assert isinstance(decision, ComplianceDecision)
    assert decision.score > 0


# ==========================================================================
# Scenario 6: Email in response — PII detection
# ==========================================================================


def test_e2e_email_in_response(bundle, index, no_judge_config):
    response = "Please contact user@example.com for your electronics refund."
    decision = enforce(
        query="How do I get a refund?",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-006",
    )
    assert decision.action == ComplianceAction.ESCALATE
    assert any("email" in v.lower() for v in decision.violations)


# ==========================================================================
# Scenario 7: Credit card in response
# ==========================================================================


def test_e2e_credit_card_in_response(bundle, index, no_judge_config):
    response = "Your card 4111-1111-1111-1111 was charged $200 for electronics."
    decision = enforce(
        query="refund request",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-007",
    )
    assert decision.action == ComplianceAction.ESCALATE


# ==========================================================================
# Scenario 8: Empty response
# ==========================================================================


def test_e2e_empty_response(bundle, index, no_judge_config):
    decision = enforce(
        query="I want to return my laptop",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: "",
        session_id="e2e-008",
    )
    assert isinstance(decision, ComplianceDecision)
    # Empty response → SMT passes (no facts), regex passes, coverage 0
    assert decision.score > 0


# ==========================================================================
# Scenario 9: Determinism — same query twice produces identical scores
# ==========================================================================


def test_e2e_determinism(bundle, index, no_judge_config):
    response = (
        "The customer has receipt for electronics. 10 days since purchase. "
        "Full refund per electronics_refund_v2."
    )
    d1 = enforce(
        query="I want to return my laptop",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-det-1",
    )
    d2 = enforce(
        query="I want to return my laptop",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-det-2",
    )
    assert d1.score == d2.score
    assert d1.action == d2.action
    assert d1.violations == d2.violations


# ==========================================================================
# Scenario 10: Audit log integrity after multiple runs
# ==========================================================================


def test_e2e_audit_integrity(bundle, index, no_judge_config, tmp_path):
    log_path = str(tmp_path / "e2e_audit.jsonl")
    logger = AuditLogger(log_path)

    queries = [
        ("I want to return my laptop", "Full refund approved per receipt."),
        ("Refund for clothing purchase", "Clothing refund within 30 days approved."),
        ("What is the weather?", None),
    ]
    for query, response in queries:
        gen_fn = (lambda r: lambda _: r)(response) if response else None
        enforce(
            query=query,
            bundle=bundle,
            bundle_index=index,
            llm_client=StubLLM(),
            config=no_judge_config,
            generate_fn=gen_fn,
            audit_logger=logger,
        )

    assert logger.verify_integrity()

    # Verify log file has expected entries
    with open(log_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert len(entries) == 3
    # Each entry should have hash chain fields
    for entry in entries:
        assert "entry_hash" in entry
        assert "session_id" in entry
        assert "compliance_score" in entry


# ==========================================================================
# Scenario 11: Late electronics return — store credit path
# ==========================================================================


def test_e2e_late_return_store_credit(bundle, index, no_judge_config):
    response = (
        "The customer has receipt for electronics. "
        "Days since purchase: 20. Per electronics_refund_late_v2, "
        "you are eligible for store credit only."
    )
    decision = enforce(
        query="I want to return my laptop bought 20 days ago with receipt",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-009",
    )
    assert isinstance(decision, ComplianceDecision)
    assert decision.score > 0


# ==========================================================================
# Scenario 12: model_dump() produces valid JSON
# ==========================================================================


def test_e2e_decision_serializable(bundle, index, no_judge_config):
    response = "Full refund approved for electronics with receipt."
    decision = enforce(
        query="refund request",
        bundle=bundle,
        bundle_index=index,
        llm_client=StubLLM(),
        config=no_judge_config,
        generate_fn=lambda _: response,
        session_id="e2e-010",
    )
    # model_dump should produce a JSON-serializable dict
    dumped = decision.model_dump()
    serialized = json.dumps(dumped, default=str)
    assert serialized
    reloaded = json.loads(serialized)
    assert isinstance(reloaded["score"], (int, float))
    assert "action" in reloaded
    assert "violations" in reloaded
