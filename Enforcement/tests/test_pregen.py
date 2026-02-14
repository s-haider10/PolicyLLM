"""Tests for pregen.py — query classification, rule retrieval, dominance resolution."""
import os
import sys

import pytest

from Enforcement.bundle_loader import load_bundle
from Enforcement.pregen import (
    apply_dominance,
    build_context,
    classify_query,
    retrieve_rules,
)

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

    # Use environment variables for provider and model
    provider = os.getenv("LLM_PROVIDER", "chatgpt")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    return LLMClient(
        provider=provider,
        model_id=model,
        temperature=0.0,
        max_tokens=512,
    )


# --- Query classification ---


class TestClassifyQuery:
    def test_refund_query(self, bundle, llm_client):
        domain, intent, conf = classify_query("I want to return my laptop", bundle, llm_client)
        assert domain == "refund"

    def test_refund_keywords(self, bundle, llm_client):
        domain, intent, conf = classify_query("I want a refund for my receipt item", bundle, llm_client)
        assert domain == "refund"
        assert conf > 0

    def test_privacy_query(self, bundle, llm_client):
        domain, intent, conf = classify_query("What personal data do you store under GDPR?", bundle, llm_client)
        assert domain == "privacy"

    def test_unknown_query(self, bundle, llm_client):
        domain, intent, conf = classify_query("How is the weather today?", bundle, llm_client)
        # Should either be "unknown" or a low-confidence match
        assert conf < 0.6 or domain == "unknown"

    def test_intent_classification(self, bundle, llm_client):
        domain, intent, conf = classify_query("What is the policy on refund returns?", bundle, llm_client)
        assert intent in ("refund_request", "policy_inquiry", "unknown")

    def test_llm_only_classification(self, bundle, llm_client):
        """LLM-only classification handles semantic understanding."""
        domain, intent, conf = classify_query("I want a refund for my exchange receipt", bundle, llm_client)
        assert domain == "refund"
        assert conf > 0


# --- Rule retrieval ---


class TestRetrieveRules:
    def test_retrieve_refund_rules(self, index):
        rules, paths, constraints = retrieve_rules("refund", index)
        assert len(rules) == 3
        pids = {r.policy_id for r in rules}
        assert "electronics_refund_v2" in pids
        assert "clothing_refund_v1" in pids

    def test_retrieve_privacy_rules(self, index):
        rules, paths, constraints = retrieve_rules("privacy", index)
        assert len(rules) == 1
        assert rules[0].policy_id == "privacy_pii_v1"

    def test_retrieve_unknown_domain(self, index):
        rules, paths, constraints = retrieve_rules("nonexistent", index)
        assert rules == []

    def test_constraints_include_always_scope(self, index):
        rules, paths, constraints = retrieve_rules("refund", index)
        scopes = {c.scope for c in constraints}
        assert "always" in scopes

    def test_constraints_include_domain_scope(self, index):
        rules, paths, constraints = retrieve_rules("refund", index)
        scopes = {c.scope for c in constraints}
        assert "refund" in scopes

    def test_paths_match_retrieved_rules(self, index):
        rules, paths, constraints = retrieve_rules("refund", index)
        rule_pids = {r.policy_id for r in rules}
        path_pids = {p.policy_id for p in paths}
        assert path_pids.issubset(rule_pids)

    def test_temporal_filtering(self, index):
        """Rules with eff_date in the future should be filtered out."""
        rules, paths, constraints = retrieve_rules("refund", index, effective_date="2023-01-01")
        # electronics_refund_v2 eff_date is 2024-01-01 — should be filtered
        pids = {r.policy_id for r in rules}
        assert "electronics_refund_v2" not in pids

    def test_temporal_filtering_includes_active(self, index):
        rules, paths, constraints = retrieve_rules("refund", index, effective_date="2025-01-01")
        pids = {r.policy_id for r in rules}
        assert "electronics_refund_v2" in pids


# --- Dominance resolution ---


class TestApplyDominance:
    def test_dominance_override(self, bundle, index):
        """When both electronics and clothing rules fire, electronics wins."""
        rules, paths, constraints = retrieve_rules("refund", index, effective_date="2025-01-01")
        filtered_rules, filtered_paths, applied = apply_dominance(
            rules, paths, index, bundle.priority_lattice,
        )
        # clothing_refund_v1 should be removed by dominance
        pids = {r.policy_id for r in filtered_rules}
        assert "electronics_refund_v2" in pids
        assert "clothing_refund_v1" not in pids
        assert len(applied) > 0

    def test_no_dominance_for_non_conflicting(self, bundle, index):
        """Privacy rule shouldn't be affected by refund dominance."""
        rules, paths, constraints = retrieve_rules("privacy", index)
        filtered_rules, filtered_paths, applied = apply_dominance(
            rules, paths, index, bundle.priority_lattice,
        )
        assert len(filtered_rules) == 1
        assert filtered_rules[0].policy_id == "privacy_pii_v1"


# --- build_context ---


class TestBuildContext:
    def test_build_context_refund(self, bundle, index, llm_client):
        ctx = build_context("I want to return my laptop", bundle, index, session_id="test-001", llm_client=llm_client)
        assert ctx.session_id == "test-001"
        assert ctx.domain == "refund"
        assert len(ctx.applicable_rules) > 0
        assert ctx.timestamp

    def test_build_context_unknown(self, bundle, index, llm_client):
        ctx = build_context("weather forecast", bundle, index, session_id="test-002", llm_client=llm_client)
        # Should still produce a valid context
        assert ctx.session_id == "test-002"

    def test_build_context_escalation_contacts(self, bundle, index, llm_client):
        ctx = build_context("I want a refund for electronics", bundle, index, llm_client=llm_client)
        # Escalation entry references electronics_refund_v2 / electronics_refund_late_v2
        # At least one should be in applicable_rules
        rule_pids = {r.policy_id for r in ctx.applicable_rules}
        if "electronics_refund_v2" in rule_pids or "electronics_refund_late_v2" in rule_pids:
            assert len(ctx.escalation_contacts) > 0
