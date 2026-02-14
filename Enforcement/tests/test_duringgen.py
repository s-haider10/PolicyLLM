"""Tests for duringgen.py — scaffold serialization and prompt injection."""
import os

import pytest

from Enforcement.bundle_loader import load_bundle
from Enforcement.duringgen import (
    build_injection_bundle,
    format_full_prompt,
    serialize_constraints,
    serialize_scaffold,
)
from Enforcement.pregen import build_context
from Enforcement.schemas import Constraint, RuleMetadata

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
def refund_context(bundle, index):
    return build_context("I want to return my laptop", bundle, index, session_id="test-dg")


# --- serialize_constraints ---


class TestSerializeConstraints:
    def test_empty_constraints(self):
        result = serialize_constraints([])
        assert result == ""

    def test_not_constraint_becomes_never(self):
        c = Constraint(
            policy_id="test",
            constraint="NOT(disclose_pii)",
            scope="always",
            metadata=RuleMetadata(domain="privacy"),
        )
        result = serialize_constraints([c])
        assert "NEVER" in result
        assert "disclose pii" in result

    def test_plain_constraint_becomes_always(self):
        c = Constraint(
            policy_id="test",
            constraint="verify_identity_first",
            scope="always",
            metadata=RuleMetadata(domain="security"),
        )
        result = serialize_constraints([c])
        assert "ALWAYS comply with" in result

    def test_numbered_output(self):
        constraints = [
            Constraint(policy_id="a", constraint="NOT(x)", scope="always", metadata=RuleMetadata()),
            Constraint(policy_id="b", constraint="NOT(y)", scope="always", metadata=RuleMetadata()),
        ]
        result = serialize_constraints(constraints)
        assert "1)" in result
        assert "2)" in result


# --- serialize_scaffold ---


class TestSerializeScaffold:
    def test_empty_paths(self):
        result = serialize_scaffold([], {}, [], [])
        assert result == []

    def test_scaffold_has_steps(self, bundle, refund_context):
        steps = serialize_scaffold(
            refund_context.applicable_paths,
            bundle.variables,
            bundle.decision_nodes,
            refund_context.dominance_applied,
        )
        assert len(steps) > 0
        assert any("STEP" in s for s in steps)

    def test_scaffold_final_step(self, bundle, refund_context):
        steps = serialize_scaffold(
            refund_context.applicable_paths,
            bundle.variables,
            bundle.decision_nodes,
            refund_context.dominance_applied,
        )
        assert "FINAL" in steps[-1]

    def test_bool_var_instruction(self, bundle, refund_context):
        steps = serialize_scaffold(
            refund_context.applicable_paths,
            bundle.variables,
            bundle.decision_nodes,
            refund_context.dominance_applied,
        )
        # has_receipt is a bool var
        has_receipt_step = [s for s in steps if "has_receipt" in s and "STEP" in s]
        assert len(has_receipt_step) > 0
        assert "DO NOT assume" in has_receipt_step[0]

    def test_enum_var_lists_values(self, bundle, refund_context):
        steps = serialize_scaffold(
            refund_context.applicable_paths,
            bundle.variables,
            bundle.decision_nodes,
            refund_context.dominance_applied,
        )
        product_step = [s for s in steps if "product_category" in s and "Must be one of" in s]
        assert len(product_step) > 0
        assert "electronics" in product_step[0]

    def test_determinism(self, bundle, refund_context):
        """Same input should produce same scaffold — run twice and compare."""
        steps1 = serialize_scaffold(
            refund_context.applicable_paths,
            bundle.variables,
            bundle.decision_nodes,
            refund_context.dominance_applied,
        )
        steps2 = serialize_scaffold(
            refund_context.applicable_paths,
            bundle.variables,
            bundle.decision_nodes,
            refund_context.dominance_applied,
        )
        assert steps1 == steps2


# --- build_injection_bundle ---


class TestBuildInjectionBundle:
    def test_returns_injection_bundle(self, refund_context, bundle):
        inj = build_injection_bundle(refund_context, bundle)
        assert inj.scaffold_steps
        assert inj.generation_params["temperature"] == 0.0

    def test_system_prompt_has_enforcement_markers(self, refund_context, bundle):
        inj = build_injection_bundle(refund_context, bundle)
        assert "BEGIN POLICY ENFORCEMENT" in inj.system_prompt_additions
        assert "END POLICY ENFORCEMENT" in inj.system_prompt_additions


# --- format_full_prompt ---


class TestFormatFullPrompt:
    def test_basic_prompt(self, refund_context, bundle):
        inj = build_injection_bundle(refund_context, bundle)
        prompt = format_full_prompt("I want to return my laptop", inj)
        assert "system" in prompt
        assert "user" in prompt
        assert "I want to return my laptop" in prompt["user"]

    def test_scaffold_in_user_prompt(self, refund_context, bundle):
        inj = build_injection_bundle(refund_context, bundle)
        prompt = format_full_prompt("refund request", inj)
        assert "enforcement scaffold" in prompt["user"].lower()

    def test_base_system_prompt_preserved(self, refund_context, bundle):
        inj = build_injection_bundle(refund_context, bundle)
        prompt = format_full_prompt("test", inj, base_system_prompt="You are a helpful assistant.")
        assert "You are a helpful assistant." in prompt["system"]
        assert "POLICY ENFORCEMENT" in prompt["system"]
