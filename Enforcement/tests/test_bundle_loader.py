"""Tests for bundle_loader.py — loading, validation, and indexing."""
import json
import os
import tempfile

import pytest

from Enforcement.bundle_loader import BundleIndex, load_bundle, validate_bundle_integrity
from Enforcement.schemas import CompiledPolicyBundle

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")


@pytest.fixture
def bundle_and_index():
    bundle, index = load_bundle(FIXTURE_PATH)
    return bundle, index


@pytest.fixture
def bundle(bundle_and_index):
    return bundle_and_index[0]


@pytest.fixture
def index(bundle_and_index):
    return bundle_and_index[1]


# --- Loading ---


def test_load_bundle_returns_tuple():
    result = load_bundle(FIXTURE_PATH)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_load_bundle_parses_schema(bundle):
    assert isinstance(bundle, CompiledPolicyBundle)
    assert bundle.schema_version == "1.0"


def test_load_bundle_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_bundle("/nonexistent/path.json")


def test_load_bundle_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json")
        f.flush()
        try:
            with pytest.raises(json.JSONDecodeError):
                load_bundle(f.name)
        finally:
            os.unlink(f.name)


def test_load_bundle_schema_mismatch():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"variables": "not a dict"}, f)
        f.flush()
        try:
            with pytest.raises(Exception):  # pydantic ValidationError
                load_bundle(f.name)
        finally:
            os.unlink(f.name)


# --- Bundle contents ---


def test_bundle_variables(bundle):
    assert "has_receipt" in bundle.variables
    assert bundle.variables["has_receipt"].type == "bool"
    assert "product_category" in bundle.variables
    assert bundle.variables["product_category"].type == "enum"
    assert "electronics" in bundle.variables["product_category"].values


def test_bundle_rules(bundle):
    assert len(bundle.conditional_rules) == 4
    pids = [r.policy_id for r in bundle.conditional_rules]
    assert "electronics_refund_v2" in pids
    assert "privacy_pii_v1" in pids


def test_bundle_constraints(bundle):
    assert len(bundle.constraints) == 2
    scopes = [c.scope for c in bundle.constraints]
    assert "always" in scopes
    assert "refund" in scopes


def test_bundle_paths(bundle):
    assert len(bundle.compiled_paths) == 3


def test_bundle_dominance(bundle):
    assert len(bundle.dominance_rules) == 1
    dr = bundle.dominance_rules[0]
    assert "electronics_refund_v2" in dr.when["policies_fire"]


def test_bundle_priority_lattice(bundle):
    assert bundle.priority_lattice["regulatory"] < bundle.priority_lattice["company"]


# --- Indexing ---


def test_index_rules_by_domain(index):
    assert "refund" in index.rules_by_domain
    assert len(index.rules_by_domain["refund"]) == 3
    assert "privacy" in index.rules_by_domain
    assert len(index.rules_by_domain["privacy"]) == 1


def test_index_rules_by_policy_id(index):
    assert "electronics_refund_v2" in index.rules_by_policy_id
    rule = index.rules_by_policy_id["electronics_refund_v2"]
    assert rule.action.type == "refund"
    assert rule.action.value == "full"


def test_index_paths_by_domain(index):
    assert "refund" in index.paths_by_domain
    assert len(index.paths_by_domain["refund"]) == 3


def test_index_constraints_by_scope(index):
    assert "always" in index.constraints_by_scope
    assert len(index.constraints_by_scope["always"]) == 1
    assert "refund" in index.constraints_by_scope
    assert len(index.constraints_by_scope["refund"]) == 1


def test_index_dominance_lookup(index):
    key = frozenset({"electronics_refund_v2", "clothing_refund_v1"})
    assert key in index.dominance_lookup


def test_index_escalation_lookup(index):
    key = frozenset({"electronics_refund_v2", "electronics_refund_late_v2"})
    assert key in index.escalation_lookup
    esc = index.escalation_lookup[key]
    assert "refund_team@corp.com" in esc.owners_to_notify


# --- Integrity validation ---


def test_validate_bundle_integrity_clean(bundle):
    warnings = validate_bundle_integrity(bundle)
    assert warnings == []


def test_validate_bundle_integrity_undefined_var():
    raw = json.loads(open(FIXTURE_PATH).read())
    raw["conditional_rules"][0]["conditions"].append(
        {"var": "nonexistent_var", "op": "==", "value": True}
    )
    bundle = CompiledPolicyBundle.model_validate(raw)
    warnings = validate_bundle_integrity(bundle)
    assert any("nonexistent_var" in w for w in warnings)


def test_validate_bundle_integrity_bad_decision_node():
    raw = json.loads(open(FIXTURE_PATH).read())
    raw["decision_nodes"].append("ghost_node")
    bundle = CompiledPolicyBundle.model_validate(raw)
    warnings = validate_bundle_integrity(bundle)
    assert any("ghost_node" in w for w in warnings)
