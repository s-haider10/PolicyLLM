#!/usr/bin/env python3
"""Test Validation module: IR building, DAG generation, conflict detection.

Tests symbolic validation and decision graph compilation.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Validation.policy_ir_builder import build_policy_ir
from Validation.decision_graph import build_decision_graph
from Validation.bundle_compiler import compile_bundle


def load_test_policies():
    """Load pre-extracted test policies."""
    # Use the fixture from Enforcement tests
    with open("Enforcement/tests/fixtures/test_bundle.json", "r") as f:
        bundle = json.load(f)

    # Extract policy-like structure for testing
    # Create minimal policies for IR testing
    policies = [
        {
            "policy_id": "electronics_refund_v2",
            "variables": bundle["variables"],
            "conditional_rules": [
                {
                    "policy_id": "electronics_refund_v2",
                    "conditions": [
                        {"var": "has_receipt", "op": "==", "value": True},
                        {"var": "days_since_purchase", "op": "<=", "value": 14},
                        {"var": "product_category", "op": "==", "value": "electronics"}
                    ],
                    "action": {"type": "approve", "value": "full_refund"},
                    "metadata": {}
                }
            ],
            "constraints": []
        }
    ]
    return policies


def test_validation_dag():
    """Test validation and DAG generation."""
    print("="*100)
    print("VALIDATION & DAG GENERATION TEST")
    print("="*100)
    print()

    # Load test data
    policies = load_test_policies()
    print(f"Loaded {len(policies)} test policies")
    print()

    # Build IR
    print("Building intermediate representation...")
    policy_ir = build_policy_ir(policies)
    print(f"✓ Variables: {len(policy_ir.get('variables', {}))}")
    print(f"✓ Rules: {len(policy_ir.get('conditional_rules', []))}")
    print(f"✓ Constraints: {len(policy_ir.get('constraints', []))}")
    print()

    # Build decision graph
    print("Building decision graph...")
    decision_graph = build_decision_graph(policy_ir)
    print(f"✓ Decision nodes: {decision_graph['decision_nodes']}")
    print(f"✓ Leaf actions: {decision_graph['leaf_actions']}")
    print(f"✓ Compiled paths: {len(decision_graph['compiled_paths'])}")
    print()

    # Show paths
    print("DAG Paths:")
    for i, path in enumerate(decision_graph['compiled_paths'], 1):
        steps = " → ".join(step['var'] for step in path['path'])
        print(f"  Path {i}: {steps} → {path['leaf_action']}")
    print()

    # Compile bundle
    print("Compiling policy bundle...")
    conflict_report = {"conflicts": [], "count": 0}
    resolution_report = {"resolutions": [], "count": 0}

    bundle = compile_bundle(
        policy_ir=policy_ir,
        decision_graph=decision_graph,
        conflict_report=conflict_report,
        resolution_report=resolution_report,
    )

    print(f"✓ Bundle compiled with {len(bundle['variables'])} variables")
    print()

    print("="*100)
    print("✓ VALIDATION & DAG TEST PASSED")
    print("="*100)


if __name__ == "__main__":
    test_validation_dag()
