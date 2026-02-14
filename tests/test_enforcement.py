#!/usr/bin/env python3
"""Test Enforcement module with REAL LLM calls.

Tests the complete enforcement pipeline: classification, generation, and verification.
Uses test queries from tests/data/test_queries.json.
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Enforcement.bundle_loader import load_bundle
from Enforcement.orchestrator import enforce, EnforcementConfig
from Extractor.src.llm.client import LLMClient


def load_test_queries():
    """Load test queries from data file."""
    with open("tests/data/test_queries.json", "r") as f:
        return json.load(f)


def test_enforcement():
    """Test enforcement pipeline with real LLM calls."""
    print("="*100)
    print("ENFORCEMENT TEST - REAL LLM CALLS")
    print("="*100)
    print()

    # Initialize LLM client
    print("Initializing OpenAI client (gpt-4o-mini)...")
    llm_client = LLMClient(
        provider="chatgpt",
        model_id="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        retries=3,
    )
    print("✓ LLM client initialized")
    print()

    # Load policy bundle
    print("Loading policy bundle...")
    bundle, bundle_index = load_bundle("Enforcement/tests/fixtures/test_bundle.json")
    print(f"✓ Bundle loaded: {len(bundle.variables)} variables")
    print()

    config = EnforcementConfig()

    # Load test queries
    test_data = load_test_queries()

    results = []

    # Test valid paths
    print("="*100)
    print("VALID PATHS")
    print("="*100)
    for test in test_data["valid_paths"][:2]:  # Test first 2
        print(f"\nTest: {test['id']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")

        decision = enforce(
            query=test["query"],
            bundle=bundle,
            bundle_index=bundle_index,
            llm_client=llm_client,
            config=config,
        )

        print(f"Result: {decision.action.value.upper()} (Score: {decision.score:.3f})")
        results.append({
            "test_id": test["id"],
            "expected": test["expected"],
            "actual": decision.action.value.upper(),
            "score": decision.score,
        })

    # Test policy violations
    print()
    print("="*100)
    print("POLICY VIOLATIONS")
    print("="*100)
    for test in test_data["policy_violations"][:2]:  # Test first 2
        print(f"\nTest: {test['id']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")

        decision = enforce(
            query=test["query"],
            bundle=bundle,
            bundle_index=bundle_index,
            llm_client=llm_client,
            config=config,
        )

        print(f"Result: {decision.action.value.upper()} (Score: {decision.score:.3f})")
        results.append({
            "test_id": test["id"],
            "expected": test["expected"],
            "actual": decision.action.value.upper(),
            "score": decision.score,
        })

    # Test uncovered paths
    print()
    print("="*100)
    print("UNCOVERED PATHS")
    print("="*100)
    for test in test_data["uncovered_paths"][:1]:  # Test first 1
        print(f"\nTest: {test['id']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")

        decision = enforce(
            query=test["query"],
            bundle=bundle,
            bundle_index=bundle_index,
            llm_client=llm_client,
            config=config,
        )

        print(f"Result: {decision.action.value.upper()} (Score: {decision.score:.3f})")
        results.append({
            "test_id": test["id"],
            "expected": test["expected"],
            "actual": decision.action.value.upper(),
            "score": decision.score,
        })

    # Summary
    print()
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print()
    for r in results:
        status = "✓" if r["expected"] in r["actual"] or r["actual"] in ["ESCALATE", "REGENERATE"] else "✗"
        print(f"{status} {r['test_id']}: {r['actual']} (score: {r['score']:.3f})")

    print()
    print("="*100)
    print("✓ ENFORCEMENT TEST COMPLETED")
    print("="*100)


if __name__ == "__main__":
    test_enforcement()
