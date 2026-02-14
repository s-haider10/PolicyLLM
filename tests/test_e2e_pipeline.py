#!/usr/bin/env python3
"""Complete end-to-end pipeline test: Extraction → Validation → Enforcement.

Tests the entire PolicyLLM pipeline with REAL LLM calls using test data.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Extractor.src import pipeline as extraction_pipeline
from Extractor.src.config import load_config
from Validation.policy_ir_builder import build_policy_ir
from Validation.decision_graph import build_decision_graph
from Validation.bundle_compiler import compile_bundle
from Enforcement.bundle_loader import load_bundle
from Enforcement.orchestrator import enforce, EnforcementConfig
from Extractor.src.llm.client import LLMClient


def test_e2e_pipeline():
    """Test complete end-to-end pipeline with real LLM calls."""
    print("="*100)
    print("END-TO-END PIPELINE TEST - REAL LLM CALLS")
    print("="*100)
    print()
    print("Pipeline: Policy Document → Extraction → Validation → Enforcement")
    print("LLM: OpenAI gpt-4o-mini")
    print()

    # Setup
    test_dir = "tests/output/e2e"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    policy_doc = "tests/data/sample_policy.md"

    # ========================================================================
    # STAGE 1: EXTRACTION
    # ========================================================================
    print("="*100)
    print("STAGE 1: POLICY EXTRACTION (REAL LLM)")
    print("="*100)
    print(f"Input: {policy_doc}")
    print()

    config = load_config("Extractor/configs/config.chatgpt.yaml")

    print("Running 6-pass extraction pipeline...")
    extraction_pipeline.run_pipeline(
        input_path=policy_doc,
        output_dir=test_dir,
        tenant_id="e2e_test",
        batch_id=f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        stage5_input=None,
    )

    # Load extracted policies
    jsonl_files = [f for f in os.listdir(test_dir) if f.endswith(".jsonl")]
    policies_file = os.path.join(test_dir, jsonl_files[0])

    with open(policies_file, "r") as f:
        policies = [json.loads(line) for line in f if line.strip()]

    print(f"✓ Extracted {len(policies)} policies")
    for i, pol in enumerate(policies, 1):
        print(f"  Policy {i}: {pol.get('metadata', {}).get('domain', 'unknown')} domain")
    print()

    # ========================================================================
    # STAGE 2: VALIDATION
    # ========================================================================
    print("="*100)
    print("STAGE 2: VALIDATION & COMPILATION (SYMBOLIC)")
    print("="*100)
    print()

    # Build IR
    print("Building intermediate representation...")
    policy_ir = build_policy_ir(policies)
    print(f"✓ IR built: {len(policy_ir.get('variables', {}))} variables, {len(policy_ir.get('conditional_rules', []))} rules")

    # Build decision graph
    print("Building decision graph (DAG)...")
    decision_graph = build_decision_graph(policy_ir)
    print(f"✓ Decision graph: {len(decision_graph['decision_nodes'])} nodes, {len(decision_graph['compiled_paths'])} paths")

    # Compile bundle
    print("Compiling policy bundle...")
    bundle_path = os.path.join(test_dir, "compiled_bundle.json")

    conflict_report = {"conflicts": [], "count": 0, "summary": "No conflicts"}
    resolution_report = {"resolutions": [], "count": 0, "summary": "No resolutions"}

    bundle_data = compile_bundle(
        policy_ir=policy_ir,
        decision_graph=decision_graph,
        conflict_report=conflict_report,
        resolution_report=resolution_report,
    )

    with open(bundle_path, 'w') as f:
        json.dump(bundle_data, f, indent=2)

    print(f"✓ Bundle compiled: {bundle_path}")
    print()

    # Show DAG structure
    print("Decision Graph:")
    print(f"  Nodes: {decision_graph['decision_nodes']}")
    print(f"  Paths: {len(decision_graph['compiled_paths'])}")
    print()

    # ========================================================================
    # STAGE 3: ENFORCEMENT
    # ========================================================================
    print("="*100)
    print("STAGE 3: ENFORCEMENT (REAL LLM)")
    print("="*100)
    print()

    # Load bundle
    bundle, bundle_index = load_bundle(bundle_path)
    print(f"✓ Bundle loaded: {len(bundle.variables)} variables")

    # Initialize LLM
    llm_client = LLMClient(
        provider="chatgpt",
        model_id="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        retries=3,
    )
    print("✓ LLM client initialized")
    print()

    config = EnforcementConfig()

    # Load test queries
    with open("tests/data/test_queries.json", "r") as f:
        test_queries = json.load(f)

    # Test sample queries
    test_cases = [
        test_queries["valid_paths"][0],
        test_queries["policy_violations"][0],
        test_queries["uncovered_paths"][0],
    ]

    results = []

    for test in test_cases:
        print("-"*100)
        print(f"Test: {test['id']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print()

        decision = enforce(
            query=test["query"],
            bundle=bundle,
            bundle_index=bundle_index,
            llm_client=llm_client,
            config=config,
        )

        print(f"Result:")
        print(f"  Action: {decision.action.value.upper()}")
        print(f"  Score: {decision.score:.3f}")
        print(f"  SMT: {decision.audit_trail['scores']['smt']:.2f}")
        print(f"  Judge: {decision.audit_trail['scores']['judge']:.2f}")
        print()

        results.append({
            "test_id": test["id"],
            "expected": test["expected"],
            "actual": decision.action.value.upper(),
            "score": decision.score,
        })

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print()

    print("Stage 1 (Extraction):")
    print(f"  ✓ Policies extracted: {len(policies)}")
    print(f"  ✓ LLM calls: ~{len(policies) * 6}")
    print()

    print("Stage 2 (Validation):")
    print(f"  ✓ Variables: {len(policy_ir.get('variables', {}))}")
    print(f"  ✓ DAG paths: {len(decision_graph['compiled_paths'])}")
    print()

    print("Stage 3 (Enforcement):")
    print(f"  ✓ Test cases: {len(test_cases)}")
    print()

    print("Test Results:")
    for r in results:
        print(f"  {r['test_id']}: {r['actual']} (score: {r['score']:.3f})")
    print()

    print("="*100)
    print("✓ END-TO-END PIPELINE TEST COMPLETED")
    print("="*100)
    print()
    print(f"Output directory: {test_dir}/")


if __name__ == "__main__":
    test_e2e_pipeline()
