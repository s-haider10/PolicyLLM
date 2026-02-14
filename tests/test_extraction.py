#!/usr/bin/env python3
"""Test policy extraction with REAL LLM calls.

Tests the Extractor module's 6-pass pipeline with real OpenAI API calls.
"""

import json
import os
import shutil
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Extractor.src import pipeline
from Extractor.src.config import load_config


def test_extraction():
    """Test extraction pipeline with real LLM calls."""
    print("="*100)
    print("EXTRACTION TEST - REAL LLM CALLS")
    print("="*100)
    print()

    # Setup
    test_dir = "tests/output/extraction"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # Load config
    config = load_config("Extractor/configs/config.chatgpt.yaml")
    policy_file = "tests/data/sample_policy.md"

    print(f"Provider: {config.llm.provider}")
    print(f"Model: {config.llm.model_id}")
    print(f"Input: {policy_file}")
    print()

    # Run extraction
    print("Running 6-pass extraction pipeline...")
    pipeline.run_pipeline(
        input_path=policy_file,
        output_dir=test_dir,
        tenant_id="test",
        batch_id="extraction_test",
        config=config,
        stage5_input=None,
    )

    # Analyze results
    jsonl_files = [f for f in os.listdir(test_dir) if f.endswith(".jsonl")]
    assert jsonl_files, "No policies extracted"

    policies_file = os.path.join(test_dir, jsonl_files[0])
    with open(policies_file, "r") as f:
        policies = [json.loads(line) for line in f if line.strip()]

    print()
    print(f"✓ Extracted {len(policies)} policies")

    for i, pol in enumerate(policies, 1):
        print(f"\nPolicy {i}:")
        print(f"  Domain: {pol.get('metadata', {}).get('domain', 'unknown')}")
        print(f"  Variables: {len(pol.get('variables', {}))}")
        print(f"  Rules: {len(pol.get('conditional_rules', []))}")
        print(f"  Constraints: {len(pol.get('constraints', []))}")

    print()
    print("="*100)
    print("✓ EXTRACTION TEST PASSED")
    print("="*100)
    return policies


if __name__ == "__main__":
    test_extraction()
