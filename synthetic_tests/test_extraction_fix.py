#!/usr/bin/env python3
"""
Quick test of the improved extraction pipeline on stage1_explicit/doc_001.md
Should extract 4 separate policies instead of merging them into 1.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Extractor.src import pipeline as extraction_pipeline
from Extractor.src.config import load_config

def test_extraction():
    """Test that stage1_explicit now extracts 4 policies."""
    
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "Extractor/configs/config.chatgpt.yaml"
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    
    config = load_config(str(config_path))
    
    # Override with Bedrock settings
    config.llm.provider = "bedrock_claude"
    config.llm.model_id = "arn:aws:bedrock:us-east-2:660201002087:inference-profile/global.anthropic.claude-opus-4-5-20251101-v1:0"
    config.llm.region = "us-east-2"
    config.llm.max_tokens = 8192  # Increase for policy discovery
    
    # Test document
    test_doc = project_root / "synthetic_data/stage1_explicit/documents/doc_001.md"
    if not test_doc.exists():
        print(f"❌ Test document not found: {test_doc}")
        return
    
    # Output directory
    output_dir = project_root / "synthetic_tests/test_extraction_fix"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TESTING IMPROVED EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"\nDocument: {test_doc.name}")
    print(f"Expected: 4 distinct policies")
    print(f"Previous: 1 merged policy\n")
    
    # Run extraction
    print("Running extraction...")
    extraction_pipeline.run_pipeline(
        input_path=str(test_doc),
        output_dir=str(output_dir),
        tenant_id="test",
        batch_id="extraction_fix_test",
        config=config,
        stage5_input=None
    )
    
    # Load results
    import json
    jsonl_files = list(output_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("❌ No policies extracted!")
        return
    
    policies = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    policies.append(json.loads(line))
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: Extracted {len(policies)} policies")
    print(f"{'=' * 80}\n")
    
    for i, pol in enumerate(policies, 1):
        policy_id = pol.get("policy_id", "UNKNOWN")
        domain = pol.get("metadata", {}).get("domain", "unknown")
        num_conditions = len(pol.get("conditions", []))
        num_actions = len(pol.get("actions", []))
        
        print(f"Policy {i}: {policy_id}")
        print(f"  Domain: {domain}")
        print(f"  Conditions: {num_conditions}")
        print(f"  Actions: {num_actions}")
        
        # Show action types
        actions = [a.get("action") for a in pol.get("actions", [])]
        if actions:
            print(f"  Action types: {', '.join(actions)}")
        print()
    
    # Verify results
    print(f"{'=' * 80}")
    if len(policies) >= 4:
        print("✅ SUCCESS: Extracted multiple policies (expected 4, got {})".format(len(policies)))
        
        # Check if domains are separated
        domains = [pol.get("metadata", {}).get("domain") for pol in policies]
        unique_domains = set(domains)
        print(f"✅ Domains separated: {unique_domains}")
        
        # Check if policy IDs are different
        policy_ids = [pol.get("policy_id") for pol in policies]
        if len(set(policy_ids)) == len(policy_ids):
            print(f"✅ All policy IDs are unique")
        else:
            print(f"⚠️  Some policy IDs are duplicated")
            
    else:
        print(f"❌ FAILED: Only extracted {len(policies)} policies (expected 4)")
        print("   The policies may still be getting merged.")
    
    print(f"{'=' * 80}\n")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    test_extraction()
