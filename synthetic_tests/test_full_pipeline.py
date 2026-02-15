#!/usr/bin/env python3
"""
Full pipeline test using synthetic data with AWS Bedrock Claude.

Tests the complete PolicyLLM pipeline on synthetic documents:
Document → Extraction → Validation → Enforcement → ComplianceDecision

This aligns with the project scope: extracting, validating, and enforcing
organizational policies on LLM outputs at runtime.
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Extractor.src import pipeline as extraction_pipeline
from Extractor.src.config import load_config
from Extractor.src.llm.client import LLMClient
from Validation.policy_ir_builder import build_policy_ir
from Validation.decision_graph import build_decision_graph
from Validation.conflict_detector import detect_conflicts
from Validation.resolution import resolve_conflicts
from Validation.bundle_compiler import compile_bundle
from Enforcement.bundle_loader import load_bundle
from Enforcement.orchestrator import enforce, EnforcementConfig


def test_single_document(doc_path: Path, stage_name: str, output_dir: Path, queries: list = None):
    """
    Test complete pipeline on a single synthetic document.
    
    Pipeline: Document → Extraction → Validation → Enforcement
    """
    print("\n" + "="*100)
    print(f"TESTING: {stage_name} - {doc_path.name}")
    print("="*100)
    
    # Create output directory for this test
    test_output = output_dir / stage_name / doc_path.stem
    test_output.mkdir(parents=True, exist_ok=True)
    
    results = {
        "document": str(doc_path),
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "status": "started"
    }
    
    try:
        # ====================================================================
        # STAGE 1: EXTRACTION
        # ====================================================================
        print("\n[1/3] EXTRACTION - Extracting policies from document...")
        
        extraction_dir = test_output / "extraction"
        extraction_dir.mkdir(exist_ok=True)
        
        # Load extraction config (using ChatGPT config as template, but will use Bedrock)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "Extractor/configs/config.chatgpt.yaml"
        
        if not config_path.exists():
            raise Exception(f"Config file not found: {config_path}")
        
        config = load_config(str(config_path))
        
        # Override with Bedrock settings (use full ARN as model_id)
        config.llm.provider = "bedrock_claude"
        config.llm.model_id = "arn:aws:bedrock:us-east-2:660201002087:inference-profile/global.anthropic.claude-opus-4-5-20251101-v1:0"
        config.llm.region = os.getenv("AWS_REGION", "us-east-2")
        
        # Run extraction pipeline
        extraction_pipeline.run_pipeline(
            input_path=str(doc_path),
            output_dir=str(extraction_dir),
            tenant_id="synthetic_test",
            batch_id=f"{stage_name}_{doc_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            stage5_input=None
        )
        
        # Load extracted policies
        jsonl_files = [f for f in os.listdir(extraction_dir) if f.endswith(".jsonl")]
        if not jsonl_files:
            raise Exception("No policies.jsonl file generated")
        
        policies_file = extraction_dir / jsonl_files[0]
        with open(policies_file) as f:
            policies = [json.loads(line) for line in f if line.strip()]
        
        print(f"✓ Extracted {len(policies)} policies")
        results["extraction"] = {
            "status": "success",
            "policies_count": len(policies),
            "policies_file": str(policies_file)
        }
        
        # ====================================================================
        # STAGE 2: VALIDATION
        # ====================================================================
        print("\n[2/3] VALIDATION - Building policy IR and decision graph...")
        
        validation_dir = test_output / "validation"
        validation_dir.mkdir(exist_ok=True)
        
        # Build intermediate representation
        policy_ir = build_policy_ir(policies)
        
        # Build decision graph
        decision_graph = build_decision_graph(policy_ir)
        
        # Detect conflicts
        conflict_report = detect_conflicts(decision_graph, policy_ir)
        
        # Resolve conflicts
        resolution_report = resolve_conflicts(conflict_report, decision_graph)
        
        # Compile bundle
        bundle = compile_bundle(policy_ir, decision_graph, conflict_report, resolution_report)
        
        # Save bundle
        bundle_path = validation_dir / "compiled_bundle.json"
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2, default=str)
        
        print(f"✓ Validation complete: {len(policy_ir.get('variables', {}))} variables, "
              f"{len(policy_ir.get('conditional_rules', []))} rules")
        print(f"✓ Decision graph: {len(decision_graph['decision_nodes'])} nodes, "
              f"{len(decision_graph['compiled_paths'])} paths")
        print(f"✓ Conflicts: {len(conflict_report.get('logical_conflicts', []))} detected, "
              f"{len(resolution_report.get('escalations', []))} escalated")
        
        results["validation"] = {
            "status": "success",
            "variables_count": len(policy_ir.get('variables', {})),
            "rules_count": len(policy_ir.get('conditional_rules', [])),
            "decision_nodes": len(decision_graph['decision_nodes']),
            "compiled_paths": len(decision_graph['compiled_paths']),
            "conflicts_detected": len(conflict_report.get('logical_conflicts', [])),
            "escalations": len(resolution_report.get('escalations', [])),
            "bundle_path": str(bundle_path)
        }
        
        # ====================================================================
        # STAGE 3: ENFORCEMENT
        # ====================================================================
        print("\n[3/3] ENFORCEMENT - Testing policy enforcement on queries...")
        
        enforcement_dir = test_output / "enforcement"
        enforcement_dir.mkdir(exist_ok=True)
        
        # Load bundle for enforcement (returns tuple: bundle, index)
        loaded_bundle, bundle_index = load_bundle(str(bundle_path))
        
        # Create LLM client for enforcement (reuse Bedrock settings)
        llm_client = LLMClient(
            provider="bedrock_claude",
            model_id="arn:aws:bedrock:us-east-2:660201002087:inference-profile/global.anthropic.claude-opus-4-5-20251101-v1:0",
            region="us-east-2",
            temperature=0.1,
            max_tokens=4096
        )
        
        # Create enforcement config (use defaults)
        enforcement_config = EnforcementConfig()
        
        # Use provided queries or create test queries
        if not queries:
            queries = [
                {
                    "query_id": "TEST-001",
                    "query": "Can you help me process this customer request?",
                    "category": "valid_path"
                }
            ]
        
        enforcement_results = []
        for query in queries[:3]:  # Test first 3 queries
            try:
                result = enforce(
                    query=query["query"],
                    bundle=loaded_bundle,
                    bundle_index=bundle_index,
                    llm_client=llm_client,
                    config=enforcement_config
                )
                
                enforcement_results.append({
                    "query_id": query.get("query_id"),
                    "query": query.get("query"),
                    "expected_category": query.get("category"),
                    "action": str(result.action),
                    "score": result.score,
                    "violations": result.violations,
                    "status": "success"
                })
                
                print(f"  ✓ Query {query.get('query_id')}: {result.action} (score: {result.score:.2f})")
                
            except Exception as e:
                enforcement_results.append({
                    "query_id": query.get("query_id"),
                    "query": query.get("query"),
                    "status": "failed",
                    "error": str(e)
                })
                print(f"  ✗ Query {query.get('query_id')} failed: {e}")
        
        # Save enforcement results
        enforcement_results_path = enforcement_dir / "results.json"
        with open(enforcement_results_path, "w") as f:
            json.dump(enforcement_results, f, indent=2)
        
        print(f"✓ Enforcement tested on {len(enforcement_results)} queries")
        
        results["enforcement"] = {
            "status": "success",
            "queries_tested": len(enforcement_results),
            "results_file": str(enforcement_results_path)
        }
        
        results["status"] = "completed"
        print(f"\n✓ Pipeline test COMPLETED for {stage_name}/{doc_path.name}")
        
    except Exception as e:
        print(f"\n✗ Pipeline test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
    
    return results


def main():
    """Run full pipeline tests on synthetic data."""
    print("\n" + "="*100)
    print("SYNTHETIC DATA - FULL PIPELINE TEST")
    print("="*100)
    print("Project: PolicyLLM (ConstitutionCreator)")
    print("Pipeline: Document → Extraction → Validation → Enforcement")
    print(f"LLM: Claude Opus 4.5 via AWS Bedrock")
    print(f"GPU: 0 (CUDA_VISIBLE_DEVICES)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    # Setup paths
    synthetic_root = Path("/scratch2/f004ndc/ConstitutionCreator/synthetic_data")
    output_dir = Path(__file__).parent / "output" / "pipeline_tests"
    
    if not synthetic_root.exists():
        print(f"\n✗ Synthetic data root not found: {synthetic_root}")
        sys.exit(1)
    
    # Clear and create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test queries
    queries_file = synthetic_root / "stage4_mixed" / "test_queries.json"
    if queries_file.exists():
        with open(queries_file) as f:
            queries_data = json.load(f)
        test_queries = queries_data.get("queries", [])
        print(f"\n✓ Loaded {len(test_queries)} test queries")
    else:
        test_queries = []
        print("\n⚠ No test queries found, will use default queries")
    
    # Select test documents (one from each stage for comprehensive testing)
    test_cases = [
        {
            "stage": "stage1_explicit",
            "doc": synthetic_root / "stage1_explicit" / "documents" / "doc_001.md"
        },
        {
            "stage": "stage2_conflicts",
            "doc": synthetic_root / "stage2_conflicts" / "documents" / "doc_001.md"
        },
        {
            "stage": "stage3_implicit",
            "doc": synthetic_root / "stage3_implicit" / "documents" / "doc_001.md"
        },
        {
            "stage": "stage4_mixed",
            "doc": synthetic_root / "stage4_mixed" / "documents" / "doc_001.md"
        },
    ]
    
    # Run pipeline tests
    all_results = []
    for test_case in test_cases:
        if not test_case["doc"].exists():
            print(f"\n⚠ Skipping {test_case['stage']}: document not found")
            continue
        
        result = test_single_document(
            doc_path=test_case["doc"],
            stage_name=test_case["stage"],
            output_dir=output_dir,
            queries=test_queries
        )
        all_results.append(result)
    
    # Create summary
    print("\n" + "="*100)
    print("TEST SUMMARY")
    print("="*100)
    
    successful = sum(1 for r in all_results if r["status"] == "completed")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    for result in all_results:
        status_icon = "✓" if result["status"] == "completed" else "✗"
        print(f"\n{status_icon} {result['stage']}")
        
        if result["status"] == "completed":
            print(f"   Extraction: {result['extraction']['policies_count']} policies")
            print(f"   Validation: {result['validation']['variables_count']} variables, "
                  f"{result['validation']['rules_count']} rules")
            print(f"   Enforcement: {result['enforcement']['queries_tested']} queries tested")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Save comprehensive summary
    summary = {
        "test_run": {
            "timestamp": datetime.now().isoformat(),
            "project": "PolicyLLM (ConstitutionCreator)",
            "pipeline": "Document → Extraction → Validation → Enforcement",
            "llm": {
                "model": "claude-opus-4-5-20251101",
                "provider": "aws-bedrock",
                "gpu": 0
            }
        },
        "results": {
            "total": len(all_results),
            "successful": successful,
            "failed": failed
        },
        "test_cases": all_results
    }
    
    summary_path = output_dir.parent / "pipeline_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*100}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
