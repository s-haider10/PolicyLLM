#!/usr/bin/env python3
"""Final validation: Test all 4 stages with improved extraction"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/scratch2/f004ndc/ConstitutionCreator')
os.chdir('/scratch2/f004ndc/ConstitutionCreator')

from dotenv import load_dotenv
load_dotenv('.env')

from Extractor.src.pipeline import run_pipeline
from Extractor.src.config import load_config

stages = ['stage1_explicit', 'stage2_conflicts', 'stage3_implicit', 'stage4_mixed']
results = {}

project_root = Path('/scratch2/f004ndc/ConstitutionCreator')
config_path = project_root / "Extractor/configs/config.chatgpt.yaml"
config = load_config(str(config_path))

# Override with Bedrock
config.llm.provider = "bedrock_claude"
config.llm.model_id = "arn:aws:bedrock:us-east-2:660201002087:inference-profile/global.anthropic.claude-opus-4-5-20251101-v1:0"
config.llm.region = os.getenv("AWS_REGION", "us-east-2")

print("\n" + "="*80)
print("FINAL VALIDATION TEST - All Stages")
print("="*80)

for stage in stages:
    doc_dir = project_root / f"synthetic_data/{stage}/documents"
    doc_file = list(doc_dir.glob('*.md'))[0]
    
    output_dir = Path(f'/tmp/final_validation_{stage}')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{stage:20} - ", end='', flush=True)
    
    try:
        run_pipeline(
            input_path=str(doc_file),
            output_dir=str(output_dir),
            tenant_id=f"final_test_{stage}",
            batch_id=f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            stage5_input=None
        )
        
        jsonl_files = [f for f in os.listdir(output_dir) if f.endswith(".jsonl")]
        if jsonl_files:
            with open(output_dir / jsonl_files[0]) as f:
                policies = [json.loads(line) for line in f if line.strip()]
            results[stage] = len(policies)
            print(f"✓ {len(policies)} policies")
        else:
            results[stage] = 0
            print(f"✗ No output file")
    except Exception as e:
        results[stage] = -1
        print(f"✗ Error: {str(e)[:60]}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

expected = {
    'stage1_explicit': 4,
    'stage2_conflicts': 4,
    'stage3_implicit': 4,
    'stage4_mixed': 4,
}

all_pass = True
for stage, expected_count in expected.items():
    actual_count = results.get(stage, -1)
    status = "✓ PASS" if actual_count == expected_count else "✗ FAIL"
    print(f"{stage:20}: {actual_count:2} policies (expected {expected_count}) {status}")
    if actual_count != expected_count:
        all_pass = False

print("="*80)
if all_pass:
    print("✓ ALL STAGES PASS - Extraction fix is successful!")
else:
    print("✗ Some stages failed - Review results above")
print("="*80)

