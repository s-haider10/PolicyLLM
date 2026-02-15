#!/usr/bin/env python3
"""Quick test of stage3_implicit extraction"""
import sys
import os
import json
import glob
from pathlib import Path

# Setup path
sys.path.insert(0, '/scratch2/f004ndc/ConstitutionCreator')
os.chdir('/scratch2/f004ndc/ConstitutionCreator')

from dotenv import load_dotenv
load_dotenv('.env')

from Extractor.src.pipeline import run_pipeline
from Extractor.src.config import load_config
from datetime import datetime

doc_dir = Path('/scratch2/f004ndc/ConstitutionCreator/synthetic_data/stage3_implicit/documents')
doc_file = list(doc_dir.glob('*.md'))[0]
print(f"Testing: {doc_file}")

# Create temp output dir
output_dir = Path('/tmp/stage3_test')
output_dir.mkdir(exist_ok=True)

# Load config
project_root = Path(__file__).parent
config_path = project_root / "Extractor/configs/config.chatgpt.yaml"
config = load_config(str(config_path))

# Override with Bedrock
config.llm.provider = "bedrock_claude"
config.llm.model_id = "arn:aws:bedrock:us-east-2:660201002087:inference-profile/global.anthropic.claude-opus-4-5-20251101-v1:0"
config.llm.region = os.getenv("AWS_REGION", "us-east-2")

try:
    print("Running extraction pipeline...")
    run_pipeline(
        input_path=str(doc_file),
        output_dir=str(output_dir),
        tenant_id="stage3_test",
        batch_id=f"stage3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        stage5_input=None
    )
    
    # Load results
    jsonl_files = [f for f in os.listdir(output_dir) if f.endswith(".jsonl")]
    if jsonl_files:
        with open(output_dir / jsonl_files[0]) as f:
            policies = [json.loads(line) for line in f if line.strip()]
        print(f"✓ Extraction: {len(policies)} policies")
        for pol in policies:
            print(f"  - {pol.get('policy_id', 'NO-ID')}: {pol.get('domain', 'unknown')}")
    else:
        print("✗ No policies.jsonl found")
        for f in os.listdir(output_dir):
            print(f"  - {f}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
