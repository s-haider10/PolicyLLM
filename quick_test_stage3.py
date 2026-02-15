#!/usr/bin/env python3
"""Quick test of stage3_implicit extraction"""
import sys
import os
import json

# Setup path
sys.path.insert(0, '/scratch2/f004ndc/ConstitutionCreator/Extractor/src')
os.chdir('/scratch2/f004ndc/ConstitutionCreator')

from dotenv import load_dotenv
load_dotenv('.env')

# Now test
from Extractor.src.pipeline import run_document_pipeline

doc_path = '/scratch2/f004ndc/ConstitutionCreator/synthetic_data/stage3_implicit/documents'

# Get first markdown file
import glob
markdown_files = glob.glob(f'{doc_path}/*.md')
if not markdown_files:
    print(f"No markdown files in {doc_path}")
    sys.exit(1)

md_file = markdown_files[0]
print(f"Testing: {md_file}")

try:
    result = run_document_pipeline(md_file)
    print(f"✓ Extraction: {len(result.get('extracted_policies', []))} policies")
    for pol in result.get('extracted_policies', []):
        print(f"  - {pol.get('policy_id', 'NO-ID')}: {pol.get('domain', 'unknown')}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

