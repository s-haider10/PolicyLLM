#!/bin/bash
# Quick activation script for bedrock_env

cd "$(dirname "$0")"
source bedrock_env/bin/activate
echo "✓ Bedrock environment activated"
echo "✓ Python: $(python --version)"
echo "✓ Ready for generation with Claude via AWS Bedrock"
