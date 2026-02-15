#!/bin/bash

# AWS Bedrock Claude Synthetic Data Generation Pipeline
# Launches the full generation with Claude via AWS Bedrock API

set -e

export CUDA_VISIBLE_DEVICES=0

# Load environment variables from .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Verify API key is set
if [ -z "$AWS_BEDROCK_API_KEY" ]; then
    echo "ERROR: AWS_BEDROCK_API_KEY not set. Please check .env file."
    exit 1
fi

echo "==================================="
echo "Synthetic Data Generation with Claude (AWS Bedrock)"
echo "==================================="

SESSION_NAME="synth_bedrock"
ROOT_DIR="/scratch2/f004ndc/ConstitutionCreator/synthetic_data"

# Kill any existing session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR/generation_scripts"

echo ""
echo "==================================="
echo "Tmux session created: $SESSION_NAME"
echo "==================================="
echo "To monitor progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach: Ctrl+b, then d"
echo ""
echo "To check progress:"
echo "  find $ROOT_DIR -type d -name documents | while read d; do echo \"\$d: \$(ls \"\$d\" 2>/dev/null | wc -l) files\"; done"
echo "==================================="
echo ""

# Send the generation command
tmux send-keys -t "$SESSION_NAME" "
echo 'Starting synthetic dataset generation with Claude via Bedrock...'
export AWS_BEDROCK_API_KEY='$AWS_BEDROCK_API_KEY'
export AWS_REGION='${AWS_REGION:-us-west-2}'
'/scratch2/f004ndc/LLM Second-Order Effects/2OE_env/bin/python' generate_dataset.py \\
    --root '$ROOT_DIR/generation_scripts/..' \\
    --num-policies 20 \\
    --num-documents 20 \\
    --num-queries 50 \\
    --seed 42 \\
    --use-bedrock
echo 'Generation complete!'
echo 'Output directory: $ROOT_DIR/generation_scripts/..'
" "Enter"
