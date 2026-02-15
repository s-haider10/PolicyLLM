#!/bin/bash
# Launch script for synthetic data generation using Llama-3-8B on GPU 0
# Llama-3 is much better at following instructions than Phi-2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
ROOT_DIR="$SCRIPT_DIR/.."
PYTHON_ENV="/scratch2/f004ndc/LLM Second-Order Effects/2OE_env/bin/python"
SESSION_NAME="synth_llama3"

echo "==================================="
echo "Synthetic Data Generation with Llama-3-8B"
echo "==================================="
echo "Starting generation in tmux session: $SESSION_NAME"
echo "Using GPU: 0"
echo "Model: Meta-Llama-3-8B (8 billion parameters)"
echo "==================================="

# Create tmux session
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR"

# Set GPU and run generation
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=0" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting synthetic dataset generation with Llama-3-8B...'" C-m
tmux send-keys -t $SESSION_NAME "'$PYTHON_ENV' generate_dataset.py --root '$ROOT_DIR' --num-policies 20 --num-documents 20 --num-queries 50 --seed 42 --use-llama3" C-m
tmux send-keys -t $SESSION_NAME "echo 'Generation complete!'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Output directory: $ROOT_DIR'" C-m

echo ""
echo "==================================="
echo "Tmux session created: $SESSION_NAME"
echo "==================================="
echo "To monitor progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach: Ctrl+b, then d"
echo "To check progress:"
echo "  find $ROOT_DIR -type d -name documents | while read d; do echo \"\$d: \$(ls \"\$d\" 2>/dev/null | wc -l) files\"; done"
echo "==================================="
