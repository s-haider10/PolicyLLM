#!/bin/bash
# Run synthetic data pipeline tests on GPU 0 with AWS Bedrock Claude

set -e

# Configuration
PROJECT_DIR="/scratch2/f004ndc/ConstitutionCreator"
TEST_DIR="$PROJECT_DIR/synthetic_tests"
VENV_DIR="$PROJECT_DIR/.venv"
AWS_REGION="us-east-2"
GPU_ID="0"

# Setup GPU environment
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "======================================================================"
echo "SYNTHETIC DATA PIPELINE TEST - GPU $GPU_ID"
echo "======================================================================"
echo "Project: $PROJECT_DIR"
echo "Test dir: $TEST_DIR"
echo "GPU: $GPU_ID"
echo "AWS Region: $AWS_REGION"
echo "======================================================================"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(cat "$PROJECT_DIR/.env" | grep -v '^#' | xargs)
    echo "✓ Loaded environment from $PROJECT_DIR/.env"
else
    echo "⚠ No .env file found at $PROJECT_DIR/.env"
fi

# Also load from synthetic_data generation scripts
if [ -f "$PROJECT_DIR/synthetic_data/generation_scripts/.env" ]; then
    export $(cat "$PROJECT_DIR/synthetic_data/generation_scripts/.env" | grep -v '^#' | xargs)
    echo "✓ Loaded environment from synthetic_data generation scripts"
fi

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✓ Activated virtual environment: $VENV_DIR"
else
    echo "✗ Virtual environment not found at $VENV_DIR"
    echo "   Creating venv..."
    cd "$PROJECT_DIR"
    python3.12 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "✓ Created and activated virtual environment"
fi

# Install dependencies if needed
echo ""
echo "Installing dependencies..."
cd "$PROJECT_DIR"

# Install core requirements
pip install -q \
    python-dotenv \
    numpy \
    torch \
    transformers \
    boto3

# Install from requirements if it exists
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -q -r "$PROJECT_DIR/requirements.txt"
elif [ -f "$PROJECT_DIR/main.py" ]; then
    echo "✓ Project structure confirmed"
fi

echo "✓ Dependencies installed"

# Verify AWS credentials
echo ""
echo "Verifying AWS configuration..."
if [ -z "$AWS_BEARER_TOKEN_BEDROCK" ] && [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "⚠ Warning: No AWS credentials found in environment"
fi

# Run tests
echo ""
echo "Starting FULL PIPELINE tests..."
echo "======================================================================"
cd "$TEST_DIR"
python test_full_pipeline.py

echo ""
echo "======================================================================"
echo "Full pipeline tests completed!"
echo "Results saved to: $TEST_DIR/output"
echo "======================================================================"
