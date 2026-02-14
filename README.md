# PolicyLLM

Neuro-symbolic framework for extracting, validating, and enforcing organizational policies on LLM outputs at runtime.

**Pipeline:** `Document → Extraction → Validation → Enforcement → ComplianceDecision`

## Quick Start

```bash
# Clone and setup
cd PolicyLLM
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Set up OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Run end-to-end test
uv run python tests/test_e2e_pipeline.py
```

## Testing the Pipeline

PolicyLLM includes 4 organized test files using shared test data:

### 1. End-to-End Test (Recommended)

Tests complete pipeline: Extraction → Validation → Enforcement with **REAL OpenAI calls**.

```bash
uv run python tests/test_e2e_pipeline.py
```

- **What:** Extracts policies from `tests/data/sample_policy.md` → builds DAG → enforces on 3 queries
- **Runtime:** 30-60 seconds
- **Cost:** ~$0.01-0.02 (gpt-4o-mini)

### 2. Extraction Tests

```bash
uv run python tests/test_extraction.py
```

- Tests 6-pass extraction pipeline with real LLM
- Validates policy structure, variables, rules
- **Runtime:** 20-40 seconds

### 3. Validation Tests (Symbolic, Fast)

```bash
uv run python tests/test_validation_dag.py
```

- Tests IR builder, DAG compilation, bundle generation
- **Runtime:** < 5 seconds
- **Cost:** $0 (no LLM calls)

### 4. Enforcement Tests

```bash
uv run python tests/test_enforcement.py
```

- Tests query classification, scaffold injection, verification
- Uses test queries from `tests/data/test_queries.json`
- **Runtime:** 30-50 seconds

### Using pytest

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_e2e_pipeline.py -v -s
```

## Test Data

Shared test data in `tests/data/`:

- **`sample_policy.md`** — Electronics return policy
- **`test_queries.json`** — Query test set (valid paths, violations, uncovered cases, edge cases)

## Usage Examples

### Individual Stages

```bash
# Extract policies
uv run python main.py extract tests/data/sample_policy.md \
    --out output/ \
    --config Extractor/configs/config.chatgpt.yaml

# Validate and compile
uv run python main.py validate output/policies_*.jsonl \
    --out output/compiled_bundle.json

# Enforce on query
uv run python main.py enforce \
    --bundle output/compiled_bundle.json \
    --query "I want to return my laptop" \
    --provider chatgpt \
    --model gpt-4o-mini
```

### Full Pipeline

```bash
uv run python main.py run tests/data/sample_policy.md \
    --query "Can I return my phone without a receipt?" \
    --provider chatgpt \
    --model gpt-4o-mini
```

## Compliance Scoring

```
S = 0.55 × SMT + 0.25 × Judge + 0.10 × Regex + 0.10 × Coverage
```

| Score | Action | Behavior |
|-------|--------|----------|
| ≥ 0.95 | PASS | Deliver response |
| 0.85-0.95 | AUTO_CORRECT | Retry with hints |
| 0.70-0.85 | REGENERATE | Retry with constraints |
| < 0.70 | ESCALATE | Block response |

**Override:** PII detected → ESCALATE always

## Architecture

```
┌─────────────────┐
│ Policy Document │  (PDF, DOCX, HTML, MD, TXT)
└────────┬────────┘
         │
    ┌────▼────┐
    │Extract  │  6-pass LLM pipeline → policies.jsonl
    └────┬────┘
         │
    ┌────▼────┐
    │Validate │  DAG + Z3 conflict detection → bundle.json
    └────┬────┘
         │
    ┌────▼────┐
    │Enforce  │  Scaffold injection → LLM → Verify (SMT+Judge+Regex+Coverage)
    └────┬────┘
         │
    ┌────▼────┐
    │Decision │  PASS | AUTO_CORRECT | REGENERATE | ESCALATE
    └─────────┘
```

See [system_design.md](system_design.md) for detailed architecture documentation.

## Project Structure

```
PolicyLLM/
├── tests/                     # Test suite
│   ├── test_e2e_pipeline.py   # E2E with real LLM
│   ├── test_extraction.py     # Extraction tests
│   ├── test_validation_dag.py # Validation (symbolic)
│   ├── test_enforcement.py    # Enforcement tests
│   └── data/                  # Shared test data
│       ├── sample_policy.md
│       └── test_queries.json
├── Extractor/                 # Stage 1: Extraction
│   ├── src/pipeline.py        # 6-pass orchestrator
│   ├── src/llm/client.py      # LLM abstraction
│   └── configs/               # YAML configs
├── Validation/                # Stage 2: Compilation
│   ├── policy_ir_builder.py   # Policy → IR
│   ├── decision_graph.py      # DAG builder
│   └── bundle_compiler.py     # Bundle generator
├── Enforcement/               # Stage 3: Runtime
│   ├── orchestrator.py        # Main pipeline
│   ├── postgen/smt.py         # Z3 verification
│   └── scoring.py             # Compliance scoring
├── main.py                    # Unified CLI
├── system_design.md           # Architecture docs
└── requirements.txt           # Dependencies
```

## Troubleshooting

### OpenAI API Setup

```bash
# Check API key
cat .env | grep OPENAI_API_KEY

# Test connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Import Errors

```bash
# From project root
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Test Failures

```bash
# Verbose output
uv run pytest tests/test_e2e_pipeline.py -v -s

# Single test
uv run python tests/test_validation_dag.py
```

## Documentation

- [system_design.md](system_design.md) — Detailed architecture
- [Extractor/overview/paper.tex](Extractor/overview/paper.tex) — ACL Industry Track paper

## Key Dependencies

- **pydantic** v2 — Type-safe schemas
- **z3-solver** — SMT verification
- **openai** — LLM API
- **sentence-transformers** — Embeddings
- **pytest** — Testing
