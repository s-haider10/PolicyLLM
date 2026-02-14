# PolicyLLM

End-to-end pipeline for extracting, validating, and enforcing organizational policies on LLM outputs.

**Pipeline:** `Document -> Extractor -> Validation -> Enforcement -> ComplianceDecision`

## Architecture

```
                    ┌──────────────┐
                    │  Policy Docs │  (PDF, DOCX, HTML, MD, TXT)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Extractor   │  6-pass extraction pipeline
                    │  (Group 1)   │  classify → components → entities → merge → metadata → validate
                    └──────┬───────┘
                           │  policies.jsonl
                    ┌──────▼───────┐
                    │  Validation  │  IR builder → decision graph → Z3 conflict detection → resolution
                    │  (Group 2)   │  → bundle compiler
                    └──────┬───────┘
                           │  compiled_policy_bundle.json
                    ┌──────▼───────┐
                    │ Enforcement  │  pregen → duringgen → LLM → postgen → scoring → action routing
                    │  (Group 3)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Decision    │  PASS | AUTO_CORRECT | REGENERATE | ESCALATE
                    └──────────────┘
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For local LLM (default):

```bash
# Install Ollama: https://ollama.com/download
ollama pull mistral
```

## Usage

### Full Pipeline

```bash
python main.py run policy_doc.pdf \
    --query "I want to return my laptop" \
    --provider ollama \
    --model mistral:latest
```

### For Evaluations

```bash
source .venv/bin/activate && python -m evals.run --suite evals/fixtures/refund_privacy_scenarios.json --provider stub --no-judge 2>&1 | tail -5

source .venv/bin/activate && python -m evals.run --suite evals/fixtures/refund_privacy_scenarios.json --provider chatgpt --model gpt-4o-mini --output /tmp/chatgpt_eval_report_v3.json 2>&1
```

### Individual Stages

```bash
# Stage 1: Extract policies from a document
python main.py extract policy_doc.pdf --out out/ --config Extractor/configs/config.example.yaml

# Stage 2: Compile extracted policies into an enforcement bundle
python main.py validate out/policies.jsonl --out compiled_policy_bundle.json

# Stage 3: Enforce policies against a user query
python main.py enforce \
    --bundle compiled_policy_bundle.json \
    --query "I want to return my laptop" \
    --provider ollama \
    --model mistral:latest
```

### Module-Level CLIs

Each module also has its own CLI:

```bash
# Extractor
python -m Extractor.src.cli input.pdf --out out/ --config Extractor/configs/config.example.yaml

# Validation
python -m Validation.cli policies.jsonl --out compiled_policy_bundle.json

# Enforcement
python -m Enforcement.cli --bundle compiled_policy_bundle.json --query "refund request" --provider ollama
```

## Compliance Scoring

Enforcement uses a weighted compliance score:

```
S = 0.55 * SMT + 0.25 * Judge + 0.10 * Regex + 0.10 * Coverage
```

| Threshold   | Action       | Behavior                         |
| ----------- | ------------ | -------------------------------- |
| S >= 0.95   | PASS         | Deliver response                 |
| 0.85 - 0.95 | AUTO_CORRECT | Re-generate with violation hints |
| 0.70 - 0.85 | REGENERATE   | Tighten scaffold, re-generate    |
| S < 0.70    | ESCALATE     | Block and notify policy owners   |

Regex failure (PII detected) always triggers ESCALATE regardless of score.

## Project Structure

```
PolicyLLM/
├── main.py                    # Unified CLI entry point
├── requirements.txt           # Python dependencies
├── Extractor/                 # Stage 1: Policy extraction from documents
│   ├── src/
│   │   ├── cli.py             # Extractor CLI
│   │   ├── pipeline.py        # 6-pass orchestrator
│   │   ├── config.py          # YAML config loader
│   │   ├── llm/client.py      # LLM provider abstraction (Ollama/Bedrock/OpenAI/Anthropic)
│   │   ├── passes/            # Pass 1-6 implementations
│   │   ├── regularize/        # PDF/DOCX/HTML/MD normalizers
│   │   ├── schemas/           # Canonical and policy Pydantic models
│   │   └── storage/           # JSONL + index writers
│   ├── tests/                 # Extractor unit tests
│   ├── configs/               # YAML config templates
│   ├── sample_docs/           # Sample input documents
│   └── overview/              # Design documents
├── Validation/                # Stage 2: Policy IR, conflict detection, bundle compilation
│   ├── policy_ir_builder.py   # Extractor output -> Policy IR (the bridge)
│   ├── decision_graph.py      # Decision tree compilation
│   ├── conflict_detector.py   # Z3-based pairwise conflict detection
│   ├── resolution.py          # Priority-based conflict resolution
│   ├── bundle_compiler.py     # Produces compiled_policy_bundle.json
│   ├── schema_discovery.py    # Action clustering via Sentence-BERT
│   ├── z3_utils.py            # Shared Z3 helpers
│   └── cli.py                 # Validation CLI
├── Enforcement/               # Stage 3: Runtime policy enforcement
│   ├── schemas.py             # All Pydantic models
│   ├── bundle_loader.py       # Bundle loading + O(1) indexes
│   ├── ir.py                  # Z3 variable creation and encoding
│   ├── pregen.py              # Query classification + rule retrieval + dominance
│   ├── duringgen.py           # Scaffold serialization + prompt injection
│   ├── postgen/               # Post-generation verification
│   │   ├── regex.py           # PII/pattern detection
│   │   ├── smt.py             # Z3 formal verification
│   │   └── judge.py           # Judge LLM semantic evaluation
│   ├── scoring.py             # Weighted compliance scoring
│   ├── orchestrator.py        # Full pipeline orchestration + retry loop
│   ├── audit.py               # JSONL audit log with SHA256 hash chain
│   ├── cli.py                 # Enforcement CLI
│   └── tests/                 # 119 unit + integration + e2e tests
└── system_design.md           # Detailed system architecture documentation
```

## LLM Providers

Configured via `--provider` flag:

- `ollama` — Local Ollama (default, uses `mistral:latest`)
- `bedrock_claude` — AWS Bedrock Claude
- `chatgpt` — OpenAI GPT
- `anthropic` — Anthropic Claude (direct API)
- `stub` — No-op stub for testing

## Testing

```bash
# Enforcement tests (119 tests)
python -m pytest Enforcement/tests/ -v

# Extractor tests
python -m pytest Extractor/tests/test_passes.py -v

# All tests
python -m pytest Enforcement/tests/ Extractor/tests/test_passes.py -v
```

## Key Dependencies

- **pydantic** — Schema validation across all modules
- **z3-solver** — SMT formal verification of LLM responses
- **sentence-transformers** — Action clustering in schema discovery
- **torch** — Embedding computation backend
- **boto3** — AWS Bedrock LLM provider
