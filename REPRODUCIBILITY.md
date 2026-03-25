# Reproducibility Guide

This document describes the exact configuration, commands, and environment used to produce the evaluation results reported in the paper.

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| OS | macOS (Darwin) / Linux |
| LLM Provider | OpenAI (ChatGPT API) |
| LLM Model | `gpt-4o-mini` |
| Z3 Solver | 4.15.8.0 |
| sentence-transformers | 5.2.2 |
| pydantic | 2.12.5 |

## Setup

```bash
# 1. Clone and enter project
git clone <repo-url> PolicyLLM
cd PolicyLLM

# 2. Create virtual environment
uv venv
source .venv/bin/activate

# 3. Install dependencies (pinned versions)
uv pip install -r requirements.txt

# 4. Set OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Extraction Configuration

All extraction results use `Extractor/configs/config.chatgpt.yaml`:

```yaml
llm:
  provider: chatgpt
  model_id: gpt-4o-mini
  temperature: 0.1          # Low temperature for consistency
  max_tokens: 4096
  retries: 3
  backoff: 1.5
merge:
  similarity_threshold: 0.9  # Deduplication threshold
scope:
  fallback: all
  enable_regex: true
double_run:
  enabled: false             # Single run (set true for consensus mode)
```

**Key reproducibility settings:**
- `temperature: 0.1` — Near-deterministic LLM outputs
- `similarity_threshold: 0.9` — Conservative deduplication
- `double_run.enabled: false` — Single extraction pass (faster, cheaper)

## Enforcement Configuration

Enforcement uses these fixed parameters (defined in `Enforcement/orchestrator.py`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `generation_temperature` | 0.0 | Deterministic generation |
| `generation_max_tokens` | 2048 | Sufficient for policy responses |
| `judge_enabled` | true | Full 4-checker pipeline |
| `smt_enabled` | true | Z3 verification active |
| `regex_enabled` | true | PII hard gate active |
| `max_retries` | 2 | Up to 2 regeneration attempts |

## Scoring Weights

Compliance score: `S = 0.60 × SMT + 0.30 × Judge + 0.10 × Coverage`

| Component | Weight | Type |
|-----------|--------|------|
| SMT (Z3) | 0.60 | Weighted |
| Judge (LLM) | 0.30 | Weighted |
| Coverage | 0.10 | Weighted |
| Regex (PII) | — | Hard gate (triggers ESCALATE regardless of score) |

Action thresholds: PASS ≥ 0.95, AUTO_CORRECT ≥ 0.85, REGENERATE ≥ 0.70, ESCALATE < 0.70.

## Reproducing Paper Results

### Step 1: Extraction + Validation (on test PDFs)

```bash
python run_extract_tests_pdfs.py
```

This extracts policies from all PDFs in `tests/`, compiles bundles, and stores results in `results/`.

### Step 2: Full Evaluation (with all baselines)

```bash
# With LLM-dependent baselines (requires OpenAI API key)
python run_extract_tests_pdfs.py --eval-only

# Without LLM API (Keyword, Semantic, SMT-only baselines only)
python run_extract_tests_pdfs.py --eval-only --no-api
```

### Step 3: Generate Tables Only (from cached results)

```bash
python run_extract_tests_pdfs.py --tables-only
```

## Output Files

| File | Description |
|------|-------------|
| `results/per_dataset_results.csv` | Per-document extraction statistics |
| `results/ACL_metrics_comparison.csv` | Full comparison table (CSV) |
| `results/ACL_metrics_comparison.md` | Full comparison table (Markdown) |
| `results/computed_metrics.json` | Raw computed metrics (JSON, machine-readable) |
| `results/<dataset>/compiled_policy_bundle.json` | Compiled bundle per document |

## Evaluation Framework

All metrics are computed from actual pipeline runs:

- **Extraction metrics** (Policy Recall/Precision, Condition F1) are computed against expert-annotated ground truth in `eval/reference_data/extraction_gt.json`.
- **Enforcement metrics** (Compliance %, FP %, Latency) are computed by running each method on test scenarios in `eval/reference_data/enforcement_gt.json`.
- **Conflict F1** is computed from bundle conflict detection output vs. reference conflicts.
- **Baselines** are implemented in `eval/baselines.py` and run on the same data as PolicyLLM.

No numbers in the comparison table are hardcoded. Run `python run_extract_tests_pdfs.py --eval-only` to regenerate all metrics.

## Test Documents

| Document | Source | Domain | Pages |
|----------|--------|--------|-------|
| Zara Terms & Conditions | Public (zara.com) | Retail / Refund | ~25 |
| Health Insurance Portability Act | Public (US Federal Law) | Healthcare / Privacy | ~50 |
| NICE Guidelines | Public (NICE, UK) | Healthcare / HR | ~15 |
| WTO Agreement | Public (WTO) | Trade / Escalation | ~10 |
| YC SAFE | Public (Y Combinator) | Finance / Investment | ~5 |

## Cost Estimate

| Stage | Model | Approx. Cost |
|-------|-------|-------------|
| Extraction (5 PDFs) | gpt-4o-mini | ~$0.05–0.10 |
| Enforcement (24 scenarios) | gpt-4o-mini | ~$0.02–0.05 |
| LLM Baselines (6 methods) | gpt-4o-mini | ~$0.10–0.20 |
| **Total** | | **~$0.20–0.35** |

## Notes on Non-Determinism

- LLM outputs with `temperature > 0` introduce variation. All reported results use `temperature ≤ 0.1`.
- Z3 SAT solving and regex matching are fully deterministic.
- Sentence-transformer embeddings are deterministic given the same model weights.
- Small variations (±1-2%) between runs are expected due to LLM stochasticity at non-zero temperature.
