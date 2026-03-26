# Reproducibility Guide

This document describes the exact configuration, commands, and environment used to produce the evaluation results reported in the paper.

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.11 / 3.12 |
| OS | macOS (Darwin) / Linux |
| LLM Provider | OpenAI (ChatGPT API) |
| LLM Model | `gpt-4o-mini` |
| Z3 Solver | 4.12.2.0 |
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

# 5. (macOS x86_64 only) ensure Z3 binary architecture matches Python
uv pip install "z3-solver==4.12.2.0"
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

### Step 2b: Embedding Ablation (RAG baselines)

```bash
# Single run for each embedding model + merged summary artifacts
python eval/external_benchmarks/run_embedding_ablation.py \
  --llm-provider chatgpt --llm-model gpt-4o-mini \
  --output-dir results/embedding_ablation

# Direct single-run variant (if needed)
python -m eval.runner --embedding-model all-MiniLM-L6-v2 \
  --llm-provider chatgpt --llm-model gpt-4o-mini --require-api \
  --output results/embedding_ablation/all_MiniLM_L6_v2.json
python -m eval.runner --embedding-model BAAI/bge-large-en-v1.5 \
  --llm-provider chatgpt --llm-model gpt-4o-mini --require-api \
  --output results/embedding_ablation/BAAI_bge_large_en_v1.5.json
python -m eval.runner --embedding-model text-embedding-3-small \
  --llm-provider chatgpt --llm-model gpt-4o-mini --require-api \
  --output results/embedding_ablation/text_embedding_3_small.json
```

### Step 2c: External Benchmark Pilot Runs

All external benchmark scripts default to fixed seed `42` and persist sampled IDs.
External adapters support two explicit protocols:
- `--response-mode asserted_probe`: fixed assertion response, used as a bundle-consistency probe.
- `--response-mode generated_response`: model-generated response followed by enforcement.

```bash
# ContractNLI (10 dev + 50 test; calibrated frozen thresholds)
python eval/external_benchmarks/contract_nli_adapter.py \
  --dataset-config contractnli_a \
  --seed 42 --dev-examples 10 --max-examples 50 \
  --response-mode asserted_probe \
  --baseline-max-chars 0 \
  --llm-provider chatgpt --model gpt-4o-mini \
  --config Extractor/configs/config.chatgpt.yaml \
  --output results/external/contract_nli_results.json

# LegalBench: unfair_tos (category->binary mapping: Other=>not_unfair, all specific categories=>unfair)
python eval/external_benchmarks/legalbench_adapter.py \
  --task unfair_tos --seed 42 --dev-examples 10 --max-examples 50 \
  --response-mode asserted_probe \
  --baseline-max-chars 0 \
  --unfair-tos-bundle-mode extracted_reference \
  --llm-provider chatgpt --model gpt-4o-mini \
  --config Extractor/configs/config.chatgpt.yaml \
  --output results/external/legalbench_unfair_tos_results.json

# LegalBench: privacy_policy_entailment
python eval/external_benchmarks/legalbench_adapter.py \
  --task privacy_policy_entailment --seed 42 --dev-examples 10 --max-examples 50 \
  --response-mode asserted_probe \
  --baseline-max-chars 0 \
  --llm-provider chatgpt --model gpt-4o-mini \
  --config Extractor/configs/config.chatgpt.yaml \
  --output results/external/legalbench_privacy_policy_entailment_results.json

# CUAD (10 contracts, deterministic random selection)
python eval/external_benchmarks/cuad_adapter.py \
  --dataset-name theatticusproject/cuad-qa \
  --seed 42 --num-contracts 10 \
  --selection-strategy random \
  --max-policyllm-chars 30000 --chunk-overlap-chars 1000 \
  --llm-provider chatgpt --model gpt-4o-mini \
  --config Extractor/configs/config.chatgpt.yaml \
  --output results/external/cuad_results.json

# Orchestrated run across all external adapters + summary artifacts
python eval/external_benchmarks/run_all.py \
  --seed 42 --dev-examples 10 --max-examples 50 --num-contracts 10 \
  --response-mode asserted_probe \
  --baseline-max-chars 0 \
  --unfair-tos-bundle-mode extracted_reference \
  --cuad-selection-strategy random \
  --max-policyllm-chars 30000 --cuad-chunk-overlap-chars 1000 \
  --llm-provider chatgpt --model gpt-4o-mini \
  --config Extractor/configs/config.chatgpt.yaml \
  --output-dir results/external
```

### Optional Offline Smoke Runs (No API Spend)

```bash
# External benchmark smoke suite with deterministic tiny samples
python eval/external_benchmarks/run_all.py \
  --seed 42 --dev-examples 2 --max-examples 3 --num-contracts 3 \
  --response-mode asserted_probe \
  --baseline-max-chars 0 \
  --unfair-tos-bundle-mode extracted_reference \
  --cuad-selection-strategy random \
  --max-policyllm-chars 12000 --cuad-chunk-overlap-chars 500 \
  --llm-provider stub --model stub \
  --config Extractor/configs/config.stub.yaml \
  --output-dir results/external/dryrun_suite

# Embedding ablation smoke for sentence-transformer models only
python eval/external_benchmarks/run_embedding_ablation.py \
  --embedding-models all-MiniLM-L6-v2 BAAI/bge-large-en-v1.5 \
  --llm-provider stub --llm-model stub \
  --output-dir results/embedding_ablation/dryrun_suite
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
| `results/embedding_ablation/summary.{json,csv,md}` | Embedding ablation summaries |
| `results/external/summary.{json,csv,md}` | Aggregated external benchmark summaries |
| `results/external/*_split_ids.json` | Persisted dev/test sampled IDs (seeded) |

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
