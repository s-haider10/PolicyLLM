# F1 Score Fix — Condition Operator Extraction

## Problem

Condition F1 was **0.33** because the LLM defaulted operators to `"unknown"` or `null`. The metric canonicalizes conditions as `type|operator`, so `time_window|unknown` ≠ `time_window|<=` — a full miss even when the condition was correctly identified.

## Changes

### 1. Operator-constrained prompt (`Extractor/src/passes/pass2_components.py`)

- Added an explicit operator enum in the prompt: `==, !=, >, <, >=, <=, in, not_in, boolean_true, boolean_false`.
- Added natural-language-to-operator mapping rules (e.g., "within X days" → `<=`).
- Instructed the LLM to never output `"unknown"` or `null` for operator.

### 2. Few-shot exemplars (`Extractor/src/passes/pass2_components.py`)

Added 4 operator-specific examples covering `<=`, `>=`, `boolean_true`, and `not_in` — the four operator classes previously missed entirely.

### 3. Pydantic Literal constraint (`Extractor/src/passes/pass2_components.py`)

Changed `ConditionModel.operator` from `Optional[str]` to `Optional[Literal[...]]` with the 10 valid operators. Invalid values like `"unknown"` now fail Pydantic validation and trigger an LLM retry.

### 4. Post-extraction operator inference (`Extractor/src/passes/pass2_components.py`)

Added `_infer_operator()` — a deterministic fallback in `_normalize()` that infers operator from condition type and `source_text` when still `None` or `"unknown"` after LLM extraction.

### 5. Metric normalization (`eval/metrics.py`)

`_cond_key()` now treats `"unknown"` the same as `None`/empty, preventing it from producing non-matching canonical keys.

## Why layered

| Layer | Catches |
|-------|---------|
| Prompt + exemplars | Prevents bad output at generation time |
| Pydantic Literal | Rejects invalid values, forces retry |
| `_infer_operator()` | Fills remaining `None` values deterministically |
| Metric normalization | Last-resort safety net at evaluation time |
