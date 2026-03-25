# PolicyLLM — System Design

## Overview

PolicyLLM is a neuro-symbolic framework for extracting, validating, and enforcing organizational policies on LLM outputs at runtime. It bridges the gap between unstructured policy documents and formally verifiable enforcement by combining multi-pass LLM extraction with Z3-based symbolic verification.

## Architecture

```
┌─────────────────────────┐
│    Policy Documents      │  PDF, DOCX, HTML, Markdown, TXT
└───────────┬─────────────┘
            │
    ┌───────▼────────┐
    │  Regularization │  PyMuPDF / python-docx / BeautifulSoup
    │  (Extractor)    │  → CanonicalDocument {sections, paragraphs, spans}
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  6-Pass LLM     │  Pass 1: Classify → Pass 2: Components → Pass 3: Entities
    │  Extraction     │  Pass 4: Merge → Pass 5: Metadata → Pass 6: Validate
    │  (Extractor)    │  → policies.jsonl
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Policy IR      │  policy_ir_builder.py: policies → {variables, rules, constraints}
    │  Builder        │
    │  (Validation)   │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Decision Graph │  decision_graph.py: ordered decision nodes, compiled paths,
    │  Compiler       │  leaf actions, variable schema
    │  (Validation)   │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Z3 Conflict    │  conflict_detector.py: SAT-based pairwise conflict detection
    │  Detection      │  → conflict report (logical + semantic)
    │  (Validation)   │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Priority       │  resolution.py: priority lattice → auto-resolve or escalate
    │  Resolution     │  → dominance rules, escalation contacts
    │  (Validation)   │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Bundle         │  bundle_compiler.py: merge all outputs →
    │  Compiler       │  compiled_policy_bundle.json
    │  (Validation)   │
    └───────┬────────┘
            │
            ▼  ═══════════  RUNTIME  ═══════════
            │
    ┌───────▼────────┐
    │  Pre-Generation │  Query classification (LLM) → rule retrieval (domain index)
    │  (Enforcement)  │  → dominance resolution → EnforcementContext
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  During-Gen     │  Scaffold injection: ordered reasoning steps + invariant constraints
    │  (Enforcement)  │  + priority guidance → InjectionBundle → formatted prompt
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  LLM Generation │  Temperature 0.0, max_tokens 2048
    │                 │  Provider-agnostic: OpenAI / Anthropic / Bedrock / Ollama
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Post-Gen       │  4 parallel checkers:
    │  Verification   │    1) Regex: PII patterns, forbidden language (hard gate)
    │  (Enforcement)  │    2) SMT:  Z3 fact extraction + rule verification
    │                 │    3) Judge: LLM-as-judge semantic compliance scoring
    │                 │    4) Coverage: decision-node mention coverage
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Compliance     │  S = 0.60×SMT + 0.30×Judge + 0.10×Coverage
    │  Scoring        │  Regex = hard gate (failure → ESCALATE always)
    │  (Enforcement)  │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Action Routing │  PASS (≥0.95) │ AUTO_CORRECT (0.85-0.95)
    │                 │  REGENERATE (0.70-0.85) │ ESCALATE (<0.70 or PII)
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Audit Logger   │  JSONL with SHA-256 hash chain for tamper detection
    └─────────────────┘
```

## Module Descriptions

### Extractor (Stage 1)

| File | Purpose |
|------|---------|
| `src/pipeline.py` | Orchestrates regularization + 6-pass extraction |
| `src/regularize/router.py` | Detects file type, dispatches to PDF/DOCX/HTML/MD handler |
| `src/regularize/pdf_native.py` | PyMuPDF-based PDF parser with heading detection |
| `src/llm/client.py` | Unified LLM client (OpenAI, Anthropic, Bedrock, Ollama, Stub) |
| `src/passes/pass1_classify.py` | Is this section a policy? (binary + confidence) |
| `src/passes/pass2_components.py` | Extract scope, conditions, actions, exceptions |
| `src/passes/pass3_entities.py` | Named entity extraction (people, orgs, regulations) |
| `src/passes/pass4_merge.py` | Deduplicate via similarity threshold |
| `src/passes/pass5_metadata.py` | Infer owner, domain, effective date, regulatory linkage |
| `src/passes/pass6_validate.py` | Self-consistency check, flag low-confidence policies |
| `configs/` | YAML configs per provider (chatgpt, ollama, stub) |

### Validation (Stage 2)

| File | Purpose |
|------|---------|
| `policy_ir_builder.py` | Policies → IR (variables, conditional rules, constraints) |
| `decision_graph.py` | IR → ordered decision graph with compiled paths |
| `conflict_detector.py` | Z3 SAT-based pairwise conflict detection |
| `resolution.py` | Priority-lattice resolution (auto-resolve / escalate) |
| `bundle_compiler.py` | Merge all Validation outputs → `compiled_policy_bundle.json` |

### Enforcement (Stage 3)

| File | Purpose |
|------|---------|
| `orchestrator.py` | Full enforcement pipeline (pregen → duringgen → postgen → action routing) |
| `pregen.py` | Query classification, rule retrieval, dominance resolution |
| `duringgen.py` | Scaffold serialization, prompt injection, constraint formatting |
| `postgen/smt.py` | Z3 fact extraction + rule verification |
| `postgen/judge.py` | LLM-as-judge semantic compliance scoring |
| `postgen/regex.py` | PII / forbidden-language pattern matching |
| `scoring.py` | Weighted compliance score + action thresholds |
| `bundle_loader.py` | Load + validate bundle, build in-memory indexes |
| `schemas.py` | All Pydantic models (single source of truth) |
| `audit.py` | JSONL audit log with SHA-256 hash chain |

### Evaluation

| File | Purpose |
|------|---------|
| `eval/metrics.py` | Precision, recall, F1, compliance rate, FP rate |
| `eval/baselines.py` | 9 baseline implementations (ablations + alternatives) |
| `eval/runner.py` | Orchestrates running all methods, computes all metrics |
| `eval/reference_data/` | Ground truth annotations for extraction + enforcement |

## Key Design Decisions

1. **Neuro-symbolic split**: LLM does fuzzy extraction and judging; Z3 does deterministic verification. Neither alone is sufficient.
2. **Regex as hard gate**: PII patterns bypass the score entirely — safety cannot be overridden by high compliance scores.
3. **Scaffold injection**: Rather than hoping the LLM "remembers" policies, we inject ordered reasoning steps that the model must follow.
4. **Priority lattice**: Regulatory > core values > company > department > situational. Conflicts at the same level are escalated; different levels are auto-resolved.
5. **Hash-chain audit**: Every enforcement decision is logged with a SHA-256 chain for tamper detection.

## Data Flow

```
Input:   PDF / DOCX / HTML / MD / TXT
  ↓
Extract: policies.jsonl  (per section: policy_id, conditions, actions, metadata)
  ↓
Validate: compiled_policy_bundle.json  (variables, rules, constraints, paths, dominance)
  ↓
Enforce: ComplianceDecision  (score, action, violations, evidence, audit_trail)
```
