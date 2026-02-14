# PolicyLLM System Design

## Overview

PolicyLLM is a neuro-symbolic framework that extracts structured policies from documents and enforces them on LLM outputs at runtime. The system combines neural extraction (LLM-based) with symbolic verification (Z3 SMT solver) to ensure compliance.

**Pipeline:** `Document → Extraction → Validation → Enforcement → ComplianceDecision`

**Key Artifact:** `compiled_policy_bundle.json` — the contract between offline policy compilation (Validation) and online enforcement.

---

## Architecture

```
┌─────────────────┐
│ Policy Document │  (PDF, DOCX, HTML, MD, TXT)
└────────┬────────┘
         │
    ┌────▼────┐
    │Extract  │  6-pass LLM pipeline → policies.jsonl
    │(Neural) │  classify → components → entities → merge → metadata → validate
    └────┬────┘
         │
    ┌────▼────┐
    │Validate │  IR builder → decision graph (DAG) → Z3 conflict detection
    │(Hybrid) │  → priority resolution → compiled_policy_bundle.json
    └────┬────┘
         │
    ┌────▼────┐
    │Enforce  │  pregen (classify + retrieve) → duringgen (scaffold injection)
    │(Hybrid) │  → LLM generation → postgen (regex + SMT + judge + coverage)
    └────┬────┘
         │
    ┌────▼────┐
    │Decision │  PASS | AUTO_CORRECT | REGENERATE | ESCALATE
    └─────────┘
```

---

## Stage 1: Extraction (Neural)

**Input:** Unstructured policy documents
**Output:** `policies.jsonl` (structured policy objects)

### 6-Pass Pipeline

1. **Classify:** Distinguish policy sections from boilerplate (LLM + confidence scoring)
2. **Components:** Extract scope, conditions, actions, exceptions (structured LLM prompting)
3. **Entities:** Extract dates, amounts, roles via regex + spaCy + LLM
4. **Merge:** Deduplicate similar policies using embedding similarity (threshold: 0.9)
5. **Metadata:** Resolve source, owner, domain, priority, regulatory linkage (regex + LLM)
6. **Validate:** Structural and semantic checks + LLM critique (confidence threshold: 0.7)

**Design Principles:**
- **Regularization-first:** All document formats normalized to canonical JSON before extraction
- **Multi-pass over single-pass:** Focused prompts per task reduce LLM hallucination
- **Provider abstraction:** Supports Ollama, OpenAI, AWS Bedrock, Anthropic

**Policy Schema:**
```json
{
  "policy_id": "POL-REFUND-001",
  "conditions": [{"type": "time_window", "value": 30, "unit": "days", "operator": "<="}],
  "actions": [{"type": "required", "action": "full_refund"}],
  "metadata": {"domain": "refund", "priority": "company"}
}
```

---

## Stage 2: Validation (Symbolic + Hybrid)

**Input:** `policies.jsonl`
**Output:** `compiled_policy_bundle.json` (enforcement-ready bundle)

### Pipeline

```
policies.jsonl
    ↓
Policy IR Builder  ← transforms rich policy schema to flat IR (variables, rules, constraints)
    ↓
Decision Graph Builder  ← compiles rules into DAG with ordered decision nodes and paths
    ↓
Z3 Conflict Detector  ← pairwise SAT checks for logical conflicts
    ↓
Priority Resolution  ← uses priority lattice (regulatory > company > dept)
    ↓
Bundle Compiler  ← outputs compiled_policy_bundle.json
```

### Decision Graph (DAG)

The decision graph represents all valid policy execution paths:
- **Decision Nodes:** Topologically ordered variables (bool → enum → numeric)
- **Compiled Paths:** Each path = sequence of (variable, tests) → leaf action
- **Runtime Usage:** Queries outside defined paths receive low compliance scores (§3.6)

### Z3 Conflict Detection

For each rule pair (i, j):
1. Create Z3 variables from shared schema
2. Assert: `conditions_i AND conditions_j`
3. If SAT and actions differ → conflict detected
4. Record Z3 witness (concrete variable values triggering conflict)

**Priority Lattice:** `regulatory (1) > core_values (2) > company (3) > department (4) > situational (5)`

**Bundle Schema:**
```json
{
  "variables": {"has_receipt": {"type": "bool"}},
  "conditional_rules": [{"policy_id", "conditions", "action"}],
  "constraints": [{"policy_id", "constraint", "scope"}],
  "decision_nodes": ["has_receipt", "product_category", "days_since_purchase"],
  "compiled_paths": [{"path": [{"var": "has_receipt", "tests": ["== true"]}], "leaf_action": "refund:full"}],
  "dominance_rules": [{"enforce": ["POL-1"], "suppress": ["POL-2"]}]
}
```

---

## Stage 3: Enforcement (Hybrid)

**Input:** User query + `compiled_policy_bundle.json`
**Output:** `ComplianceDecision` (score, action, violations, LLM response)

### Pipeline

```
User Query
    ↓
Pre-Gen: classify query → retrieve applicable rules → apply dominance
    ↓
During-Gen: serialize constraints + scaffold → inject into LLM prompt
    ↓
Generate: LLM call (temperature=0)
    ↓
Post-Gen: regex + SMT + coverage + judge (parallel)
    ↓
Scoring: S = 0.55*SMT + 0.25*Judge + 0.10*Regex + 0.10*Coverage
    ↓
Routing: PASS (≥0.95) | AUTO_CORRECT (≥0.85) | REGENERATE (≥0.70) | ESCALATE (<0.70)
```

### Pre-Generation

**Query Classification:**
1. Keyword matching (deterministic, fast)
2. LLM fallback if confidence < 0.6

**Rule Retrieval:**
- Filter rules by domain + temporal validity
- Apply dominance rules from bundle

### During-Generation (Scaffold Injection)

Transform decision graph into step-by-step LLM instructions:

```
STEP 1: Check variable has_receipt. If unknown, ask user.
STEP 2: Determine product_category. Must be one of: electronics, clothing.
STEP 3: If product_category == 'electronics' AND days_since_purchase <= 15
         THEN ACTION => refund:full (per POL-REFUND-001)
```

Invariant constraints injected into system prompt:
```
INVARIANTS:
- NEVER disclose PII (SSN, email, credit card)
- NEVER make unconditional promises
```

### Post-Generation Verification

Four independent checks:

1. **Regex (0.10):** Pattern matching for forbidden content (PII, over-promising)
2. **SMT (0.55):** Extract facts from response → verify against Z3-encoded rules
3. **Judge (0.25):** LLM evaluates factual accuracy, action compliance, tone
4. **Coverage (0.10):** Fraction of required decision nodes mentioned in response

**DAG Path Verification:** Responses with facts outside any compiled path receive score penalty (0.5)

### Compliance Scoring

```
S = 0.55 * SMT + 0.25 * Judge + 0.10 * Regex + 0.10 * Coverage
```

| Score | Action | Behavior |
|-------|--------|----------|
| ≥0.95 | PASS | Deliver response |
| 0.85-0.95 | AUTO_CORRECT | Append hints, retry 1x |
| 0.70-0.85 | REGENERATE | Add "DO NOT" directives, retry 2x |
| <0.70 | ESCALATE | Block response |

**Override:** PII detected → ESCALATE always

---

## Key Innovations

1. **Neuro-Symbolic Hybrid:** Neural extraction + symbolic verification reduces LLM hallucination in policy-critical paths
2. **DAG-Based Enforcement:** Decision graph ensures responses follow defined policy paths (§3.6)
3. **Z3 Conflict Detection:** Formal methods catch logical contradictions missed by LLMs
4. **Scaffold Injection:** Chain-of-thought prompting guides LLM through decision nodes
5. **Multi-Stage Verification:** Parallel postgen checks (regex, SMT, judge, coverage) with weighted scoring
6. **Priority Lattice:** Explicit conflict resolution preserving regulatory dominance

---

## Determinism Guarantees

- **LLM calls:** temperature=0 everywhere
- **Scaffold generation:** Variables in decision_nodes order, paths in policy_id alphabetical order
- **Z3 solving:** Deterministic SAT solver
- **Scoring:** Fixed weights (0.55, 0.25, 0.10, 0.10)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Schema validation | Pydantic v2 |
| SMT solver | Z3 (z3-solver) |
| Embeddings | sentence-transformers |
| Document parsing | PyMuPDF, python-docx, BeautifulSoup |
| LLM abstraction | Custom LLMClient (Ollama, OpenAI, Bedrock, Anthropic) |
| Testing | pytest (119 Enforcement tests + Extraction tests) |
| Environment | uv (fast Python package management) |

---

## Data Flow Contracts

**Extractor → Validation:**
- Format: JSONL (one policy per line)
- Schema: Pydantic `Policy` model
- Bridge: `policy_ir_builder.py` transforms rich schema to flat IR

**Validation → Enforcement:**
- Format: Single JSON file (`compiled_policy_bundle.json`)
- Schema: `CompiledPolicyBundle` Pydantic model
- Contract: Self-contained (zero runtime dependency on upstream modules)

**Enforcement → Consumer:**
- Format: `ComplianceDecision` Pydantic model
- Fields: score, action, violations, evidence, audit_trail, llm_response
- Serializable: `decision.model_dump()` → JSON
