# PolicyLLM System Design

## Overview

PolicyLLM is a three-stage pipeline that transforms unstructured policy documents into machine-enforceable rules, then uses those rules to govern LLM outputs at runtime.

```
Document → [Extractor] → policies.jsonl → [Validation] → compiled_policy_bundle.json → [Enforcement] → ComplianceDecision
```

The system is designed with a hard contract boundary: `compiled_policy_bundle.json` is the single artifact that flows between Validation (offline, build-time) and Enforcement (online, per-request). Enforcement has zero runtime dependency on upstream modules.

---

## Stage 1: Extractor (Group 1)

**Purpose:** Extract structured policy objects from heterogeneous documents.

### Input/Output
- **Input:** PDF, DOCX, HTML, Markdown, or plain text documents
- **Output:** `policies.jsonl` (one policy per line) + `index.json` (summary metadata)

### Architecture

The Extractor runs a 6-pass pipeline over regularized document sections:

```
Document → Regularizer → Canonical JSON (sections/paragraphs/tables/spans)
                              ↓
                    Pass 1: Classify (policy vs non-policy, confidence score)
                              ↓
                    Pass 2: Components (scope, conditions, actions, exceptions)
                              ↓
                    Pass 3: Entities (dates, amounts, roles via regex + spaCy + LLM)
                              ↓
                    Pass 4: Merge (embedding-based deduplication, evidence aggregation)
                              ↓
                    Pass 5: Metadata (source, owner, domain, priority, regulatory linkage)
                              ↓
                    Pass 6: Validate (rule checks + LLM critique, confidence scoring)
                              ↓
                          policies.jsonl
```

### Key Design Decisions
- **Regularization first:** All document formats are normalized into a canonical JSON structure (sections → paragraphs → spans) before extraction. This lets passes operate format-agnostically.
- **Multi-pass over single-pass:** Each pass is focused and testable. LLM prompts are smaller and more reliable than a single massive extraction prompt.
- **Double-run consensus:** Optional mode that runs selected passes twice with different seeds and merges results to reduce LLM variance.
- **Provider abstraction:** `LLMClient` supports Ollama (local), Bedrock Claude, OpenAI, and Anthropic. Default is local Ollama for cost-free development.

### Policy Schema (Output)

```json
{
  "policy_id": "POL-REFUND-001",
  "conditions": [
    {"type": "time_window", "value": 30, "unit": "days", "operator": "<=", "target": "general"},
    {"type": "boolean_flag", "value": true, "parameter": "has_receipt"}
  ],
  "actions": [
    {"type": "required", "action": "full_refund", "requires": ["has_receipt", "within_window"]},
    {"type": "prohibited", "action": "disclose_pii"}
  ],
  "metadata": {
    "source": "refund_policy_2024.pdf",
    "domain": "refund",
    "priority": "company",
    "owner": "Customer Service Dept",
    "regulatory_linkage": ["FTC Cooling-Off Rule"]
  }
}
```

---

## Stage 2: Validation (Group 2)

**Purpose:** Formalize extracted policies into a decision graph, detect conflicts, resolve them, and compile everything into a single enforcement bundle.

### Input/Output
- **Input:** `policies.jsonl` (from Extractor)
- **Output:** `compiled_policy_bundle.json` (the contract artifact)

### Architecture

```
policies.jsonl
      ↓
Policy IR Builder ← builds the missing bridge between Extractor output and IR format
      ↓
policy_ir: {variables, conditional_rules, constraints, metadata}
      ↓
Decision Graph Builder ← compiles rules into ordered decision paths
      ↓
decision_graph: {nodes, edges, leaf_actions, compiled_paths}
      ↓
Z3 Conflict Detector ← pairwise Z3 SAT checks for logical conflicts
      ↓
conflict_report: [{pair, witness, severity}]
      ↓
Priority Resolution ← uses priority lattice to resolve conflicts
      ↓
resolution_report: {dominance_rules, escalations, unresolved}
      ↓
Bundle Compiler ← merges all artifacts into compiled_policy_bundle.json
```

### Policy IR Builder (The Bridge)

The critical component that transforms Extractor's rich policy schema into the flat IR format:

| Extractor Field | IR Mapping |
|----------------|------------|
| `condition.parameter == "has_receipt"` | `variable: {name: "has_receipt", type: "bool"}` |
| `condition.type == "time_window"` | `variable: {name: "days_since_purchase", type: "int"}` |
| `condition.type == "amount_threshold"` | `variable: {name: "refund_amount", type: "float"}` |
| `condition.type == "product_category"` | `variable: {name: "product_category", type: "enum"}` |
| `action.type == "prohibited"` | `constraint: "NOT(action.action)"` |

### Z3 Conflict Detection

For each pair of rules (i, j):
1. Create Z3 variables from the shared variable schema
2. Assert conditions of rule i AND conditions of rule j
3. If `sat` → the rules can fire simultaneously
4. If their actions differ → record a conflict with the Z3 witness (concrete values)

### Priority Lattice

```
regulatory (1) > core_values (2) > company (3) > department (4) > situational (5)
```

When two conflicting rules have different priorities, the higher-priority rule dominates. Same priority → escalation entry requiring human resolution.

### compiled_policy_bundle.json Schema

```json
{
  "schema_version": "1.0",
  "variables": {"var_name": {"type": "bool|int|float|enum", "values": null|[...]}},
  "conditional_rules": [{"policy_id", "conditions", "action", "metadata"}],
  "constraints": [{"policy_id", "constraint", "scope"}],
  "decision_nodes": ["ordered variable names"],
  "compiled_paths": [{"policy_id", "path": [{"var", "tests"}], "leaf_action"}],
  "dominance_rules": [{"when": {"policies_fire": [...]}, "then": {"mode", "enforce"}}],
  "escalations": [{"conflict_type", "policies", "owners_to_notify"}],
  "priority_lattice": {"regulatory": 1, "core_values": 2, ...},
  "bundle_metadata": {"policy_count", "rule_count", "constraint_count", "path_count"}
}
```

---

## Stage 3: Enforcement (Group 3)

**Purpose:** At runtime, take a user query + the compiled bundle and produce a compliance-verified LLM response.

### Input/Output
- **Input:** User query + `compiled_policy_bundle.json`
- **Output:** `ComplianceDecision` (score, action, violations, evidence, LLM response)

### Architecture

```
User Query
    ↓
┌───────────┐
│  Pre-Gen   │  classify_query → retrieve_rules → apply_dominance → EnforcementContext
└─────┬─────┘
      ↓
┌───────────┐
│ During-Gen │  serialize_constraints → serialize_scaffold → format_full_prompt
└─────┬─────┘
      ↓
┌───────────┐
│  Generate  │  LLM call with injected scaffold (temperature=0)
└─────┬─────┘
      ↓
┌───────────┐
│  Post-Gen  │  regex → SMT → coverage → judge (parallel where possible)
└─────┬─────┘
      ↓
┌───────────┐
│  Scoring   │  S = 0.55*Z + 0.25*L + 0.10*R + 0.10*C → action routing
└─────┬─────┘
      ↓
┌───────────┐
│  Routing   │  PASS | AUTO_CORRECT (retry 1x) | REGENERATE (retry 2x) | ESCALATE
└─────┬─────┘
      ↓
┌───────────┐
│   Audit    │  JSONL append with SHA256 hash chain
└───────────┘
```

### Pre-Generation

**Query Classification:** Two-tier approach:
1. Keyword matching against `DOMAIN_KEYWORDS` and `INTENT_KEYWORDS` dictionaries (deterministic, fast)
2. LLM fallback if keyword confidence < 0.6 (temperature=0 for determinism)

**Rule Retrieval:**
- Filter `rules_by_domain[domain]` from the bundle index
- Apply temporal filtering: `eff_date <= today` or `eff_date is null`
- Merge "always" + domain-scoped constraints

**Dominance Resolution:**
- For each rule pair: check `dominance_lookup[frozenset(pids)]`
- Override mode → remove the losing rule
- No dominance rule → compare via priority lattice
- Same priority → flag for escalation

### During-Generation

The scaffold injector transforms the enforcement context into LLM prompt instructions:

```
STEP 1: Check variable has_receipt. If unknown, ask the user; DO NOT assume.
STEP 2: Determine product_category. Must be one of: electronics, clothing, other.
STEP 3: If product_category == 'electronics' AND days_since_purchase <= 15
         THEN ACTION => refund:full (per electronics_refund_v2, source: refund_policy_2024.pdf)
STEP 4: FINAL — State the action and cite the policy source.
```

Variables are processed in `decision_nodes` order (bool first, then enum, then numeric). Paths are processed in policy_id alphabetical order. This ensures deterministic scaffolds.

Invariant constraints are injected into the system prompt:
```
- INVARIANTS:
  1) NEVER disclose pii.
  2) NEVER promise before verified.
```

### Post-Generation Verification

Four independent checks run on the LLM response:

#### Regex Check (Weight: 0.10)
- Pattern matching for forbidden content: SSN, email, credit card, password disclosure, over-promising
- Constraint-derived patterns: `NOT(X)` → case-insensitive search for X
- Binary: passed=true → 1.0, any match → 0.0
- PII match → automatic ESCALATE override

#### SMT Check (Weight: 0.55)
- Extract facts from response via regex patterns (LLM fallback if < 50% extracted)
- Build Z3 variables from bundle schema
- Assert extracted facts as equalities
- For each rule: encode conditions, check satisfiability
- For each constraint: check violation
- Binary: no violations → 1.0, any violation → 0.0

#### Judge LLM Check (Weight: 0.25)
- Separate LLM call evaluating factual accuracy, action compliance, constraint adherence, tone, completeness
- Returns 0.0-1.0 continuous score
- Temperature=0 for deterministic scoring
- Failure fallback: score=0.5 (neutral)

#### Coverage Check (Weight: 0.10)
- Measures what fraction of required decision nodes are mentioned in the response
- Score = nodes_covered / nodes_required
- Uses variable name and readable name (underscore → space) matching

### Compliance Scoring

```
S = 0.55 * SMT + 0.25 * Judge + 0.10 * Regex + 0.10 * Coverage
```

| Score Range | Action | Behavior |
|------------|--------|----------|
| >= 0.95 | PASS | Deliver response as-is |
| 0.85 - 0.95 | AUTO_CORRECT | Append violation hints, re-generate once |
| 0.70 - 0.85 | REGENERATE | Add "DO NOT" directives, re-generate up to 2x |
| < 0.70 | ESCALATE | Block response, notify policy owners |

**Override:** Regex failure (PII detected) → ESCALATE always, regardless of score.

### Action Routing (Retry Loop)

```
while action in (AUTO_CORRECT, REGENERATE):
    if AUTO_CORRECT and retries < 1:
        append violation hints to prompt
        re-generate → re-score
        if score >= 0.95: return PASS
    elif REGENERATE and retries < 2:
        append "DO NOT" directives
        re-generate → re-score
        if action == PASS: return
    else:
        action = ESCALATE
        break
```

### Audit Logging

Append-only JSONL with SHA256 hash chain for tamper detection:

```json
{
  "entry_hash": "sha256...",
  "prev_hash": "sha256... or null",
  "session_id": "uuid",
  "query": "...",
  "domain": "refund",
  "compliance_score": 0.97,
  "final_action": "pass",
  "duration_ms": 1234
}
```

Hash chain: `hash(prev_hash + canonical_json) → entry_hash`. Verify by replaying the chain.

---

## Bundle Index (O(1) Lookups)

At load time, `BundleIndex` builds in-memory indexes:

| Index | Key | Value |
|-------|-----|-------|
| `rules_by_domain` | domain string | List[ConditionalRule] |
| `rules_by_policy_id` | policy_id | ConditionalRule |
| `paths_by_domain` | domain string | List[CompiledPath] |
| `constraints_by_scope` | "always" or domain | List[Constraint] |
| `dominance_lookup` | frozenset({pid1, pid2}) | DominanceRule |
| `escalation_lookup` | frozenset(policy_ids) | EscalationEntry |

---

## Determinism Guarantees

Every component is designed for deterministic output given the same input:

- **Keyword classification:** Dictionary-based, no randomness
- **LLM calls:** temperature=0 everywhere
- **Scaffold generation:** Variables in `decision_nodes` order, paths in policy_id alphabetical order
- **Z3 solving:** Deterministic SAT solver
- **Scoring:** Pure arithmetic with fixed weights
- **Pair processing:** Sorted iteration over policy_id pairs

---

## Error Handling Strategy

| Component | Failure Mode | Behavior |
|-----------|-------------|----------|
| Regex check | Exception | `RegexResult(passed=True, score=1.0)` — conservative, don't block |
| SMT check | Z3 error | Re-raise — Z3 is a hard dependency |
| Judge LLM | LLM unavailable | `JudgeResult(score=0.5)` — neutral score |
| Generation | LLM timeout | Return empty string, proceed to postgen |
| Bundle load | Schema mismatch | Raise `ValidationError` — fail fast |

---

## Data Flow Contracts

### Extractor → Validation
- **Format:** JSONL, one policy per line
- **Schema:** Pydantic `Policy` model with conditions, actions, metadata
- **Bridge:** `policy_ir_builder.py` transforms the rich schema into flat IR

### Validation → Enforcement
- **Format:** Single JSON file (`compiled_policy_bundle.json`)
- **Schema:** `CompiledPolicyBundle` Pydantic model
- **Contract:** Self-contained — Enforcement needs nothing except this file + an LLM client
- **Integrity:** `validate_bundle_integrity()` checks referential integrity at load time

### Enforcement → Consumer
- **Format:** `ComplianceDecision` Pydantic model
- **Fields:** score (0.0-1.0), action (PASS|AUTO_CORRECT|REGENERATE|ESCALATE), violations, evidence, audit_trail, llm_response
- **Serializable:** `decision.model_dump()` → JSON-safe dict

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Schema validation | Pydantic v2 | Type-safe models across all modules |
| SMT solver | Z3 (z3-solver) | Formal verification + conflict detection |
| Embeddings | sentence-transformers | Action clustering in schema discovery |
| Document parsing | PyMuPDF, python-docx, BeautifulSoup | Multi-format regularization |
| LLM abstraction | Custom LLMClient | Ollama, Bedrock, OpenAI, Anthropic |
| Config | PyYAML | Extractor configuration |
| Testing | pytest | 119 Enforcement tests + Extractor unit tests |
| Environment | uv | Fast Python package management |
