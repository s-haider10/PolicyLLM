# End-to-end plan — **Group 3: Enforcement Lead** (uses Association_for_Computational_Linguistics**ACL**conference.pdf)

Reference used: extraction + architecture details in the draft paper above.

Below is a production-grade, **complete** implementation plan for Group 3 (Enforcement Lead). It maps precisely to the Tracks and Tasks in your spec, lists **every** module, interface, data schema, verification, test, monitoring, and handoff to/from Group 2, and gives concrete examples (scaffold prompt + SMT checks + compliance score formula). Use this as your team’s engineering checklist — nothing is left out.

---

## 1) High-level responsibilities (what Group 3 _must_ deliver)

1. Integration Parser: translate Group-2 outputs (`processed` policies, pseudocode, `decision_graph`, `resolution_report`, `conflict_report`) into runtime IR (`policy_ir`) used by the runtime engine and Z3.
2. Pre-Gen: Query classifier + DPR retrieval + permission-aware policy selection + dominance resolution.
3. During-Gen (Scaffolding): Serialize decision graphs into deterministic chain-of-thought scaffolds + inject invariant constraints & priority guidance.
4. Post-Gen: Multi-stage verification — fast filters, SMT verification (Z3), Judge LLM (semantic check), compute Compliance Score, decide action (pass/auto-correct/regenerate/escalate).
5. Orchestration & Performance: pipeline orchestration, caching, parallelization to meet latency budget (target: 201ms median for end-to-end enforcement overhead).
6. Observability, Auditing, RBAC, UI for escalations & owners, tests, and rollout controls.

---

## 2) Inputs from Group 2 (explicit list & how to consume them)

Use these exact artifacts produced by Group 2:

- `processed` policies (schema_discovery output) — contains extracted `scope`, `conditions`, `actions`, `exceptions`, `metadata`. **Input to Integration Parser.**
- `pseudocode` outputs / `policy_to_pseudocode(...)` results — useful for human-readable crosschecks and fallback scaffolds.
- `decision_graph["compiled_paths"]` — **primary** asset for During-Gen scaffolds (ordered decision nodes + leaf action + metadata).
- `conflict_report` (from Z3 runs) — used to populate escalations and build dominance/dominance_rules training sets.
- `resolution_report["conflict_free_plan"]["dominance_rules"]` — authoritative dominance rules for Pre-Gen conflict resolution.
- `z3_var`, `encode_test`, `encode_path` utility code — reuse in Post-Gen SMT verification.
- Sentence-BERT embeddings and clustering outputs — reuse for semantic conflict checks and policy similarity filters.

(These are exact variable names seen in Group 2 code and should be consumed unchanged to minimize integration risk.)

---

## 3) Top-level architecture & component list (runtime modules)

Diagram mental model: `Client → API Gateway → PreGen → LLM (DuringGen injector) → LLM Response → PostGen → Action Router → Client/Admin`

Components (implement each):

1. **Integration Parser (Glue)**
   - Input: Group2 `processed` (JSON array) and `pseudocode`, `decision_graph`, `resolution_report`.
   - Output (single canonical runtime store): `policy_ir.json` containing:
     - `variables` (typed decision vars)
     - `conditional_rules` (list {policy_id, conditions, action, metadata})
     - `constraints` (global invariants)
     - `compiled_paths` (decision_graph compiled_paths)
     - `dominance_rules` (from resolution_report)
     - `policy_index_metadata` (for retrieval filters: domain, eff_date, owner, regulatory_linkage)

   - Must validate schema, set `processing_status` fields, and produce audit diff for each policy ingestion.
   - Produce a deterministic `policy_id` string (UUID v5 using source path + clause hash).

2. **Pre-Gen Stack**
   - **Query Classifier Router**: transformer classifier that outputs `domain` and `intent` plus retrieval filters (region, channel, customer_segment). Use Group2 router spec + training dataset (from DecisionsDev).
     - Endpoint: `/pre/classify` → `{query, session_metadata}` → `{domain, intent, confidence}`

   - **DPR Retriever & Index**: FAISS / Pinecone index containing policy _shards_ (Scope/Condition/Action/Exception) with metadata fields. Support vector text + metadata filtering (effective_date pruning).
     - Query returns top-K shards + composite `policy_set`.

   - **Dominance Resolver (Pre-gen)**: If retrieved set contains conflicting policies, consult `dominance_rules` to filter to winner set or mark for injection of multiple paths (compose vs override).
   - **Context Builder**: Prepare `injection_bundle` with:
     - `compiled_paths` for relevant policies
     - `invariant_constraints` to append to system prompt
     - `retrieved_docs` (for RAG)
     - `enforcement_metadata` (owners, escalation contact)

   - API: `/pre/build_context` → returns `injection_bundle`.

3. **During-Gen Injector (Scaffold module)**
   - **Scaffold Serializer**: Serializes `compiled_paths` into deterministic stepwise instructions (see example below).
   - **Invariant Prompt Injector**: Appends non-negotiable constraints (e.g., `Never disclose PII`, `Do not promise actions you cannot verify`) as high-priority system instructions. Represent them as short logical rules and short natural language statement.
   - **Priority Guidance Block**: Injects sentence: "If a regulatory policy applies, follow it first. If two policies conflict, apply this dominance rule: ...". Use only `dominance_rules` for machine-deterministic decisions.
   - Injection placement:
     - **System message**: invariant constraints + instruction hierarchy
     - **Assistant prompt (pre-gen context)**: RAG snippets + "Permission-aware boundary markers"
     - **User prompt augmentation**: short "task instruction" calling to follow scaffold steps

   - Determinism: scaffolds must be deterministic; set `temperature=0`/low and add `"/no_think"` or model-specific suppression token.
   - Output: LLM receives system + preface + scaffold + user query.

4. **Post-Gen Verification**
   - **Fast Regex / Keyword Filter** — immediate low-latency checks for explicit forbidden phrases, PII disclosure, explicit promises. If fails → escalate or auto-redact.
   - **SMT Checker (Z3)** — translate final LLM answer into assertions over `policy_ir` variables and run:
     - encode answer facts as constraints (e.g., model stated `refund_amount=100`, `days_since_purchase=31`) using the `z3_var`/`encode_test` utilities.
     - check logical consistency with rules and global constraints.
     - use `unsat_core()` or model witness to pinpoint violated constraints.

   - **Judge LLM (Semantic Evaluator)** — run a second LLM to assess tone, implied actions, and semantic compliance (e.g., "This response implies promise of immediate refund" while policy requires manager approval). Provide a confidence score. Use a calibrated scoring prompt and few-shot examples.
   - **Compliance Scorer** — aggregate signals into a single `S` (0–1). See formula below.
   - **Action Router** — perform pass / auto-correct / regenerate / escalate per `S`.

5. **Orchestration Layer**
   - Implement pipeline in a stateless microservice model (each component a container). Orchestrate with a controller that can run checks in parallel where safe (e.g., Regex & Judge LLM in parallel while SMT runs).
   - Provide circuit breakers and timeouts to guarantee response latency bounded.

6. **Audit & Escalation UI**
   - Show policy sources, decision path traversed, witness for failure, recommended owner(s), and ability to approve/override. Include a one-click “create ticket” to policy owner with evidence.

7. **Security & Access**
   - RBAC for `owner` field; encrypt policy store at rest; logs redact PII unless owner is authorized.

---

## 4) Data schemas (canonical runtime types)

Use these exact JSON examples as your canonical contract.

### `policy_ir.json` (top level)

```json
{
  "variables": { "<var_name>": {"type":"bool|int|float|enum","description":"...","values":[...]} },
  "conditional_rules": [
    {
      "policy_id": "POL-REFUND-001",
      "conditions": [{"var":"days_since_purchase","op":"<=","value":30}, ...],
      "action": {"type":"refund","value":"full"},
      "metadata": {"owner":"Customer Service Dept.","domain":"refund","priority":"company","eff_date":"2024-01-15","regulatory_linkage":["FTC Cooling-Off Rule"]}
    }
  ],
  "constraints": [
    {"policy_id":"C0_NO_PII_DISCLOSURE","constraint":"NOT(disclose_pii)","scope":"always","metadata":{...}}
  ],
  "compiled_paths": [ /* same shape as Group2 compiled_paths */ ],
  "dominance_rules": [ /* shape: {when: {policies_fire: [...]}, then: {mode:"override|compose", enforce:"POL-ID"}} ]
}
```

### `injection_bundle` (runtime)

```json
{
  "session_id": "...",
  "user_query": "...",
  "domain": "refund",
  "intent": "refund_request",
  "compiled_paths_to_inject": [ /* small set, 1-3 paths */ ],
  "invariant_constraints": [ "Never disclose PII", "Always verify receipt before promising refund" ],
  "retrieved_docs": [ {doc_id, source, excerpt, score} ],
  "dominance_rules_applied": [ ... ],
  "escalation_contacts": ["owner@email"]
}
```

### `postgen_result`

```json
{
  "llm_response": "...",
  "regex_flags": [...],
  "smt_result": {"sat": false, "violations": [{"policy_id": "...","witness": {...}}]},
  "semantic_eval": {"score": 0.92, "issues": [...], "explanation": "..."},
  "compliance_score": 0.88,
  "action": "auto_correct"
}
```

---

## 5) Scaffold serialization — concrete example (refund)

Take one `compiled_path`:

```json
{
  "policy_id": "P1_ELECTRONICS_RETURN",
  "path": [
    { "var": "has_receipt", "tests": [{ "op": "==", "value": true }] },
    {
      "var": "product_category",
      "tests": [{ "op": "==", "value": "electronics" }]
    },
    { "var": "days_since_purchase", "tests": [{ "op": "<=", "value": 15 }] }
  ],
  "leaf_action": "refund:full"
}
```

**Serialized scaffold (deterministic)** inserted into system + user prompt:

```
System message (high priority):
- INVARIANTS:
  1) NEVER disclose personal identifiers (PII) to external parties.
  2) ALWAYS verify purchase evidence before promising refunds.
- PRIORITY: regulatory > core_values > company > department > situational.

During-generation scaffold (step list to be followed verbatim, stop and CHECK each):
STEP 1: Check variable has_receipt. If unknown, ask the user for receipt details; DO NOT assume.
STEP 2: If has_receipt == TRUE, check product_category. If category == 'electronics', proceed STEP 3.
STEP 3: Check days_since_purchase <= 15. If TRUE THEN ACTION => execute refund:full. If FALSE THEN ACTION => manager_approval_required.
FINAL: State the action and cite the policy source: (source: electronics_policy_v2, eff_date: 2024-01-15).
```

**Injector rules**: add `"/no_think"` or model-specific no-cot flag. Set generation params: `temperature=0.0`, `max_tokens=...`.

---

## 6) SMT (Z3) verification strategy — example

1. **Encoding**: map statement in LLM response → z3vars using `z3_var()` (reuse Group2 code).
2. **Assertion**: For each asserted fact, add equality/assertion. E.g., model answered "refund approved for $100; purchase 31 days ago" → `days_since_purchase == 31`, `refund_amount == 100`, `refund:approved`.
3. **Check**: For each active policy rule, add its condition and check for contradiction with action. Use `solver.check()`.
4. **Witness & unsat**: If `solver.check() == sat` and contradictory actions exist, produce witness; if `unsat`, use `unsat_core()` to find minimal constraint set violated.

**Example conflict detection**:

- Rule A: `days_since_purchase <= 30 -> refund:full`
- LLM claims: `days_since_purchase == 31` and `EXECUTE refund:full`
- SMT encodes both and detects UNSAT (if we encode rules as invariants). Use `witness` to explain violation.

---

## 7) Compliance Score `S` (recommended, tunable)

Aggregate three signals (fast, formal, semantic) and coverage:

- `R` = regex/keyword pass (1 or 0) weighted `w_R = 0.10`
- `Z` = SMT pass (1 pass, 0 fail) weighted `w_Z = 0.55`
- `L` = Judge LLM semantic score in [0,1] weighted `w_L = 0.25`
- `C` = Coverage ratio = fraction of required decision nodes tested by scaffold (0–1) weighted `w_C = 0.10`

Compute:
`S = w_Z * Z + w_L * L + w_R * R + w_C * C`

Action mapping (tunable):

- `S >= 0.95` → PASS (deliver)
- `0.85 <= S < 0.95` → AUTO-CORRECT (internal stricter retry + regenerate)
- `0.70 <= S < 0.85` → REGENERATE (with stricter scaffold + stronger constraints)
- `S < 0.70` → ESCALATE (block) — notify owners with `witness` and evidence.

Justification: emphasize SMT because logical correctness is the most defensible.

---

## 8) Handoff rules & dominance resolution (how to use Group2 dominance rules)

- When Pre-Gen retrieves conflicting policies: query `dominance_rules` for `policies_fire`. If there is a rule with `mode:override`, automatically remove loser policies before scaffolding. If `mode:compose`, include both paths but mark composed step ordering (approval gating).
- If no `dominance_rules` exist for a pair: mark `same_priority_conflict` and require either:
  - Add both paths to scaffolding and force explicit verification steps (LLM must present evidence for both), **or**
  - Escalate for owner decision depending on severity (use metadata priority lattice).

---

## 9) Performance engineering & latency strategies

To meet a 201ms overhead target (goal), implement:

- **Cache**: per-session retrieval+scaffold cache (TTL based on `effective_date` and change stream). Most queries reuse same `compiled_paths` for a customer/user pair.
- **Parallel checks**: run Regex and JudgeLLM in parallel with SMT (SMT can be slower); use first failing signal to short-circuit.
- **Lightweight Judge LLM**: use distilled judge model for semantic checks (faster).
- **Batch Z3**: reuse solver contexts for similar sessions to avoid re-compiling all constraints each time; use incremental solving (`push`/`pop`).
- **Asynchronous retry for auto-correct/regenerate**: if first pass is low but service allows slightly higher tail latency for re-generation, orchestrate a quick retry. (Be careful to bound total latency.)
- **Profile**: measure per-component p50/p95/p99 and do hotspot optimizations (index sharding, FAISS GPU, Z3 worker pool).

---

## 10) Tests & validation (complete testing matrix)

Implement as code + data tests:

1. **Unit tests** for Integration Parser mapping every field (schema → policy_ir).
2. **Property tests** for determinism of scaffold serialization (same input → same scaffold).
3. **SMT regression suite**: take all conflict scenarios from Group2 `conflict_report` and assert SMT identifies same conflicts and produces same witnesses.
4. **Judge LLM calibration**: produce labeled dataset (compliant / borderline / violating) and calibrate thresholds. Store seed examples in tests.
5. **Adversarial red teaming**: prompt injections, mixing conflicting policies, missing evidence cases; evaluate pass/regenerate/escalate stats.
6. **Latency & load tests** at multiple QPS, measure tail latencies. Use synthetic DecisionDev queries and real policy corpus.
7. **End-to-end compliance tests**: ground truth responses (2,000 queries suggested in paper) — compute compliance rate, false positive rate; ensure meets target thresholds.
8. **Human in the loop**: create manual review tasks to test escalation flows and owner approvals.

---

## 11) Observability, logging and audit

- **Logging schema**: every transaction stores `session_id, query, timestamp, retrieved_policy_ids, scaffold, model_resp, postgen_result, compliance_score, final_action, owner_notified` to immutable audit log. Hash the log entry for tamper detection.
- **Retention**: logs retained per company policy (e.g., 1 year) with PII redaction options.
- **Dashboards**:
  - Compliance rate over time
  - False positive trends
  - Top policies causing escalations
  - Owner response latency & appeal outcomes

- **Alerts**:
  - Spike in escalations (>x/hr)
  - SMT failure surge
  - Degraded Judge LLM score drift (model drift)

- **Forensics**: store `witness` and snippet of decision path used for all blocked responses.

---

## 12) Governance & human workflows

- **Escalation UI**: show conflict evidence (witness), decision path, recommended action, accept/override buttons.
- **Owner notifications**: email + dashboard ticket with reproducible case and `witness`.
- **Change control**: when owners change a policy (update `effective_date` or body), Integration Parser must re-ingest, create new `policy_version`, and invalidate relevant caches.
- **Audit approvals**: owners can mark a policy `verified` → moves policy from `discovered_pattern` to `explicit` after mandatory human validation — system enforces via discovery.human_validated flag.

---

## 13) Deployment, CI/CD & infra

- Containerize each microservice (PreGen, Retriever, Injector, SMT, Judge LLM) and deploy on K8s with autoscaling.
- Secrets & RBAC via Vault/IAM.
- Use Canaries for model updates: deploy a new judge model to 5% traffic, monitor drift.
- CI gates: unit tests, security scans (SAST), and compliance smoke tests before merge.

---

## 14) Documentation & deliverables (what to hand to ops / owners)

1. `policy_ir` schema spec + JSON examples.
2. API docs: `/pre/classify`, `/pre/build_context`, `/during/inject`, `/post/verify`.
3. Integration Parser code + mapping spec (field ↔ field).
4. Scaffold templates and example injections for top domains.
5. SMT verification guide and sample constraints.
6. Test suites and red-team instruction set.
7. Operational runbook (incident response, owner notification templates).
8. RBAC & privacy controls specification.

---

## 15) Checklist — implementation tasks (ticklist you can give engineers)

- [ ] Implement Integration Parser (mapping tests)
- [ ] Build DPR index (FAISS/Pinecone) & ingestion pipeline
- [ ] Implement Query Classifier (train & eval)
- [ ] Implement Dominance Resolver (use Group2 dominance_rules)
- [ ] Scaffold Serializer & Prompt Injector
- [ ] Implement Invariant Prompt Block
- [ ] Implement Regex & fast filters
- [ ] Integrate Z3 SMT checker (reuse `z3_var`, `encode_test`)
- [ ] Build Judge LLM interface (calibrate)
- [ ] Implement Compliance Scorer & action router
- [ ] Build Audit log & Escalation UI
- [ ] Implement caching, parallel execution & performance optimizations
- [ ] Full test suites (unit, property, SMT regression, adversarial)
- [ ] Monitoring dashboards + alerts
- [ ] Documentation + Runbook

---

## 16) Example artifact — Integration Parser pseudocode (minimal)

```python
def parse_processed_to_policy_ir(processed_policy):
    # map scope -> variables (create enums/vars)
    # map actions -> conditional_rules
    # copy metadata (owner, eff_date, domain)
    # if discovery.human_validated == False: mark 'discovered' and require owner
    ...
```

(Implement deterministic hashing to generate `policy_id`.)

---

## 17) What I recommend you _do next now_ (practical short run)

- Pull Group2 outputs (one snapshot) and run the **Integration Parser** first — this unblocks Pre/During/Post work.
- Create a small test harness with 10 representative policies + 50 test queries from DecisionsDev and iterate scaffold → SMT → judge flow until the compliance score mapping produces intended actions.
- Add owners to any discovered implicit policies (required human validation) before they hit production.

---
