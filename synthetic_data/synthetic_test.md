# Synthetic Testing Framework for PolicyLLM

## Overview

This document outlines a comprehensive synthetic testing methodology for the PolicyLLM pipeline. The goal is to systematically evaluate the pipeline's ability to extract, validate, and enforce policies by generating controlled synthetic data with known ground truth.

**Testing Philosophy:** By generating a ground-truth constitution and then creating documents that encode its policies with varying degrees of explicitness and complexity, we can measure the pipeline's accuracy in reconstructing the original constitution from the documents.

**Pipeline Being Tested:** `Document → Extraction → Validation → Enforcement → ComplianceDecision`

---

## Testing Stages

The synthetic testing framework consists of four progressive stages that increase in complexity and realism:

### Stage 1: Pure Explicit Constitution

**Objective:** Establish baseline accuracy with fully explicit policy encoding.

**Constitution:** 
- Generate a ground-truth constitution with N policies
- Each policy has well-defined: scope, conditions, actions, exceptions, priority, metadata
- All policies are internally consistent (no conflicts)

**Synthetic Documents:**
- Generate M documents using an LLM
- Each document explicitly states one or more policies from the constitution
- Documents collectively encode ALL policies from the ground-truth constitution
- Language is clear and unambiguous (e.g., "If X then Y")
- No implicit policies, no conflicting statements

**Success Metrics:**
- **Extraction Accuracy:** % of ground-truth policies correctly extracted
- **Policy Fidelity:** Precision/recall of conditions, actions, exceptions
- **Structural Integrity:** Correct metadata assignment (priority, domain, owner)
- **Completeness:** All N policies recovered from M documents

**Expected Performance:** ≥95% reconstruction accuracy (high baseline)

---

### Stage 2: Explicit Constitution with Low-Priority Conflicts

**Objective:** Test the pipeline's conflict detection and priority resolution capabilities.

**Constitution:**
- Same ground-truth constitution from Stage 1 (N policies)
- All policies remain high-priority (regulatory, company-level)

**Synthetic Documents:**
- Generate M documents (same as Stage 1)
- **NEW:** Sprinkle K conflicting policies that contradict ground-truth policies
- Conflicting policies are explicitly marked as LOW priority (situational, department-level)
- Conflicts are logical contradictions (e.g., "refund within 30 days" vs. "no refunds after 14 days")
- Ground-truth policies should dominate low-priority conflicts via priority lattice

**Success Metrics:**
- **Conflict Detection Rate:** % of K conflicts detected by Z3 SMT solver
- **Priority Resolution:** % of conflicts correctly resolved in favor of high-priority policies
- **Bundle Integrity:** compiled_policy_bundle.json correctly applies dominance rules
- **Extraction Robustness:** Still achieves ≥90% accuracy on ground-truth policies despite noise

**Expected Performance:** 100% conflict detection, correct priority-based resolution

---

### Stage 3: Fully Implicit Constitution

**Objective:** Stress-test extraction on implicit, real-world-style policy language.

**Constitution:**
- Same ground-truth constitution (N policies)
- Policies remain the source of truth

**Synthetic Documents:**
- Generate M documents with NO explicit policy statements
- Policies encoded implicitly through:
  - Case examples ("John returned his laptop after 20 days and received store credit")
  - Conditional narratives ("While we typically offer refunds...")
  - Embedded assumptions ("As always, our 14-day policy applies")
  - Hedging language ("In most cases...", "Generally speaking...")
- Requires LLM to infer conditions/actions from context
- Tests extraction pipeline's ability to handle real-world ambiguity

**Success Metrics:**
- **Extraction Accuracy:** % of policies correctly inferred from implicit language
- **False Positive Rate:** % of hallucinated policies not in ground-truth
- **Confidence Calibration:** Extracted policies should have lower confidence scores than Stage 1
- **Coverage:** % of ground-truth policies at least partially recovered

**Expected Performance:** 60-80% accuracy (realistic for implicit extraction)

---

### Stage 4: Mixed Reality Simulation

**Objective:** Combine all previous stages to simulate real-world policy document diversity.

**Constitution:**
- Same ground-truth constitution (N policies)

**Synthetic Documents:**
- Generate M documents where each document is randomly assigned a stage profile:
  - **50% Stage 1 style:** Explicit policy statements
  - **15% Stage 2 style:** Explicit with low-priority conflicts
  - **25% Stage 3 style:** Fully implicit
  - **10% Hybrid:** Mixed explicit/implicit within same document
- Mimics real organizational policy sources (handbooks, emails, case logs, legal docs)
- Tests pipeline's robustness to heterogeneous input

**Why this baseline (literature-informed):**
- **Explicit-majority (50%)** is motivated by legal/policy NLP benchmarks built around explicit clause-level evidence and entailment in contracts (Koreeda & Manning, 2021; Chalkidis et al., 2022), suggesting a substantial explicit-document component in formal policy corpora.
- **Implicit-but-substantial (25%)** is motivated by organizational knowledge literature on tacit vs explicit knowledge (Nonaka, 1994; Lam, 2000), which supports that many operational rules are conveyed indirectly (practice, narrative, context) rather than as formal clauses.
- **Conflicts as minority but non-trivial (15%)** is motivated by evidence that exception/negation structures materially affect legal inference and compliance interpretation (Koreeda & Manning, 2021), and by process-compliance literature that treats constraint violations/conflicts as important but specialized cases requiring dedicated conformance checks (Lu et al., 2009).
- **Hybrid docs (10%)** captures mixed artifacts (e.g., handbooks with embedded case notes) without over-weighting them as the dominant source type.

**Important caveat:** There is no widely accepted study that directly estimates a universal explicit/conflict/implicit/hybrid percentage split for organizational policy documents. This distribution should be treated as a defensible baseline prior, then re-estimated on your domain corpus.

**Success Metrics:**
- **Overall Extraction Accuracy:** Weighted average across document types
- **Conflict Handling:** Correct resolution of mixed-priority conflicts
- **Scalability:** Performance degradation with increasing M
- **Enforcement Accuracy:** End-to-end test with actual queries
  - Generate Q test queries with known compliance outcomes
  - Measure: PASS/FAIL accuracy, false positive rate, false negative rate

**Expected Performance:** 75-85% overall accuracy, robust enforcement decisions

### Stage 4 Small-Scale Sensitivity Analysis

**Objective:** Verify that Stage 4 conclusions are not artifacts of a single document-mix assumption.

**Protocol (lightweight):**
- Keep constitution fixed (same N policies) and query set fixed (same Q queries).
- Run 3 random seeds per distribution.
- Use a small batch per run: **M = 20 documents**.
- Report mean ± std for extraction and enforcement metrics.

**Distributions to Compare:**
- **Baseline (literature-informed):** `[0.50, 0.15, 0.25, 0.10]`  
  (explicit, conflicts, implicit, hybrid)
- **Explicit-heavy:** `[0.65, 0.10, 0.15, 0.10]`
- **Implicit-heavy:** `[0.35, 0.15, 0.40, 0.10]`
- **Conflict-stress:** `[0.40, 0.30, 0.20, 0.10]`

**Primary Metrics for Sensitivity Check:**
- Δ Overall Extraction Accuracy vs baseline
- Δ Conflict Detection Rate vs baseline
- Δ Enforcement PASS/ESCALATE accuracy vs baseline
- Variance across seeds (stability)

**Acceptance Heuristic (for assumption validity):**
- Baseline assumption is considered reasonably robust if:
  - absolute metric deltas stay within **±5 percentage points** for explicit-heavy and implicit-heavy settings, and
  - conflict-stress does not reduce conflict-resolution correctness below **0.90**.

**Interpretation:**
- If performance is highly sensitive to mix choice, report Stage 4 results as **distribution-conditional** and avoid a single aggregated claim.
- If sensitivity is low, retain the baseline distribution for larger-scale runs.

---

## Synthetic Data Generation Pipeline

### 1. Constitution Generation

```python
# Pseudocode
constitution = generate_ground_truth_constitution(
    num_policies=N,
    domains=["refund", "privacy", "security", "shipping"],
    priority_levels=["regulatory", "company", "department", "situational"],
    ensure_consistency=True
)
# Output: ground_truth_constitution.json
```

**Variables:**
- N = 10-50 policies (configurable)
- Each policy has unique policy_id
- Policies span multiple domains for diversity

### 2. Document Generation (Stage-Specific)

```python
# Stage 1: Explicit
documents = generate_explicit_documents(
    constitution=constitution,
    num_documents=M,
    policies_per_doc=2-5,  # overlap allowed
    encoding="explicit"
)

# Stage 2: Explicit + Conflicts
documents = generate_conflicting_documents(
    constitution=constitution,
    num_documents=M,
    num_conflicts=K,
    conflict_priority="low"
)

# Stage 3: Implicit
documents = generate_implicit_documents(
    constitution=constitution,
    num_documents=M,
    encoding_strategies=["case_examples", "narratives", "hedging"]
)

# Stage 4: Mixed
documents = generate_mixed_documents(
    constitution=constitution,
    num_documents=M,
  stage_distribution=[0.5, 0.15, 0.25, 0.1]
)
```

**LLM Parameters:**
- Model: GPT-4, Claude Sonnet, or Llama 3 (configurable)
- Temperature: 0.7 (allow variability in phrasing)
- Max tokens: 1000-2000 per document
- Few-shot prompting with stage-specific examples

### 3. Query Generation (Stage 4 Only)

```python
queries = generate_test_queries(
    constitution=constitution,
    num_queries=Q,
    categories=["valid_path", "violation", "uncovered", "edge_case"]
)
# Output: test_queries.json with ground-truth labels
```

---

## Evaluation Metrics

### Extraction Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Policy Recall** | TP policies / N ground-truth policies | ≥0.90 |
| **Policy Precision** | TP policies / extracted policies | ≥0.85 |
| **Condition F1** | F1 score on conditions per policy | ≥0.80 |
| **Action Accuracy** | % policies with correct action | ≥0.95 |
| **Metadata Accuracy** | % correct priority/domain assignment | ≥0.90 |

### Validation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Conflict Detection Rate** | Detected conflicts / K true conflicts | 1.00 |
| **False Conflict Rate** | False conflicts / total checks | ≤0.05 |
| **Priority Correctness** | Correct resolutions / total conflicts | 1.00 |
| **Bundle Completeness** | Policies in bundle / extracted policies | 1.00 |

### Enforcement Metrics (Stage 4)

| Metric | Formula | Target |
|--------|---------|--------|
| **PASS Accuracy** | Correct PASS / total PASS queries | ≥0.95 |
| **ESCALATE Accuracy** | Correct ESCALATE / total violations | ≥0.90 |
| **False Positive Rate** | Incorrect ESCALATE / total queries | ≤0.05 |
| **False Negative Rate** | Missed violations / total violations | ≤0.10 |
| **Compliance Score Calibration** | Correlation with ground-truth | ≥0.85 |

---

## Directory Structure

```
synthetic_data/
├── stage1_explicit/
│   ├── ground_truth_constitution.json
│   ├── documents/
│   │   ├── doc_001.md
│   │   ├── doc_002.md
│   │   └── ...
│   ├── extracted_policies.jsonl
│   ├── compiled_bundle.json
│   └── evaluation_report.json
│
├── stage2_conflicts/
│   ├── ground_truth_constitution.json
│   ├── conflicting_policies.json
│   ├── documents/
│   ├── extracted_policies.jsonl
│   ├── compiled_bundle.json
│   ├── conflict_report.json
│   └── evaluation_report.json
│
├── stage3_implicit/
│   ├── ground_truth_constitution.json
│   ├── documents/
│   ├── extracted_policies.jsonl
│   ├── compiled_bundle.json
│   └── evaluation_report.json
│
├── stage4_mixed/
│   ├── ground_truth_constitution.json
│   ├── documents/
│   ├── test_queries.json
│   ├── extracted_policies.jsonl
│   ├── compiled_bundle.json
│   ├── enforcement_results.json
│   └── evaluation_report.json
│
└── generation_scripts/
    ├── generate_constitution.py
    ├── generate_documents.py
    ├── generate_queries.py
    └── evaluate_pipeline.py
```

---

## Execution Workflow

### Phase 1: Data Generation

```bash
# Generate ground-truth constitution
python synthetic_data/generation_scripts/generate_constitution.py \
    --num-policies 20 \
    --domains refund privacy security \
    --out synthetic_data/stage1_explicit/

# Generate stage-specific documents
python synthetic_data/generation_scripts/generate_documents.py \
    --stage 1 \
    --constitution synthetic_data/stage1_explicit/ground_truth_constitution.json \
    --num-documents 15 \
    --out synthetic_data/stage1_explicit/documents/
```

### Phase 2: Run Pipeline

```bash
# Extract policies
python main.py extract synthetic_data/stage1_explicit/documents/ \
    --out synthetic_data/stage1_explicit/ \
    --config Extractor/configs/config.chatgpt.yaml

# Validate and compile
python main.py validate synthetic_data/stage1_explicit/extracted_policies.jsonl \
    --out synthetic_data/stage1_explicit/compiled_bundle.json
```

### Phase 3: Evaluation

```bash
# Compare extracted policies to ground truth
python synthetic_data/generation_scripts/evaluate_pipeline.py \
    --stage 1 \
    --ground-truth synthetic_data/stage1_explicit/ground_truth_constitution.json \
    --extracted synthetic_data/stage1_explicit/extracted_policies.jsonl \
    --bundle synthetic_data/stage1_explicit/compiled_bundle.json \
    --out synthetic_data/stage1_explicit/evaluation_report.json
```

### Phase 4: Enforcement Testing (Stage 4 Only)

```bash
# Run enforcement on test queries
python main.py enforce \
    --bundle synthetic_data/stage4_mixed/compiled_bundle.json \
    --queries synthetic_data/stage4_mixed/test_queries.json \
    --out synthetic_data/stage4_mixed/enforcement_results.json

# Evaluate enforcement accuracy
python synthetic_data/generation_scripts/evaluate_pipeline.py \
    --stage 4 \
    --ground-truth synthetic_data/stage4_mixed/ground_truth_constitution.json \
    --enforcement synthetic_data/stage4_mixed/enforcement_results.json \
    --out synthetic_data/stage4_mixed/evaluation_report.json
```

---

## Configuration Parameters

### Constitution Generation

```yaml
constitution_config:
  num_policies: 20
  domains:
    - refund
    - privacy
    - security
    - shipping
    - returns
  priority_distribution:
    regulatory: 0.2
    company: 0.4
    department: 0.3
    situational: 0.1
  complexity:
    conditions_per_policy: [1, 5]  # min, max
    actions_per_policy: [1, 3]
    exceptions_per_policy: [0, 2]
```

### Document Generation

```yaml
document_config:
  num_documents: 15
  policies_per_document: [2, 5]
  llm_model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1500
  
  # Stage-specific
  stage1_explicit:
    language_style: "formal_legal"
    
  stage2_conflicts:
    num_conflicts: 5
    conflict_priority: "situational"
    
  stage3_implicit:
    encoding_strategies:
      - case_examples: 0.4
      - narratives: 0.3
      - hedging: 0.3
    
  stage4_mixed:
    stage_distribution:
      explicit: 0.5
      conflicts: 0.15
      implicit: 0.25
      hybrid: 0.1
```

### Query Generation (Stage 4)

```yaml
query_config:
  num_queries: 50
  query_distribution:
    valid_path: 0.4      # Should PASS
    violation: 0.3       # Should ESCALATE
    uncovered: 0.2       # Outside policy scope
    edge_case: 0.1       # Boundary conditions
  complexity:
    multi_condition: 0.3  # Queries requiring multiple conditions
    exception_trigger: 0.2  # Queries hitting exception clauses
```

---

## Expected Outcomes

### Stage 1 Results

- **Extraction Accuracy:** 95-98%
- **Bundle Compilation:** 100% success
- **Key Insight:** Establishes upper bound on pipeline performance

### Stage 2 Results

- **Conflict Detection:** 100% (if Z3 working correctly)
- **Priority Resolution:** 100% (deterministic lattice)
- **Extraction Robustness:** 90-95%
- **Key Insight:** Validates symbolic verification layer

### Stage 3 Results

- **Extraction Accuracy:** 60-80%
- **False Positive Rate:** 10-20%
- **Confidence Calibration:** Lower scores than Stage 1
- **Key Insight:** Identifies extraction limitations on implicit policies

### Stage 4 Results

- **Overall Accuracy:** 75-85%
- **Enforcement PASS Accuracy:** 95%+
- **Enforcement ESCALATE Accuracy:** 90%+
- **Key Insight:** Real-world performance estimate

---

## Reproducibility

- **Random Seed:** Set seed for all LLM calls and random sampling
- **LLM Temperature:** 0.7 for generation, 0.0 for extraction/enforcement
- **Versioning:** Track LLM model versions, PolicyLLM code commit SHA
- **Artifacts:** Save all intermediate outputs (documents, policies, bundles)
- **Logging:** Full audit trail for each stage

---

## Future Extensions

1. **Multi-lingual Testing:** Generate documents in multiple languages
2. **Temporal Evolution:** Test on constitutions that change over time
3. **Adversarial Testing:** Intentionally ambiguous or contradictory documents
4. **Scale Testing:** N=100+ policies, M=100+ documents
5. **Cross-domain Transfer:** Train on one domain, test on another

---

## References

- PolicyLLM System Design: [system_design.md](system_design.md)
- Extraction Pipeline: [Extractor/overview/extraction-overview.md](Extractor/overview/extraction-overview.md)
- Validation DAG: [Validation/decision_graph.py](Validation/decision_graph.py)
- Enforcement Scoring: [Enforcement/orchestrator.py](Enforcement/orchestrator.py)
- Koreeda, Y., & Manning, C. D. (2021). *ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts*. Findings of EMNLP 2021. DOI: 10.18653/v1/2021.findings-emnlp.164
- Chalkidis, I., et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*. ACL 2022. DOI: 10.18653/v1/2022.acl-long.297
- Nonaka, I. (1994). *A Dynamic Theory of Organizational Knowledge Creation*. Organization Science. DOI: 10.1287/orsc.5.1.14
- Lam, A. (2000). *Tacit Knowledge, Organizational Learning and Societal Institutions: An Integrated Framework*. Organization Studies. DOI: 10.1177/0170840600213001
- Lu, R., Sadiq, S., Governatori, G., & Namiri, K. (2009). *On compliance checking for clausal constraints in annotated process models*. Information Systems Frontiers. DOI: 10.1007/s10796-009-9179-7
- Pickering, J. (2021). *Ambiguity and legal compliance*. Criminology & Public Policy. DOI: 10.1111/1745-9133.12565
