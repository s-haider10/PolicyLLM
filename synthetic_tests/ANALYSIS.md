# Pipeline Test Results Analysis

## Summary of Issues Found

The pipeline test revealed **critical bugs** in the extraction pipeline that explain the anomalous results:

---

## 🚨 Issue 1: Extraction Pipeline Merges Multiple Policies

**Problem**: The extractor combines all policies from a document into a single "super-policy" instead of creating separate policy objects.

**Evidence**:
- **stage1_explicit** document contains **4 distinct policies**:
  - POL-RETURNS-004 (refund with 7-day window)
  - POL-PRIVACY-001 (PII disclosure)
  - POL-RETURNS-009 (refund with 14-day window)
  - POL-SHIP-008 (48-hour shipping)

- **stage1_explicit** extraction found **1 policy**:
  - POL-OTHER-001 with 5 conditions, 4 actions mixed together

**Impact**: This breaks the entire pipeline because:
1. Policy boundaries are lost
2. Actions from different domains get mixed together
3. Conflict detection sees false positives

---

## 🚨 Issue 2: False Conflict Detection (Stage1: 3 Conflicts)

**Why stage1 has 3 conflicts despite being "explicit" policies**:

The 3 "conflicts" are actually **false positives** caused by Issue 1:

```json
Conflict 1: offer_refund vs disclose_pii
Conflict 2: offer_refund vs ship_within_48_hours  
Conflict 3: disclose_pii vs ship_within_48_hours
```

All conflicts are between actions from **different domains**:
- `offer_refund` → refund/return domain
- `disclose_pii` → privacy domain
- `ship_within_48_hours` → shipping domain

**These are NOT real conflicts** because:
- They serve different purposes
- They would never be triggered simultaneously in real scenarios
- They were originally separate policies before extraction merged them

The conflict detector found that with witness inputs:
```json
{
  "has_receipt": true,
  "days_since_purchase": 0,
  "contains_pii": true,
  "item_in_stock": true
}
```

All 3 actions can fire from the same merged policy, which the validation system correctly identifies as ambiguous (but the root cause is bad extraction).

---

## ✅ Issue 3: Stage2 Has 0 Conflicts (Expected)

**Why stage2_conflicts has 0 conflicts**:

Even though extraction merged 5 policies into 1, the extracted actions are:
- `verify_identity` (refund_amount >= 400)
- `offer_refund` (refund with conditions)
- `override_standard_policy` (exception handling)

These 3 actions **don't logically conflict** because:
1. They're all in the refund domain
2. They serve complementary purposes (verification → decision → exception)
3. The Z3 solver couldn't find input assignments where incompatible actions fire together

**Interesting finding**: Stage2 was designed to have conflicts between policies (POL-REFUND-005, POL-REFUND-010, POL-REFUND-015 with 14/21/28 day windows), but extraction merged them such that conflicts were lost.

---

## 🚨 Issue 4: Stage3 Extracts 0 Policies (Design Limitation)

**Why stage3_implicit extracted nothing**:

The document uses implicit, conversational language:
```
"returns typically get processed smoothly when customers 
bring in their receipt and come back within about three 
weeks of buying something"
```

**Root cause**: The extraction pipeline expects explicit conditional statements:
- `if X then Y`
- `when X, agents must Y`
- `X == True`

It cannot parse implicit policies written in natural language.

**Recommendation**: The extraction pipeline needs an additional pass to handle implicit policies. This could use LLM-based transformation to convert implicit statements to explicit conditions before extraction.

---

## 🚨 Issue 5: Stage4 Has 9 Conflicts (Cascading from Issue 1)

**Why stage4_mixed has 9 conflicts**:

Stage4 document contains 4 policies:
- POL-PRIVACY-016 (PII disclosure)
- POL-RETURNS-009 (14-day refund)
- Unnamed shipping policy (48-hour shipping)
- POL-REFUND-010 (21-day refund)

Extraction created 1 policy with 5 compiled paths (different condition combinations).

**Conflict explosion**:
- 5 paths → C(5,2) = 10 possible pairwise comparisons
- 9 conflicts detected (90% conflict rate)

This is because the merged policy mixes refund conditions (14-day vs 21-day windows), PII handling, and shipping rules into overlapping decision paths.

---

## Root Cause Summary

All issues trace back to **one fundamental bug**: 

**The extraction pipeline treats each document as a single policy unit instead of extracting individual policies as separate objects.**

### Expected Behavior:
```
Document with 4 policies → Extract 4 policy objects
```

### Actual Behavior:
```
Document with 4 policies → Extract 1 merged policy object
```

---

## Impact on Test Validity

| Stage | Expected | Actual | Valid Test? |
|-------|----------|--------|-------------|
| stage1_explicit | 4 policies, 0 conflicts | 1 policy, 3 false conflicts | ❌ No |
| stage2_conflicts | 5 policies, conflicts | 1 policy, 0 conflicts | ❌ No |
| stage3_implicit | 4 policies extracted | 0 policies (limitation) | ⚠️ Partial |
| stage4_mixed | 4 policies, some conflicts | 1 policy, 9 conflicts | ❌ No |

**Overall test validity**: ❌ **The test successfully ran the pipeline end-to-end, but the results are not meaningful due to extraction bugs.**

---

## Recommended Solutions (LLM-Based Approach)

> **Key Insight**: Policy boundary detection via regex/parsing is brittle and won't generalize to real-world documents. Instead, use LLM intelligence to identify, segment, normalize, and canonicalize policies.

---

## 🔧 Solution 1: LLM-Based Policy Discovery Pass (NEW PASS 0)

**Location**: `Extractor/src/passes/pass0_policy_discovery.py` (NEW)

**Purpose**: Have the LLM identify all distinct policies in a document before extraction begins.

### Implementation Design

```python
def discover_policies(document_text: str, llm_client: LLMClient) -> List[PolicySegment]:
    """
    Pass 0: Policy Discovery - Identify all distinct policies in document.
    
    Returns list of PolicySegment objects, each representing one policy.
    """
    prompt = f"""
Analyze this policy document and identify ALL distinct policies it contains.

Document:
{document_text}

For each distinct policy, provide:
1. policy_id: A unique identifier (generate one if not present, format: POL-<DOMAIN>-<NUM>)
2. policy_summary: One-sentence description of what the policy governs
3. domain: The domain/topic (refund, privacy, shipping, security, etc.)
4. priority_level: regulatory/company/department/situational (infer from text)
5. text_span: The exact text of this policy from the document

Output as JSON array of policy objects.

Important:
- Each policy is a SEPARATE rule with its own conditions and actions
- If the document contains implicit/conversational policies, still identify them
- Policies about different topics/domains should be separate
- Policies with conflicting conditions (e.g., 7-day vs 14-day refund) are SEPARATE

Output only valid JSON, no other text.
"""
    
    result = llm_client.invoke_json(prompt, schema=PolicyDiscoveryResponse)
    return result["policies"]
```

**Benefits**:
- **Handles any document format**: Explicit, implicit, conversational, mixed
- **Domain-aware**: Separates policies by topic automatically
- **Conflict-aware**: Identifies when multiple policies govern the same situation
- **Works without policy IDs**: Generates them when missing

---

## 🔧 Solution 2: Variable Schema Canonicalization System

**Location**: `Extractor/src/variable_schema.py` (NEW)

**Purpose**: Maintain a canonical variable schema so the same concepts use the same variable names across policies and documents.

### Implementation Design

#### A. Schema Structure

```python
@dataclass
class VariableSchema:
    """Canonical schema for policy variables."""
    
    # Core schema: canonical_name -> metadata
    variables: Dict[str, VariableDefinition] = field(default_factory=dict)
    
    # Reverse index: synonyms -> canonical_name
    synonym_map: Dict[str, str] = field(default_factory=dict)
    
    # Domain-specific variable groups
    domain_variables: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class VariableDefinition:
    canonical_name: str          # e.g., "days_since_purchase"
    type: str                    # boolean, numeric, enum
    domain: str                  # refund, privacy, shipping
    description: str             # Human-readable meaning
    synonyms: List[str]          # Alternative phrasings
    unit: Optional[str] = None   # days, dollars, etc.
```

#### B. Schema Initialization (Seed with Common Variables)

```python
DEFAULT_SCHEMA = VariableSchema(
    variables={
        "has_receipt": VariableDefinition(
            canonical_name="has_receipt",
            type="boolean",
            domain="refund",
            description="Whether customer has proof of purchase",
            synonyms=["receipt_present", "has_proof_of_purchase", "valid_receipt"]
        ),
        "days_since_purchase": VariableDefinition(
            canonical_name="days_since_purchase",
            type="numeric",
            domain="refund",
            description="Number of days elapsed since purchase date",
            synonyms=["purchase_age", "time_since_purchase", "days_elapsed"],
            unit="days"
        ),
        "refund_amount": VariableDefinition(
            canonical_name="refund_amount",
            type="numeric",
            domain="refund",
            description="Dollar amount of refund being requested",
            synonyms=["refund_value", "return_amount"],
            unit="dollars"
        ),
        "physical_damage": VariableDefinition(
            canonical_name="physical_damage",
            type="boolean",
            domain="refund",
            description="Whether item shows physical damage",
            synonyms=["damaged", "item_damaged", "has_damage"]
        ),
        "contains_pii": VariableDefinition(
            canonical_name="contains_pii",
            type="boolean",
            domain="privacy",
            description="Whether data contains personally identifiable information",
            synonyms=["has_pii", "personal_data_present", "pii_detected"]
        ),
        "item_in_stock": VariableDefinition(
            canonical_name="item_in_stock",
            type="boolean",
            domain="shipping",
            description="Whether requested item is available in inventory",
            synonyms=["in_stock", "inventory_available", "stock_available"]
        ),
    }
)
```

#### C. Variable Canonicalization Pass (NEW)

```python
def canonicalize_variables(
    policy_text: str,
    extracted_conditions: List[Dict],
    schema: VariableSchema,
    llm_client: LLMClient
) -> Tuple[List[Dict], VariableSchema]:
    """
    Normalize variable names to canonical schema.
    Updates schema with new variables if needed.
    """
    
    prompt = f"""
Given this policy text and extracted conditions, map each variable to its canonical name.

Policy Text:
{policy_text}

Extracted Conditions:
{json.dumps(extracted_conditions, indent=2)}

Current Variable Schema:
{json.dumps([{
    "canonical_name": v.canonical_name,
    "description": v.description,
    "synonyms": v.synonyms
} for v in schema.variables.values()], indent=2)}

For each extracted condition variable:
1. If it matches an existing canonical variable (same concept), map to that name
2. If it's a NEW concept, propose a canonical name following snake_case convention
3. Provide brief description and list any synonyms

Output format:
{{
  "mappings": [
    {{"extracted": "has_receipt", "canonical": "has_receipt", "is_new": false}},
    {{"extracted": "purchase_age_days", "canonical": "days_since_purchase", "is_new": false}}
  ],
  "new_variables": [
    {{
      "canonical_name": "severe_weather_alert",
      "type": "boolean",
      "domain": "shipping",
      "description": "Whether severe weather alert is currently active",
      "synonyms": ["weather_alert_active", "severe_weather"]
    }}
  ]
}}
"""
    
    result = llm_client.invoke_json(prompt, schema=VariableMappingResponse)
    
    # Update schema with new variables
    for new_var in result["new_variables"]:
        schema.variables[new_var["canonical_name"]] = VariableDefinition(**new_var)
        for syn in new_var.get("synonyms", []):
            schema.synonym_map[syn] = new_var["canonical_name"]
    
    # Apply mappings to conditions
    canonicalized = []
    for cond in extracted_conditions:
        mapping = next(m for m in result["mappings"] if m["extracted"] == cond["parameter"])
        cond["parameter"] = mapping["canonical"]
        canonicalized.append(cond)
    
    return canonicalized, schema
```

**Benefits**:
- **Consistency across documents**: Same concept = same variable name
- **Evolving schema**: Learns new variables as it encounters them
- **Synonym handling**: Maps alternative phrasings to canonical names
- **Type safety**: Tracks data types and units

---

## 🔧 Solution 3: Enhanced Extraction Pipeline Architecture

### New Pipeline Structure

```
INPUT: Raw document text

↓

PASS 0: Policy Discovery (LLM)
├─ Identify all distinct policies
├─ Assign policy IDs
├─ Determine domains and priorities
└─ Extract text span for each policy

↓ [Process each policy independently]

PASS 0.5: Normalization (LLM)
├─ Convert implicit → explicit conditionals
├─ Standardize format: "IF <conditions> THEN <action>"
└─ Resolve ambiguous language

↓

PASS 1-6: Existing extraction passes (per policy)
├─ Classification
├─ Condition extraction
├─ Action extraction
├─ Exception handling
├─ Metadata enrichment
└─ Validation

↓

PASS 7: Variable Canonicalization (LLM + Schema)
├─ Map variables to canonical schema
├─ Update global schema with new variables
└─ Ensure consistency across all policies

↓

OUTPUT: List of N policy objects (one per discovered policy)
└─ Each with consistent variable names
```

### Key Changes from Current Pipeline

**Current (BROKEN)**:
```
Document → [Process as single unit] → 1 merged policy
```

**Fixed (PROPOSED)**:
```
Document → [Discover N policies] → [Process each separately] → N policies
                                  ↓
                        [Canonicalize variables across all]
```

---

## 🔧 Solution 4: Handling Implicit Policies

**Location**: `Extractor/src/passes/pass0_5_normalization.py` (NEW)

**Purpose**: Transform implicit/conversational policy statements into explicit conditionals that existing extraction passes can handle.

### Implementation

```python
def normalize_policy_text(policy_text: str, llm_client: LLMClient) -> str:
    """
    Pass 0.5: Normalization - Convert implicit policy to explicit conditionals.
    """
    
    prompt = f"""
Rewrite this policy statement as explicit conditional rules using clear IF-THEN format.

Original Policy:
{policy_text}

Requirements:
1. Convert implicit language to explicit conditions
   - "typically" → "when/if"
   - "usually" → "if"
   - "tends to" → specify the condition
   
2. Use explicit conditional format:
   IF <condition1> AND <condition2> THEN <action>
   
3. Use concrete variable names in snake_case:
   - "has a receipt" → has_receipt == True
   - "within 3 weeks" → days_since_purchase <= 21
   - "amount is high" → refund_amount >= 400
   
4. Make boolean comparisons explicit:
   - NOT "when receipt present"
   - USE "when has_receipt == True"
   
5. Specify operators clearly:
   - Use ==, !=, <, <=, >, >= explicitly
   
Example:
Original: "Returns typically get processed when customers have their receipt and come back within three weeks."
Normalized: "IF has_receipt == True AND days_since_purchase <= 21 THEN agents must offer_refund."

Now normalize the policy above. Output only the normalized text, no explanations.
"""
    
    result = llm_client.invoke_json(prompt)
    return result["normalized_policy"]
```

**Benefits**:
- **Handles implicit policies**: Stage3 documents will now extract successfully
- **Standardized format**: All policies go through extraction in same format
- **Preserves intent**: Maintains original policy meaning while making it parseable

---

## 🔧 Solution 5: Cross-Document Schema Harmonization

**Challenge**: When processing multiple documents, ensure variables remain consistent.

### Approach: Persistent Schema with Global State

```python
class ExtractionOrchestrator:
    """
    Manages extraction across multiple documents with shared schema.
    """
    
    def __init__(self):
        self.schema = load_or_create_schema()  # Persistent across runs
        self.policy_registry = {}              # Track all extracted policies
        
    def extract_document(self, document_path: str) -> List[Policy]:
        """Extract policies from document using global schema."""
        
        # 1. Discover policies
        policies = discover_policies(document_text, self.llm)
        
        extracted_policies = []
        for policy_segment in policies:
            # 2. Normalize (implicit → explicit)
            normalized_text = normalize_policy_text(policy_segment.text, self.llm)
            
            # 3. Run extraction passes
            raw_policy = run_extraction_passes(normalized_text, self.config)
            
            # 4. Canonicalize variables using GLOBAL schema
            canonical_policy, self.schema = canonicalize_variables(
                normalized_text,
                raw_policy.conditions,
                self.schema,  # Shared across all documents
                self.llm
            )
            
            extracted_policies.append(canonical_policy)
        
        # 5. Save updated schema for next document
        self.schema.save()
        
        return extracted_policies
```

**Benefits**:
- **Multi-document consistency**: Same variables across entire corpus
- **Schema evolution**: Learns as it processes more documents
- **Audit trail**: Can review how schema evolved over time

---

## 🔧 Solution 6: Conflict-Aware Policy Discovery

**Enhancement**: Make the LLM proactively identify potential conflicts during discovery.

```python
# Add to Pass 0 prompt:
"""
Additionally, identify any policies that might CONFLICT with each other:
- Same domain but different conditions (e.g., 7-day vs 14-day refund window)
- Same action but different priorities (regulatory vs department)
- Overlapping conditions with incompatible actions

Flag these as "conflict_candidates" with brief explanation.
"""
```

This helps stage2_conflicts tests work as intended by:
1. Ensuring conflicting policies are extracted separately
2. Flagging potential conflicts early
3. Providing hints to validation stage

---

## Implementation Priority

### Phase 1: Core Fixes (Week 1)
1. ✅ **Pass 0: Policy Discovery** - Most critical, fixes the multi-policy merge issue
2. ✅ **Pass 0.5: Normalization** - Handles implicit policies
3. ✅ **Update pipeline.py** - Process each discovered policy independently

### Phase 2: Consistency (Week 2)
4. ✅ **Variable Schema System** - Build canonical schema
5. ✅ **Pass 7: Canonicalization** - Apply schema to extracted policies
6. ✅ **Schema Persistence** - Save/load schema across runs

### Phase 3: Cross-Document (Week 3)
7. ✅ **Orchestrator Pattern** - Manage global schema across documents
8. ✅ **Schema Evolution UI** - Tool to review/approve new variables
9. ✅ **Conflict-Aware Discovery** - Enhanced conflict detection

---

## Expected Results After Fixes

| Stage | Current Result | Expected After Fix |
|-------|----------------|-------------------|
| stage1_explicit | 1 policy, 3 false conflicts | **4 policies, 0 conflicts** |
| stage2_conflicts | 1 policy, 0 conflicts | **5 policies, 2-3 real conflicts** |
| stage3_implicit | 0 policies | **4 policies** (after normalization) |
| stage4_mixed | 1 policy, 9 conflicts | **4 policies, 1-2 real conflicts** |

---

## Testing Strategy

### 1. Unit Tests for Each New Pass
- `test_pass0_policy_discovery.py` - Test policy segmentation
- `test_pass0_5_normalization.py` - Test implicit→explicit conversion
- `test_variable_canonicalization.py` - Test schema mapping

### 2. Integration Test
```python
def test_multi_policy_extraction():
    """Verify multiple policies extracted separately."""
    doc = load_test_document("stage1_explicit/doc_001.md")
    policies = extract_with_new_pipeline(doc)
    
    assert len(policies) == 4, "Should extract 4 separate policies"
    assert policies[0].policy_id == "POL-RETURNS-004"
    assert policies[1].policy_id == "POL-PRIVACY-001"
    # Verify no variable name collisions
    all_vars = [v for p in policies for v in p.conditions]
    assert len(set(all_vars)) > len(all_vars) * 0.8, "Variables should be mostly unique"
```

### 3. Re-run Full Pipeline Test
```bash
cd synthetic_tests
bash run_tests_gpu0.sh
```

Expected: All 4 stages pass with correct policy counts and meaningful conflict detection.

---

## Architectural Benefits

This LLM-based approach provides:

1. **Robustness**: Works with any document format (explicit, implicit, mixed)
2. **Scalability**: Schema evolves as corpus grows
3. **Maintainability**: No brittle regex patterns to update
4. **Accuracy**: LLM understands semantic policy boundaries
5. **Flexibility**: Easy to add new domains/variables
6. **Debuggability**: Clear separation between discovery, normalization, extraction

---

## Next Steps

1. **Implement Pass 0 (Policy Discovery)** - Highest impact fix
2. **Test on synthetic data** - Verify it correctly identifies 4 policies in stage1
3. **Implement Pass 0.5 (Normalization)** - Fix stage3_implicit
4. **Build Variable Schema System** - Ensure consistency
5. **Re-run full pipeline tests** - Validate fixes
6. **Iterate on prompts** - Improve discovery/normalization quality

The good news: **The test infrastructure works correctly**. Extraction, validation, conflict detection, and enforcement all ran successfully. The issue is purely in extraction quality, which these LLM-based solutions directly address.
