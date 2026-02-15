# LLM Prompt Templates for Extraction Pipeline Fixes

This document contains **ready-to-use prompt templates** for implementing the three core solutions.

---

## 1. Pass 0: Policy Discovery Prompt

### Purpose
Identify all distinct policies in a document and segment them for independent processing.

### Prompt Template

```python
POLICY_DISCOVERY_PROMPT = """
Analyze this policy document and identify ALL distinct policies it contains.

A policy is distinct if:
- It governs a different domain/topic (refund vs privacy vs shipping)
- It has conflicting conditions with another policy (7-day vs 14-day window)
- It is explicitly labeled with a different policy ID
- It serves a different purpose or applies to different situations

Document:
{document_text}

For each distinct policy, provide:

1. **policy_id**: 
   - If present in text, extract it (e.g., "POL-RETURNS-004")
   - If missing, generate format: POL-<DOMAIN>-<NUMBER>
   - Examples: POL-REFUND-001, POL-PRIVACY-001, POL-SHIPPING-001

2. **policy_summary**: 
   - One-sentence description of what this policy governs
   - Focus on the core rule/requirement

3. **domain**: 
   - The primary topic area (refund, privacy, shipping, security, data_retention, etc.)
   - Use snake_case lowercase

4. **priority_level**: 
   - Infer from text: regulatory/company/department/situational
   - If unclear, use: company

5. **text_span**: 
   - The EXACT text from the document that contains this policy
   - Include full sentences, don't truncate

6. **scope**: 
   - Who does this apply to? (all_customers, VIP_only, employees, etc.)
   - Default: all_customers

IMPORTANT RULES:
- If two policies are about the SAME topic but have DIFFERENT conditions (e.g., different time windows), they are SEPARATE policies
- If text says "Under Policy POL-X" and later "Under Policy POL-Y", these are definitely separate
- Implicit policies (conversational language) should still be identified as distinct policies
- Even if policies seem complementary, if they have different conditions/actions, separate them

Output as JSON array:
{
  "policies": [
    {
      "policy_id": "POL-RETURNS-004",
      "policy_summary": "Customers with receipt and purchase within 7 days can get refund unless item is damaged",
      "domain": "refund",
      "priority_level": "regulatory",
      "text_span": "Under Policy POL-RETURNS-004...",
      "scope": "all_customers",
      "conflict_candidates": ["POL-RETURNS-009"]  // Optional: List IDs this might conflict with
    }
  ],
  "total_policies_found": 4,
  "notes": "Found overlapping refund policies with different time windows (7-day vs 14-day)"
}

Output ONLY valid JSON, no other text.
"""
```

### Expected Output Example

```json
{
  "policies": [
    {
      "policy_id": "POL-RETURNS-004",
      "policy_summary": "7-day refund policy with receipt required, denied if physical damage present",
      "domain": "refund",
      "priority_level": "regulatory",
      "text_span": "Under Policy POL-RETURNS-004, designated as regulatory priority, agents must offer a refund to customers when the customer has a receipt (has_receipt == True) and the purchase was made within seven days or fewer (days_since_purchase <= 7); however, if physical damage is present (physical_damage == True), agents must deny the refund regardless of receipt status or days since purchase.",
      "scope": "all_customers",
      "conflict_candidates": ["POL-RETURNS-009"]
    },
    {
      "policy_id": "POL-PRIVACY-001",
      "policy_summary": "PII disclosure requirement when data contains personally identifiable information",
      "domain": "privacy",
      "priority_level": "company",
      "text_span": "Under Policy POL-PRIVACY-001, designated as company priority, when any data or information contains personally identifiable information (contains_pii equals True), agents shall be required to disclose the presence of such personally identifiable information (disclose_pii).",
      "scope": "all_data_handlers",
      "conflict_candidates": []
    }
  ],
  "total_policies_found": 4,
  "notes": "Document contains policies across multiple domains: refund (2), privacy (1), shipping (1). Two refund policies have different time windows."
}
```

---

## 2. Pass 0.5: Normalization Prompt

### Purpose
Convert implicit/conversational policy text into explicit conditional statements.

### Prompt Template

```python
NORMALIZATION_PROMPT = """
Rewrite this policy statement as explicit conditional rules using clear IF-THEN format.

Original Policy:
{policy_text}

TRANSFORMATION RULES:

1. **Convert implicit language to explicit conditions:**
   - "typically" → "IF" / "WHEN"
   - "usually" → "IF"
   - "tends to" → specify the actual condition
   - "in most cases" → "IF"
   - "generally" → "IF"
   - "as a rule" → always/must

2. **Use explicit conditional format:**
   ```
   IF <condition1> AND <condition2> THEN agents must <action>
   ```

3. **Use concrete variable names in snake_case:**
   - "customer has a receipt" → has_receipt == True
   - "within 3 weeks" → days_since_purchase <= 21
   - "amount is at least $400" → refund_amount >= 400
   - "item is damaged" → physical_damage == True
   - "data contains PII" → contains_pii == True

4. **Make boolean comparisons explicit:**
   - NOT: "when receipt present"
   - USE: "WHEN has_receipt == True"

5. **Specify comparison operators clearly:**
   - Use: ==, !=, <, <=, >, >= (not "at least", "more than", etc.)

6. **Quantify vague terms:**
   - "within a few weeks" → <= 21 days (3 weeks)
   - "recently" → <= 7 days
   - "high amount" → >= 400 (dollars)
   - "quickly" → <= 48 hours

7. **Structure exceptions properly:**
   ```
   IF <conditions> THEN <action>
   HOWEVER IF <exception_condition> THEN <exception_action> INSTEAD
   ```

EXAMPLES:

Input: "Returns typically get processed smoothly when customers bring in their receipt and come back within about three weeks of buying something."

Output: "IF has_receipt == True AND days_since_purchase <= 21 THEN agents must offer_refund."

---

Input: "When security matters come up, staff have typically started by looking at whether the refund amount is at least 400. In those situations, the usual practice has been to verify the person's identity before moving forward."

Output: "IF refund_amount >= 400 THEN agents must verify_identity before processing."

---

Input: "Under Policy POL-RETURNS-004, agents must offer a refund to customers when the customer has a receipt and the purchase was made within seven days; however, if physical damage is present, agents must deny the refund."

Output: "IF has_receipt == True AND days_since_purchase <= 7 THEN agents must offer_refund. HOWEVER IF physical_damage == True THEN agents must deny_refund INSTEAD."

---

Now normalize the policy above. 

Output format:
{
  "normalized_policy": "IF ... THEN ... [HOWEVER IF ... THEN ... INSTEAD]",
  "variables_identified": [
    {"name": "has_receipt", "type": "boolean"},
    {"name": "days_since_purchase", "type": "numeric", "unit": "days"}
  ],
  "actions_identified": ["offer_refund", "deny_refund"],
  "confidence": 0.95,
  "notes": "Converted implicit 'typically' language to explicit IF-THEN"
}

Output ONLY valid JSON, no explanations outside the JSON.
"""
```

### Expected Output Example

```json
{
  "normalized_policy": "IF has_receipt == True AND days_since_purchase <= 21 THEN agents must offer_refund.",
  "variables_identified": [
    {"name": "has_receipt", "type": "boolean"},
    {"name": "days_since_purchase", "type": "numeric", "unit": "days"}
  ],
  "actions_identified": ["offer_refund"],
  "confidence": 0.92,
  "notes": "Converted 'typically get processed smoothly' to explicit IF-THEN. Interpreted 'three weeks' as 21 days."
}
```

---

## 3. Pass 7: Variable Canonicalization Prompt

### Purpose
Map extracted variables to canonical schema, ensuring consistency across policies and documents.

### Prompt Template

```python
VARIABLE_CANONICALIZATION_PROMPT = """
Map the extracted variables from this policy to canonical variable names from the schema.

Policy Text:
{policy_text}

Extracted Conditions (from extraction pipeline):
{extracted_conditions}

Current Canonical Variable Schema:
{schema_json}

TASK:
For each extracted variable, determine:
1. Does it match an EXISTING canonical variable (same semantic meaning)?
   - If YES: Map to that canonical name
   - If NO: Propose a NEW canonical name

2. For NEW variables:
   - Use snake_case: lowercase_with_underscores
   - Be descriptive: days_since_purchase (not days or purchase_date)
   - Match domain conventions: refund_, shipping_, privacy_, etc.

MATCHING RULES:
- "purchase_age" matches "days_since_purchase" (same concept)
- "receipt_present" matches "has_receipt" (same concept)
- "amount" in refund context matches "refund_amount"
- "damaged" matches "physical_damage"
- Look at synonyms in schema to find matches

OUTPUT FORMAT:
{
  "mappings": [
    {
      "extracted": "purchase_age",
      "canonical": "days_since_purchase",
      "is_new": false,
      "confidence": 0.98,
      "reason": "Same concept as existing variable"
    },
    {
      "extracted": "severe_weather",
      "canonical": "severe_weather_alert",
      "is_new": true,
      "confidence": 1.0,
      "reason": "New variable not in schema, proposed canonical name"
    }
  ],
  "new_variables": [
    {
      "canonical_name": "severe_weather_alert",
      "type": "boolean",
      "domain": "shipping",
      "description": "Whether a severe weather alert is currently active in the region",
      "synonyms": ["weather_alert_active", "severe_weather", "extreme_weather_flag"],
      "unit": null
    }
  ],
  "schema_update_summary": "Added 1 new variable (severe_weather_alert). Mapped 3 existing variables."
}

IMPORTANT:
- Be consistent: if "has_receipt" exists, use it (not "receipt_present")
- Preserve semantic meaning: don't force-fit variables into wrong canonical names
- When uncertain, propose new canonical name with good description

Output ONLY valid JSON.
"""
```

### Expected Output Example

```json
{
  "mappings": [
    {
      "extracted": "has_receipt",
      "canonical": "has_receipt",
      "is_new": false,
      "confidence": 1.0,
      "reason": "Exact match with existing schema variable"
    },
    {
      "extracted": "purchase_age_days",
      "canonical": "days_since_purchase",
      "is_new": false,
      "confidence": 0.95,
      "reason": "Synonym match - same semantic concept"
    },
    {
      "extracted": "item_damaged",
      "canonical": "physical_damage",
      "is_new": false,
      "confidence": 0.92,
      "reason": "Synonym match via 'damaged' in schema synonyms"
    },
    {
      "extracted": "severe_weather",
      "canonical": "severe_weather_alert",
      "is_new": true,
      "confidence": 0.88,
      "reason": "New concept not in schema"
    }
  ],
  "new_variables": [
    {
      "canonical_name": "severe_weather_alert",
      "type": "boolean",
      "domain": "shipping",
      "description": "Whether severe weather alert is currently active",
      "synonyms": ["weather_alert_active", "severe_weather", "extreme_weather_flag"],
      "unit": null
    }
  ],
  "schema_update_summary": "Added 1 new variable. Mapped 3 existing (100% match), 1 synonym match."
}
```

---

## 4. Conflict-Aware Policy Discovery (Enhanced Pass 0)

### Purpose
Identify potential conflicts during discovery phase for better testing.

### Enhanced Prompt Addition

```python
# Add this section to the Policy Discovery prompt:

CONFLICT_DETECTION_SECTION = """
ADDITIONALLY, identify potential conflicts between policies:

Look for:
1. **Overlapping Conditions with Different Actions**
   - Same domain, similar conditions, different outcomes
   - Example: POL-A says "offer_refund if days<=7", POL-B says "deny_refund if days<=14"

2. **Priority Conflicts**
   - Lower priority policy contradicts higher priority policy
   - Example: Department policy overrides company policy

3. **Missing Escalation Paths**
   - Conflicting actions with same priority level
   - Requires human review/escalation

For each conflict candidate pair, add:
{
  "conflict_type": "overlapping_conditions" | "priority_mismatch" | "action_conflict",
  "policy_ids": ["POL-A", "POL-B"],
  "conflict_description": "Brief explanation of conflict",
  "requires_escalation": true/false
}

Example:
{
  "conflict_type": "overlapping_conditions",
  "policy_ids": ["POL-RETURNS-004", "POL-RETURNS-009"],
  "conflict_description": "Both policies govern refunds with receipt, but have different time windows (7 days vs 14 days). Customer within 8-14 days would match POL-009 but not POL-004.",
  "requires_escalation": false,
  "resolution_hint": "POL-009 likely supersedes POL-004 (longer window is more permissive)"
}
"""
```

---

## 5. Complete Pipeline Orchestrator

### Purpose
Coordinate all passes with shared schema.

### Pseudocode Example

```python
class PolicyExtractionOrchestrator:
    """
    Orchestrates multi-policy extraction with variable canonicalization.
    """
    
    def __init__(self, llm_client: LLMClient, schema_path: str = "variable_schema.json"):
        self.llm = llm_client
        self.schema = self.load_or_create_schema(schema_path)
        self.schema_path = schema_path
    
    def extract_document(self, document_path: str) -> List[Policy]:
        """
        Extract all policies from document using LLM-based pipeline.
        """
        document_text = self.load_document(document_path)
        
        # ═══════════════════════════════════════════════════════════
        # PASS 0: Policy Discovery
        # ═══════════════════════════════════════════════════════════
        print("Pass 0: Discovering policies...")
        discovery_result = self.llm.invoke_json(
            POLICY_DISCOVERY_PROMPT.format(document_text=document_text),
            schema=PolicyDiscoveryResponse
        )
        
        print(f"  Found {len(discovery_result['policies'])} policies")
        
        extracted_policies = []
        
        # Process each policy independently
        for policy_segment in discovery_result["policies"]:
            print(f"\nProcessing {policy_segment['policy_id']}...")
            
            # ═══════════════════════════════════════════════════════
            # PASS 0.5: Normalization (implicit → explicit)
            # ═══════════════════════════════════════════════════════
            print("  Pass 0.5: Normalizing...")
            normalized_result = self.llm.invoke_json(
                NORMALIZATION_PROMPT.format(policy_text=policy_segment["text_span"]),
                schema=NormalizationResponse
            )
            normalized_text = normalized_result["normalized_policy"]
            
            # ═══════════════════════════════════════════════════════
            # PASS 1-6: Existing Extraction Pipeline
            # ═══════════════════════════════════════════════════════
            print("  Pass 1-6: Extracting conditions/actions...")
            raw_policy = self.run_extraction_passes(
                policy_text=normalized_text,
                policy_id=policy_segment["policy_id"],
                domain=policy_segment["domain"],
                priority=policy_segment["priority_level"]
            )
            
            # ═══════════════════════════════════════════════════════
            # PASS 7: Variable Canonicalization
            # ═══════════════════════════════════════════════════════
            print("  Pass 7: Canonicalizing variables...")
            canon_result = self.llm.invoke_json(
                VARIABLE_CANONICALIZATION_PROMPT.format(
                    policy_text=normalized_text,
                    extracted_conditions=json.dumps(raw_policy.conditions),
                    schema_json=self.schema.to_json()
                ),
                schema=VariableCanonicalizationResponse
            )
            
            # Apply mappings to policy
            canonical_policy = self.apply_variable_mappings(
                raw_policy, 
                canon_result["mappings"]
            )
            
            # Update global schema with new variables
            for new_var in canon_result["new_variables"]:
                self.schema.add_variable(VariableDefinition(**new_var))
                print(f"    + Added new variable: {new_var['canonical_name']}")
            
            extracted_policies.append(canonical_policy)
        
        # Save updated schema
        self.schema.save(self.schema_path)
        print(f"\n✓ Extracted {len(extracted_policies)} policies")
        print(f"✓ Schema updated with {len(canon_result.get('new_variables', []))} new variables")
        
        return extracted_policies
```

---

## Testing the Prompts

### Quick Test Script

```python
# Test Pass 0 on stage1_explicit document
def test_policy_discovery():
    with open("synthetic_data/stage1_explicit/documents/doc_001.md") as f:
        doc = f.read()
    
    result = llm_client.invoke_json(
        POLICY_DISCOVERY_PROMPT.format(document_text=doc)
    )
    
    print(f"Found {result['total_policies_found']} policies:")
    for p in result["policies"]:
        print(f"  - {p['policy_id']}: {p['policy_summary']}")
    
    # Expected: 4 policies (POL-RETURNS-004, POL-PRIVACY-001, POL-RETURNS-009, POL-SHIP-008)
    assert len(result["policies"]) == 4, "Should find 4 distinct policies"
```

---

## Prompt Engineering Tips

1. **Be Explicit**: Specify exactly what format you want (JSON schema)
2. **Give Examples**: Show input→output transformations
3. **Handle Edge Cases**: Address implicit language, missing IDs, etc.
4. **Request Confidence**: Ask LLM to rate its confidence on mappings
5. **Iterative Refinement**: Test on synthetic data, adjust prompts based on errors

---

## Next Steps

1. Implement Pass 0 with this prompt
2. Test on stage1_explicit/doc_001.md (should find 4 policies)
3. Implement Pass 0.5 with normalization prompt
4. Test on stage3_implicit/doc_001.md (should extract policies)
5. Build variable schema system
6. Implement Pass 7 with canonicalization prompt
7. Re-run full pipeline test

Expected outcome: All 4 test stages should pass with correct policy counts and meaningful conflict detection.
