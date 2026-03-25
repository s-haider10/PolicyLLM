
**Why it's low:**

Condition F1 is measured at the individual condition level (type + operator), not the policy level. So a policy like "refund if purchase within 30 days AND receipt present AND not hygiene item" has 3 separate conditions. Your system might find the policy but miss or mistype individual conditions. From your synthetic results, the failure modes are clear: operators default to "unknown" instead of `==` or `>=`, short conditions get merged into neighboring policies, and domain typing collapses. So you're extracting conditions but losing their structure — the canonicalization to `type|operator` then penalizes every mistyped operator as a full miss.

**How to improve it:**

1. **Operator-specific few-shot exemplars** — your current few-shot uses 2 generic examples. Add exemplars that explicitly show `>=`, `==`, `<`, `boolean` operator extraction. The synthetic eval showed 0% operator accuracy — this is the lowest-hanging fruit.

2. **Condition-level extraction pass** — add a dedicated pass in your multi-pass pipeline that focuses solely on conditions after policies are identified. Right now conditions are extracted alongside everything else in Pass 2.

3. **Schema-constrained decoding** — force the LLM to output conditions in a strict JSON schema with an `operator` enum field (`==`, `>=`, `<=`, `!=`, `boolean`). This prevents the "unknown" collapse.

4. **Length-aware deduplication** — short single-condition policies get merged into neighbors. Add a rule that preserves policies below a token threshold as standalone entries instead of merging them.

5. **Stronger backbone** — GPT-4o (full) instead of GPT-4o-mini for the extraction passes specifically. Condition extraction requires more precise reasoning than policy detection. Even using the full model only for Pass 2 would help without blowing up cost.

Realistically, fixes 1 + 3 alone could push condition F1 from 0.33 to 0.50+ based on where the losses are.

The problem is in Pass 2 of your extraction pipeline. Here's what's happening concretely:

**The failure chain:**

Your policy says: *"Items may be returned within 30 days if receipt is presented."*

Your system extracts:
```json
{
  "conditions": [
    {"type": "time_window", "operator": "unknown", "value": "30"},
    {"type": "boolean_flag", "operator": "unknown", "value": "receipt"}
  ]
}
```

Ground truth expects:
```json
{
  "conditions": [
    {"type": "time_window", "operator": "<=", "value": "30"},
    {"type": "boolean_flag", "operator": "==", "value": "true"}
  ]
}
```

Your metric canonicalizes to `type|operator`. So `time_window|unknown` ≠ `time_window|<=` — that's a full miss on both conditions even though you found them. Two FPs and two FNs from a single policy where you actually got the content right.

**Fix 1: Constrain the extraction schema**

In your Pass 2 prompt, change:

```
Extract conditions as JSON with fields: type, operator, value
```

To:

```
Extract conditions as JSON. The operator field MUST be one of:
==, !=, >, <, >=, <=, in, not_in, boolean_true, boolean_false.
Never use "unknown". If the policy says "within 30 days", the
operator is "<=". If it says "must have receipt", the operator
is "boolean_true". If it says "at least $50", the operator is ">=".
```

This is the single highest-impact change — your synthetic eval showed 0% operator accuracy because every operator defaulted to "unknown."

**Fix 2: Add operator-specific few-shot exemplars**

Your current 2 exemplars (one refund, one privacy) don't demonstrate operator reasoning. Add these to Pass 2:

```
Example input: "Refund available within 14 days of purchase"
Example output: {"type": "time_window", "operator": "<=", "value": "14"}

Example input: "Orders above $100 qualify for free shipping"
Example output: {"type": "numeric_threshold", "operator": ">=", "value": "100"}

Example input: "Customer must provide valid ID"
Example output: {"type": "boolean_flag", "operator": "boolean_true", "value": "valid_id"}

Example input: "Excludes hazardous materials"
Example output: {"type": "category_exclusion", "operator": "not_in", "value": "hazardous"}
```

Four exemplars covering the four operator classes your system currently misses.

**Fix 3: Split condition extraction into its own pass**

Right now Pass 2 extracts scope, conditions, actions, and exceptions simultaneously. The LLM is doing too much and conditions get the least attention. Add a Pass 2b:

```
Pass 2a: Extract policy scope, action, exceptions (what you have now)
Pass 2b: Given the policy text AND the extracted action, extract
         ONLY the conditions with explicit type, operator, and value.
```

This costs one extra LLM call per section but isolates the hardest subtask.

**Fix 4: Post-extraction operator inference**

Add a deterministic cleanup after Pass 2 that catches the "unknown" fallback:

```python
def infer_operator(condition):
    if condition["operator"] != "unknown":
        return condition
    if condition["type"] == "boolean_flag":
        condition["operator"] = "boolean_true"
    elif condition["type"] == "time_window":
        condition["operator"] = "<="
    elif condition["type"] in ("numeric_threshold", "monetary"):
        # look for "at least", "minimum", "above" in source text
        if re.search(r"at least|minimum|above|more than", condition["source"]):
            condition["operator"] = ">="
        else:
            condition["operator"] = "<="
    return condition
```

This is a safety net that uses type-level defaults when the LLM fails. It won't be perfect but it converts 0% operator accuracy to ~70% on the common cases.

**Fix 5: Short-policy deduplication guard**

Your synthetic eval shows simple 1-condition policies being merged into neighboring multi-condition policies. In Pass 4 (deduplication), add:

```python
def should_merge(policy_a, policy_b):
    # Don't merge if either policy has only 1 condition
    if len(policy_a["conditions"]) <= 1 or len(policy_b["conditions"]) <= 1:
        return False
    return jaccard(policy_a, policy_b) > MERGE_THRESHOLD
```

This preserves short standalone rules like "Gifts: store credit only" instead of folding them into the nearest refund policy.

**Expected impact:**

Fix 1 + 2 alone (prompt changes, zero code) should move condition F1 from 0.33 to ~0.45–0.50. Adding Fix 3 and 4 should push to ~0.55–0.60. Fix 5 addresses the remaining long-tail of merged short policies. None of these require a different model or retraining.
