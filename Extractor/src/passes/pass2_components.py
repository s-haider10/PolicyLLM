"""Pass 2: extract structured scope/conditions/actions/exceptions components."""
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

VALID_OPERATORS = Literal[
    "==", "!=", ">", "<", ">=", "<=", "in", "not_in",
    "boolean_true", "boolean_false",
]

COMPONENT_PROMPT = """You are a policy extraction assistant. Extract structured policy components as valid JSON.

CRITICAL: Respond with ONLY valid JSON. Escape all quotes in text values using backslash.

Example response format:
{
  "scope": {
    "customer_segments": ["all"],
    "product_categories": ["electronics"],
    "channels": [],
    "regions": []
  },
  "conditions": [
    {
      "type": "time_window",
      "value": 14,
      "unit": "days",
      "operator": "<=",
      "target": "purchase_date",
      "parameter": "days_since_purchase",
      "source_text": "purchased within 14 days"
    }
  ],
  "actions": [
    {
      "type": "required",
      "action": "full_refund",
      "requires": ["receipt"],
      "source_text": "eligible for a full refund"
    }
  ],
  "exceptions": []
}

Condition operator examples:

Input: "Refund available within 14 days of purchase"
Output condition: {"type": "time_window", "value": 14, "unit": "days", "operator": "<=", "target": "purchase_date", "parameter": "days_since_purchase", "source_text": "within 14 days of purchase"}

Input: "Orders above $100 qualify for free shipping"
Output condition: {"type": "amount_threshold", "value": 100, "unit": "dollars", "operator": ">=", "target": "order_total", "parameter": "order_amount", "source_text": "orders above $100"}

Input: "Customer must provide valid ID"
Output condition: {"type": "boolean_flag", "value": true, "operator": "boolean_true", "target": "valid_id", "parameter": "has_valid_id", "source_text": "must provide valid ID"}

Input: "Excludes hazardous materials"
Output condition: {"type": "product_category", "value": "hazardous", "operator": "not_in", "target": "product_type", "parameter": "product_category", "source_text": "excludes hazardous materials"}

Field specifications:
- scope: Object with arrays of strings (customer_segments, product_categories, channels, regions)
- conditions: Array of objects with type, value, unit, operator, target, parameter, source_text
  * Condition types: time_window, amount_threshold, customer_tier, product_category, geographic, boolean_flag, role_requirement, other
  * The operator field MUST be one of: ==, !=, >, <, >=, <=, in, not_in, boolean_true, boolean_false.
    Never use "unknown" or null for operator. Infer the operator from context:
    - "within X days" -> "<="
    - "at least / above / more than / minimum" -> ">="
    - "must have / requires / must provide" -> "boolean_true"
    - "excludes / not allowed / prohibited" -> "not_in"
    - "equals / exactly / is" -> "=="
    - "does not have / without" -> "boolean_false"
- actions: Array of objects with type, action, requires (array), source_text
  * Action types: required, prohibited, fallback, conditional, discovered_pattern, other
- exceptions: Array of objects with description, source_text

Rules:
1. Use lowercase for all string values
2. Escape quotes in text using backslash: \\"text\\"
3. If a field is empty, use empty array [] or null. Never set operator to "unknown" or null.
4. Do not hallucinate - only extract what is explicitly stated
5. Keep source_text concise (max 100 chars)

Respond with ONLY the JSON object."""


class ScopeModel(BaseModel):
    customer_segments: List[str] = Field(default_factory=list)
    product_categories: List[str] = Field(default_factory=list)
    channels: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)


class ConditionModel(BaseModel):
    type: str
    value: Optional[Union[float, int, str, bool]] = None
    unit: Optional[str] = None
    operator: Optional[VALID_OPERATORS] = None
    target: Optional[str] = None
    parameter: Optional[str] = None
    source_text: Optional[str] = None


class ActionModel(BaseModel):
    type: str
    action: str
    requires: List[str] = Field(default_factory=list)
    source_text: Optional[str] = None


class ExceptionModel(BaseModel):
    description: str
    source_text: Optional[str] = None


class ComponentsModel(BaseModel):
    scope: ScopeModel
    conditions: List[ConditionModel] = Field(default_factory=list)
    actions: List[ActionModel] = Field(default_factory=list)
    exceptions: List[ExceptionModel] = Field(default_factory=list)


def _empty_scope() -> Dict[str, List[str]]:
    return {
        "customer_segments": [],
        "product_categories": [],
        "channels": [],
        "regions": [],
    }


def _infer_operator(condition: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback: infer operator from type when missing or 'unknown'."""
    op = condition.get("operator")
    if op is not None and op != "unknown":
        return condition
    ctype = condition.get("type", "")
    source = (condition.get("source_text") or "").lower()
    if ctype == "boolean_flag":
        condition["operator"] = "boolean_true"
    elif ctype == "time_window":
        condition["operator"] = "<="
    elif ctype in ("amount_threshold", "numeric_threshold", "monetary"):
        if re.search(r"at least|minimum|above|more than|exceeds|over", source):
            condition["operator"] = ">="
        else:
            condition["operator"] = "<="
    elif ctype == "product_category":
        if re.search(r"exclud|not allowed|prohibited|except", source):
            condition["operator"] = "not_in"
        else:
            condition["operator"] = "in"
    elif ctype == "customer_tier":
        condition["operator"] = "=="
    elif op is None or op == "unknown":
        condition["operator"] = "=="
    return condition


def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    if "scope" not in result or not isinstance(result["scope"], dict):
        result["scope"] = _empty_scope()
    else:
        for key in ["customer_segments", "product_categories", "channels", "regions"]:
            result["scope"].setdefault(key, [])
    for key in ["conditions", "actions", "exceptions"]:
        if key not in result or not isinstance(result[key], list):
            result[key] = []
    # Post-extraction operator inference for conditions
    result["conditions"] = [_infer_operator(c) if isinstance(c, dict) else c for c in result["conditions"]]
    return result


def run(section: Dict[str, Any], llm_client: Any) -> Dict[str, Any]:
    heading = section.get("heading") or ""
    paras = [p.get("text", "") if isinstance(p, dict) else str(p) for p in section.get("paragraphs", [])]
    text = "\n\n".join(paras)
    prompt = f"{COMPONENT_PROMPT}\n\nHeading: {heading}\n\nText:\n{text}"
    result = llm_client.invoke_json(prompt, schema=ComponentsModel)
    return _normalize(result)
