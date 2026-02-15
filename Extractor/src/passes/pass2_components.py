"""Pass 2: extract structured scope/conditions/actions/exceptions components."""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

COMPONENT_PROMPT = """Extract structured policy components. Return ONLY valid JSON.

RESPONSE FORMAT (MUST be valid parseable JSON):
{
  "policies": [
    {
      "policy_id": "POL-DOMAIN-001",
      "domain": "refund",
      "scope": {
        "customer_segments": [],
        "product_categories": [],
        "channels": [],
        "regions": []
      },
      "conditions": [
        {
          "type": "time_window",
          "value": 21,
          "unit": "days",
          "operator": "<=",
          "target": "purchase_date",
          "parameter": "days_since_purchase",
          "source_text": "short text only"
        }
      ],
      "actions": [
        {
          "type": "required",
          "action": "offer_refund",
          "requires": [],
          "source_text": "short text"
        }
      ],
      "exceptions": [
        {
          "description": "exception description",
          "source_text": "short source"
        }
      ]
    }
  ]
}

RULES:
1. EXTRACT MULTIPLE policies if they exist (different domains or conditions)
2. For TEXT FIELDS: use SINGLE LINES only, NO newlines, NO quotes, NO special chars
3. If text mentions "typically", "usually", "about", "generally" - extract as implicit rule
4. source_text must fit on one line (max 100 chars), NO escaping needed
5. domain: refund, privacy, shipping, security, data_retention, customer_service, or other

Return ONLY the JSON object. No other text."""


class ScopeModel(BaseModel):
    customer_segments: List[str] = Field(default_factory=list)
    product_categories: List[str] = Field(default_factory=list)
    channels: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)


class ConditionModel(BaseModel):
    type: str
    value: Optional[Union[float, int, str, bool]] = None
    unit: Optional[str] = None
    operator: Optional[str] = None
    target: Optional[str] = None
    parameter: Optional[str] = None
    source_text: Optional[str] = None


class ActionModel(BaseModel):
    type: str
    action: str
    requires: List[str] = Field(default_factory=list)
    source_text: Optional[str] = None


class ExceptionModel(BaseModel):
    description: Optional[str] = None
    source_text: Optional[str] = None


class PolicyComponentModel(BaseModel):
    policy_id: Optional[str] = None
    domain: str = "other"
    scope: ScopeModel
    conditions: List[ConditionModel] = Field(default_factory=list)
    actions: List[ActionModel] = Field(default_factory=list)
    exceptions: List[ExceptionModel] = Field(default_factory=list)


class ComponentsModel(BaseModel):
    policies: List[PolicyComponentModel] = Field(default_factory=list)


def _empty_scope() -> Dict[str, List[str]]:
    return {
        "customer_segments": [],
        "product_categories": [],
        "channels": [],
        "regions": [],
    }


def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single policy component."""
    if "scope" not in result or not isinstance(result["scope"], dict):
        result["scope"] = _empty_scope()
    else:
        for key in ["customer_segments", "product_categories", "channels", "regions"]:
            result["scope"].setdefault(key, [])
    for key in ["conditions", "actions", "exceptions"]:
        if key not in result or not isinstance(result[key], list):
            result[key] = []
    # Ensure policy_id and domain exist
    result.setdefault("policy_id", None)
    result.setdefault("domain", "other")
    return result


def run(section: Dict[str, Any], llm_client: Any) -> List[Dict[str, Any]]:
    """Extract policy components. Returns LIST of policy component dicts."""
    heading = section.get("heading") or ""
    paras = [p.get("text", "") if isinstance(p, dict) else str(p) for p in section.get("paragraphs", [])]
    text = "\n\n".join(paras)
    prompt = f"{COMPONENT_PROMPT}\n\nHeading: {heading}\n\nText:\n{text}"
    
    result = llm_client.invoke_json(prompt, schema=ComponentsModel)
    
    # Result should be {"policies": [...]}
    policies = result.get("policies", [])
    
    # If no policies extracted, return empty list (will be filtered out by pipeline)
    if not policies:
        return []
    
    # Normalize each policy
    normalized_policies = []
    for pol in policies:
        # Convert Pydantic model to dict if needed
        if hasattr(pol, "model_dump"):
            pol = pol.model_dump()
        elif hasattr(pol, "dict"):
            pol = pol.dict()
        normalized_policies.append(_normalize(pol))
    
    return normalized_policies
