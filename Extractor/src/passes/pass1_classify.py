"""Pass 1: classify sections as policy vs non-policy using an LLM."""
from typing import Any, Dict

from pydantic import BaseModel, Field

CLASSIFY_PROMPT = """You are a policy extraction assistant. Given a section with heading and text, decide if it is policy-relevant (policy/procedure/guideline) or non-policy (intro/background/definitions/marketing/legal boilerplate).

POLICY INCLUDES: Rules, procedures, guidelines, requirements, constraints, typical workflows, common practices, standard procedures.

IMPLICIT POLICY LANGUAGE: Sections with conversational language like "typically", "usually", "most often", "generally", "tend to", "come back within", "have to bring", "staff usually", etc. should be classified as policy because they describe actual procedures/workflows.

IMPORTANT: Respond with ONLY valid JSON, no other text.

Required JSON format:
{
  "is_policy": true,
  "confidence": 0.95,
  "reason": "Contains actionable refund rules described conversationally",
  "num_distinct_policies": 1
}

Fields:
- is_policy: boolean (true if policy-relevant, false otherwise)
- confidence: number between 0.0 and 1.0
- reason: string (one sentence explanation)
- num_distinct_policies: integer (count of DISTINCT policies in this section)

CRITICAL: A section may contain MULTIPLE distinct policies. Count them separately if they:
- Have different policy IDs (e.g., POL-RETURNS-004, POL-PRIVACY-001)
- Govern different domains (refund vs privacy vs shipping)
- Have conflicting conditions (7-day window vs 14-day window)
- Serve different purposes even within same domain

Examples:
- "Under POL-A, refunds within 7 days. Under POL-B, refunds within 14 days." → 2 policies
- "Returns require receipt. PII must be disclosed." → 2 policies (different domains)
- "Returns require receipt and should be within 30 days." → 1 policy
- "Customers typically bring in their receipt, and our staff have usually started verification by that point." → 1 policy (implicit procedure)
- "About three weeks is when most items make it back, though occasionally earlier." → 1 policy (implicit timeline)

Use only the provided text. Classify as policy if it describes ANY actionable rules, requirements, procedures, workflows, constraints—whether stated explicitly or conversationally/implicitly.

Respond with ONLY the JSON object."""


class ClassifyResponse(BaseModel):
    is_policy: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    num_distinct_policies: int = Field(default=0, ge=0)


def run(section: Dict[str, Any], llm_client: Any) -> Dict[str, Any]:
    """Return classification with confidence and reason."""
    heading = section.get("heading") or ""
    paras = [p.get("text", "") if isinstance(p, dict) else str(p) for p in section.get("paragraphs", [])]
    text = "\n\n".join(paras)
    prompt = f"{CLASSIFY_PROMPT}\n\nHeading: {heading}\n\nText:\n{text}"
    return llm_client.invoke_json(prompt, schema=ClassifyResponse)
