"""Judge LLM semantic evaluation of response compliance."""
import json
from typing import Any, List, Optional

from pydantic import BaseModel

from ..schemas import ConditionalRule, Constraint, EnforcementContext, JudgeResult

JUDGE_SYSTEM_PROMPT = """\
You are a policy compliance judge. You evaluate whether an AI assistant's response \
complies with the given policy rules and constraints. Score from 0.0 (total violation) to 1.0 (full compliance).

Evaluate on these axes:
1. Factual accuracy: Does the response state facts consistent with the policy rules?
2. Action compliance: Does the response recommend/take actions permitted by the policy?
3. Constraint adherence: Does the response violate any invariant constraints?
4. Tone and implication: Does the response imply actions beyond what the policy permits?
5. Completeness: Does the response address all required decision steps?

Respond in JSON only:
{
  "score": <float 0.0-1.0>,
  "issues": ["list of specific issues found, or empty"],
  "explanation": "brief justification"
}"""


def _format_rules(rules: List[ConditionalRule]) -> str:
    lines = []
    for r in rules:
        conds = " AND ".join(f"{c.var} {c.op} {c.value}" for c in r.conditions)
        lines.append(f"- {r.policy_id}: IF {conds} THEN {r.action.type}:{r.action.value} (source: {r.metadata.source})")
    return "\n".join(lines)


def _format_constraints(constraints: List[Constraint]) -> str:
    return "\n".join(f"- {c.constraint}" for c in constraints)


def build_judge_prompt(
    response_text: str,
    context: EnforcementContext,
) -> str:
    """Construct the judge evaluation prompt."""
    rules_text = _format_rules(context.applicable_rules)
    constraints_text = _format_constraints(context.applicable_constraints)
    return (
        f"POLICY RULES IN SCOPE:\n{rules_text}\n\n"
        f"CONSTRAINTS:\n{constraints_text}\n\n"
        f"USER QUERY:\n{context.query}\n\n"
        f"AI RESPONSE TO EVALUATE:\n{response_text}\n\n"
        f"Evaluate compliance per the scoring rubric above."
    )


class JudgeOut(BaseModel):
    score: float
    issues: List[str] = []
    explanation: str = ""


def run_judge_check(
    response_text: str,
    context: EnforcementContext,
    llm_client: Any,
) -> JudgeResult:
    """Run judge LLM evaluation.

    Uses llm_client.invoke_json() with temperature=0 for deterministic scoring.
    Falls back to a neutral score of 0.5 if the LLM call fails.
    """
    try:
        prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + build_judge_prompt(response_text, context)
        result = llm_client.invoke_json(prompt, schema=JudgeOut)
        score = max(0.0, min(1.0, float(result.get("score", 0.5))))
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        explanation = result.get("explanation", "")
        return JudgeResult(score=score, issues=issues, explanation=explanation)
    except Exception:
        return JudgeResult(score=0.5, issues=["judge_llm_unavailable"], explanation="Judge LLM call failed")
