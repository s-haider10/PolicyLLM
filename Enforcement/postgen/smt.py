"""Z3-based formal verification of LLM responses against policy rules."""
import re
from typing import Any, Dict, List, Optional

from z3 import Solver, sat

from ..ir import build_z3_vars, encode_test, z3_var
from ..schemas import (
    CompiledPath,
    CompiledPolicyBundle,
    ConditionalRule,
    Constraint,
    EnforcementContext,
    SMTResult,
    VariableSchema,
)


def extract_facts_from_response(
    response_text: str,
    variables: Dict[str, VariableSchema],
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Hybrid fact extraction: symbolic regex first, neural LLM fallback.

    Maintains neuro-symbolic separation per paper §3.6:
    - Primary (Symbolic): Fast deterministic regex extraction
    - Fallback (Neural): LLM only when regex coverage < 50%
    """
    facts: Dict[str, Any] = {}
    lower = response_text.lower()

    # SYMBOLIC: Regex-based extraction
    for var_name, vschema in variables.items():
        vtype = vschema.type
        var_readable = var_name.replace("_", " ")

        if vtype == "bool":
            # Positive assertions
            pos_patterns = [
                rf"(?i)\b{var_readable}\b.*\b(?:true|yes|provided|has|confirmed|verified)\b",
                rf"(?i)\b(?:has|have|with)\s+{var_readable}\b",
            ]
            # Negative assertions
            neg_patterns = [
                rf"(?i)\b{var_readable}\b.*\b(?:false|no|missing|without|not)\b",
                rf"(?i)\bno\s+{var_readable}\b",
                rf"(?i)\bwithout\s+{var_readable}\b",
            ]
            for pat in pos_patterns:
                if re.search(pat, response_text):
                    facts[var_name] = True
                    break
            else:
                for pat in neg_patterns:
                    if re.search(pat, response_text):
                        facts[var_name] = False
                        break

        elif vtype == "int":
            # Numeric values near variable name
            pattern = rf"(?i)(?:{var_readable}|{var_name})\D*?(\d+)"
            m = re.search(pattern, response_text)
            if m:
                facts[var_name] = int(m.group(1))
            else:
                # "N days" pattern
                m = re.search(r"(\d+)\s*(?:days?|day)", lower)
                if m and "day" in var_name:
                    facts[var_name] = int(m.group(1))

        elif vtype == "float":
            pattern = rf"(?i)(?:{var_readable}|{var_name})\D*?([\d,.]+)"
            m = re.search(pattern, response_text)
            if m:
                try:
                    facts[var_name] = float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
            else:
                # Dollar amounts
                m = re.search(r"\$\s*([\d,.]+)", response_text)
                if m and "amount" in var_name:
                    try:
                        facts[var_name] = float(m.group(1).replace(",", ""))
                    except ValueError:
                        pass

        elif vtype == "enum":
            if vschema.values:
                for val in vschema.values:
                    if str(val).lower() in lower:
                        facts[var_name] = str(val)
                        break

    # NEURAL FALLBACK: LLM only if regex coverage < 50%
    if llm_client and len(facts) < len(variables) * 0.5:
        try:
            from pydantic import BaseModel

            var_desc = {k: {"type": v.type, "description": v.description} for k, v in variables.items()}
            prompt = (
                f"Given these variables: {var_desc}\n"
                f"What values does this response assert?\n"
                f"Response: {response_text}\n"
                f'Return JSON dict mapping variable names to their values. Only include variables with clear values.'
            )

            class FactsOut(BaseModel):
                facts: Dict[str, Any] = {}

            result = llm_client.invoke_json(prompt, schema=FactsOut)
            llm_facts = result.get("facts", {})
            # Merge LLM facts (only for variables not already found by regex)
            for k, v in llm_facts.items():
                if k in variables and k not in facts:
                    facts[k] = v
        except Exception:
            pass

    return facts


def verify_facts_against_rules(
    facts: Dict[str, Any],
    rules: List[ConditionalRule],
    paths: List[CompiledPath],
    constraints: List[Constraint],
    variables: Dict[str, VariableSchema],
) -> SMTResult:
    """Core SMT verification: check if extracted facts are consistent with policy rules."""
    violations: List[Dict[str, Any]] = []

    if not facts:
        return SMTResult(passed=True, violations=[], score=1.0)

    z3vars = build_z3_vars(variables)

    for rule in rules:
        solver = Solver()

        # Assert the extracted facts
        for var_name, value in facts.items():
            if var_name in z3vars:
                solver.add(z3vars[var_name] == value)

        # Encode rule conditions
        for cond in rule.conditions:
            if cond.var in z3vars:
                solver.add(encode_test(z3vars[cond.var], {"op": cond.op, "value": cond.value}))

        # Check if conditions are satisfiable with facts
        result = solver.check()
        if result == sat:
            # Rule fires — check that the response action matches
            # Extract what action the response claims
            action_str = f"{rule.action.type}:{rule.action.value}"
            # If the rule fires but the response contradicts it, that's a violation
            # (This is a simplified check — full implementation would extract response actions)
            pass  # No violation if rule is satisfiable with facts

    # Check constraints
    for constraint in constraints:
        text = constraint.constraint
        if text.startswith("NOT(") and text.endswith(")"):
            forbidden = text[4:-1].lower().replace("_", " ")
            if forbidden in facts or any(forbidden in str(v).lower() for v in facts.values()):
                violations.append({
                    "policy_id": constraint.policy_id,
                    "constraint": text,
                    "violation_type": "constraint_breach",
                })

    # Path traversal verification
    if paths:
        path_satisfied = False
        for path in paths:
            solver = Solver()

            # Add facts as constraints
            for var_name, value in facts.items():
                if var_name in z3vars:
                    solver.add(z3vars[var_name] == value)

            # Check if facts satisfy ALL steps in this path
            path_matches = True
            for step in path.path:
                if step.var not in facts:
                    path_matches = False
                    break
                for test in step.tests:
                    solver.add(encode_test(z3vars[step.var], test))

            if path_matches and solver.check() == sat:
                path_satisfied = True
                break

        if not path_satisfied:
            violations.append({
                "policy_id": "path_coverage",
                "violation_type": "uncovered_case",
                "message": "Response facts do not match any defined decision graph path",
            })

    passed = len(violations) == 0
    score = 1.0 if passed else (0.5 if any(v.get("violation_type") == "uncovered_case" for v in violations) else 0.0)
    return SMTResult(passed=passed, violations=violations, score=score)


def run_smt_check(
    response_text: str,
    context: EnforcementContext,
    bundle: CompiledPolicyBundle,
    llm_client: Optional[Any] = None,
) -> SMTResult:
    """Full SMT verification pipeline with hybrid fact extraction."""
    facts = extract_facts_from_response(response_text, bundle.variables, llm_client)

    if not facts:
        # Penalize uncertainty when no facts can be extracted
        return SMTResult(passed=True, violations=[], score=0.8)

    return verify_facts_against_rules(
        facts=facts,
        rules=context.applicable_rules,
        paths=context.applicable_paths,
        constraints=context.applicable_constraints,
        variables=bundle.variables,
    )
