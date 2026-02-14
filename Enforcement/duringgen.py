"""During-generation: scaffold serialization and prompt injection."""
from typing import Any, Dict, List

from .schemas import (
    CompiledPath,
    CompiledPolicyBundle,
    Constraint,
    DominanceRule,
    EnforcementContext,
    InjectionBundle,
    VariableSchema,
)

PRIORITY_ORDER_TEXT = "PRIORITY: regulatory > core_values > company > department > situational."


def serialize_constraints(constraints: List[Constraint]) -> str:
    """Convert constraints into a numbered invariant block."""
    if not constraints:
        return ""
    lines = ["- INVARIANTS:"]
    for i, c in enumerate(constraints, 1):
        text = c.constraint
        if text.startswith("NOT(") and text.endswith(")"):
            inner = text[4:-1].replace("_", " ")
            lines.append(f"  {i}) NEVER {inner}.")
        else:
            lines.append(f"  {i}) ALWAYS comply with: {text}.")
    return "\n".join(lines)


def serialize_scaffold(
    paths: List[CompiledPath],
    variables: Dict[str, VariableSchema],
    decision_nodes: List[str],
    dominance_applied: List[DominanceRule],
) -> List[str]:
    """Convert compiled paths into deterministic step-by-step scaffold instructions.

    Variables are processed in decision_nodes order. Within each variable,
    paths are processed in policy_id alphabetical order.
    """
    if not paths:
        return []

    # Collect variables that appear in paths, ordered by decision_nodes
    path_vars = set()
    for p in paths:
        for step in p.path:
            path_vars.add(step.var)

    ordered_vars = [v for v in decision_nodes if v in path_vars]
    # Add any vars not in decision_nodes at the end
    for v in sorted(path_vars):
        if v not in ordered_vars:
            ordered_vars.append(v)

    steps: List[str] = []
    step_num = 1
    sorted_paths = sorted(paths, key=lambda p: p.policy_id)

    for var in ordered_vars:
        vschema = variables.get(var)
        vtype = vschema.type if vschema else "unknown"

        if vtype == "bool":
            steps.append(
                f"STEP {step_num}: Check variable {var}. "
                f"If unknown, ask the user; DO NOT assume."
            )
        elif vtype == "enum":
            vals = vschema.values if vschema and vschema.values else []
            vals_str = ", ".join(str(v) for v in vals) if vals else "unknown"
            steps.append(
                f"STEP {step_num}: Determine {var}. Must be one of: {vals_str}."
            )
        else:
            steps.append(f"STEP {step_num}: Check {var}.")
        step_num += 1

        # Add conditional branches from paths that reference this variable
        for p in sorted_paths:
            for path_step in p.path:
                if path_step.var != var:
                    continue
                for test in path_step.tests:
                    op = test.get("op", "==")
                    val = test.get("value", "?")
                    source = p.metadata.source
                    eff = p.metadata.eff_date or "N/A"
                    steps.append(
                        f"  If {var} {op} {val} THEN ACTION => {p.leaf_action} "
                        f"(per {p.policy_id}, source: {source}, eff_date: {eff})."
                    )

    # Dominance notes
    if dominance_applied:
        for dr in dominance_applied:
            mode = dr.then.get("mode", "override")
            enforced = dr.then.get("enforce", "")
            notes = dr.then.get("notes", "")
            steps.append(
                f"NOTE: When policies {dr.when.get('policies_fire', [])} conflict, "
                f"mode={mode}, enforce={enforced}. {notes}"
            )

    steps.append(
        f"STEP {step_num}: FINAL — State the action and cite the policy source."
    )
    return steps


def build_injection_bundle(
    context: EnforcementContext,
    bundle: CompiledPolicyBundle,
) -> InjectionBundle:
    """Full during-gen pipeline: produce the injection bundle."""
    constraints_block = serialize_constraints(context.applicable_constraints)
    scaffold = serialize_scaffold(
        context.applicable_paths,
        bundle.variables,
        bundle.decision_nodes,
        context.dominance_applied,
    )

    priority_guidance = PRIORITY_ORDER_TEXT
    if context.dominance_applied:
        for dr in context.dominance_applied:
            enforced = dr.then.get("enforce", "")
            priority_guidance += f"\nEnforce {enforced} when in conflict."

    system_additions = ""
    if constraints_block or priority_guidance:
        system_additions = (
            "---BEGIN POLICY ENFORCEMENT---\n"
            f"{constraints_block}\n"
            f"- {priority_guidance}\n"
            "---END POLICY ENFORCEMENT---"
        )

    return InjectionBundle(
        system_prompt_additions=system_additions,
        scaffold_steps=scaffold,
        priority_guidance=priority_guidance,
        invariant_constraints=[c.constraint for c in context.applicable_constraints],
        generation_params={"temperature": 0.0, "max_tokens": 2048},
    )


def format_full_prompt(
    user_query: str,
    injection: InjectionBundle,
    base_system_prompt: str = "",
) -> Dict[str, str]:
    """Assemble the final prompt dict for the LLM."""
    system = base_system_prompt
    if injection.system_prompt_additions:
        system = system + "\n\n" + injection.system_prompt_additions if system else injection.system_prompt_additions

    user = user_query
    if injection.scaffold_steps:
        scaffold_text = "\n".join(injection.scaffold_steps)
        user = f"{user_query}\n\nFollow the enforcement scaffold below:\n{scaffold_text}"

    return {"system": system, "user": user}
