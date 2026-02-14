"""Bridge: convert Extractor policy JSONL output into the policy_ir format
required by the decision graph compiler and Z3 conflict detector."""
from datetime import date
from typing import Any, Dict, List, Optional


def _infer_variable_name(condition: Dict[str, Any]) -> Optional[str]:
    """Derive a variable name from a condition's fields."""
    ctype = condition.get("type", "")
    param = condition.get("parameter")
    target = condition.get("target")

    # Boolean flags use the parameter directly
    if ctype == "boolean_flag" and param:
        return param

    # Time windows
    if ctype == "time_window":
        return "days_since_purchase"

    # Amount thresholds
    if ctype == "amount_threshold":
        return "refund_amount"

    # Product category
    if ctype == "product_category":
        return "product_category"

    # Customer tier
    if ctype == "customer_tier":
        return "customer_tier"

    # Geographic
    if ctype == "geographic":
        return "region"

    # Role requirement
    if ctype == "role_requirement":
        return "role"

    # Fallback: use parameter or target as variable name
    if param:
        return param
    if target:
        return f"{ctype}_{target}"

    return None


def _infer_variable_type(condition: Dict[str, Any]) -> str:
    """Derive a Z3-compatible type string from a condition."""
    ctype = condition.get("type", "")
    value = condition.get("value")

    if ctype == "boolean_flag":
        return "bool"
    if ctype in ("time_window", "role_requirement"):
        return "int"
    if ctype == "amount_threshold":
        return "float"
    if ctype in ("product_category", "customer_tier", "geographic"):
        return "enum"
    # Infer from value
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "enum"
    return "enum"


def _condition_to_ir(condition: Dict[str, Any], var_name: str) -> Optional[Dict[str, Any]]:
    """Convert an Extractor condition to an IR condition triple {var, op, value}."""
    op = condition.get("operator")
    value = condition.get("value")

    # Boolean flags without explicit operator default to ==
    if condition.get("type") == "boolean_flag":
        if op is None:
            op = "=="
        if value is None:
            value = True

    if op is None or value is None:
        return None

    return {"var": var_name, "op": op, "value": value}


def _build_metadata(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from a policy dict into the IR format."""
    md = policy.get("metadata", {})
    return {
        "domain": md.get("domain", "other"),
        "priority": md.get("priority", "company"),
        "owner": md.get("owner"),
        "source": md.get("source", ""),
        "eff_date": md.get("effective_date"),
        "regulatory_linkage": md.get("regulatory_linkage", []),
    }


def build_policy_ir(policies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Transform Extractor policies into the policy_ir intermediate representation.

    Args:
        policies: List of policy dicts from Extractor JSONL output.

    Returns:
        A policy_ir dict with variables, conditional_rules, constraints, and ir_metadata.
    """
    variables: Dict[str, Dict[str, Any]] = {}
    conditional_rules: List[Dict[str, Any]] = []
    constraints: List[Dict[str, Any]] = []

    for policy in policies:
        policy_id = policy.get("policy_id", "UNKNOWN")
        conditions_raw = policy.get("conditions", [])
        actions_raw = policy.get("actions", [])
        metadata = _build_metadata(policy)

        # --- Build variables and IR conditions ---
        ir_conditions: List[Dict[str, Any]] = []

        for cond in conditions_raw:
            var_name = _infer_variable_name(cond)
            if var_name is None:
                continue

            var_type = _infer_variable_type(cond)

            # Register or update variable
            if var_name not in variables:
                desc = cond.get("source_text") or f"{cond.get('type', '')} variable"
                variables[var_name] = {"type": var_type, "description": desc}
                if var_type == "enum":
                    variables[var_name]["values"] = []

            # Collect enum values
            if var_type == "enum" and cond.get("value") is not None:
                val = cond["value"]
                if isinstance(val, str):
                    existing_values = variables[var_name].get("values", [])
                    if val not in existing_values:
                        existing_values.append(val)
                    variables[var_name]["values"] = existing_values
                # Also collect from target if it's a different value
                if cond.get("target") and cond["target"] != val:
                    target = cond["target"]
                    existing_values = variables[var_name].get("values", [])
                    if target not in existing_values:
                        existing_values.append(target)
                    variables[var_name]["values"] = existing_values

            ir_cond = _condition_to_ir(cond, var_name)
            if ir_cond:
                ir_conditions.append(ir_cond)

        # --- Build conditional rules from actions ---
        for act in actions_raw:
            act_type = act.get("type", "other")
            act_action = act.get("action", "")

            # Prohibited actions become constraints
            if act_type == "prohibited":
                constraints.append({
                    "policy_id": f"C_{policy_id}_{act_action}",
                    "constraint": f"NOT({act_action})",
                    "scope": "always",
                    "metadata": metadata,
                })
                continue

            # Skip discovered patterns that aren't validated
            discovery = policy.get("discovery")
            if act_type == "discovered_pattern":
                if isinstance(discovery, dict) and not discovery.get("human_validated", False):
                    continue

            # Build conditional rule
            if ir_conditions or act.get("requires"):
                conditional_rules.append({
                    "policy_id": policy_id,
                    "conditions": ir_conditions,
                    "action": {"type": act_action, "value": _infer_action_value(act_type)},
                    "metadata": metadata,
                })

    return {
        "variables": variables,
        "conditional_rules": conditional_rules,
        "constraints": constraints,
        "ir_metadata": {
            "generated_on": str(date.today()),
            "generator": "PolicyLLM-IRBuilder-v1",
            "notes": "Auto-generated from Extractor policy output",
        },
    }


def _infer_action_value(action_type: str) -> str:
    """Map Extractor action types to IR action values."""
    TYPE_TO_VALUE = {
        "required": "full",
        "fallback": "partial",
        "conditional": "conditional",
        "other": "unknown",
    }
    return TYPE_TO_VALUE.get(action_type, action_type)
