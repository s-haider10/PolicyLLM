"""Decision graph generation: compile policy_ir into ordered decision paths."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Atom:
    var: str
    op: str
    value: Any


@dataclass
class Rule:
    policy_id: str
    conditions: List[Atom]
    action_type: str
    action_value: Any
    metadata: Dict[str, Any]


def parse_rules(policy_ir: Dict[str, Any]) -> List[Rule]:
    """Parse conditional_rules from policy_ir into Rule objects."""
    rules: List[Rule] = []
    for r in policy_ir["conditional_rules"]:
        rules.append(
            Rule(
                policy_id=r["policy_id"],
                conditions=[Atom(c["var"], c["op"], c["value"]) for c in r["conditions"]],
                action_type=r["action"]["type"],
                action_value=r["action"]["value"],
                metadata=r["metadata"],
            )
        )
    return rules


def _variable_priority(var: str, schema: Dict[str, Any], freq: Counter) -> Tuple[int, int, str]:
    """Sort key: bool first, then enum, then numeric. Ties broken by frequency desc."""
    t = schema[var]["type"]
    bucket = 0 if t == "bool" else 1 if t == "enum" else 2
    return (bucket, -freq[var], var)


def normalize_action(action_type: str, action_value: Any) -> str:
    """Normalize to 'type:value' format."""
    return f"{action_type}:{action_value}"


def _compile_path(rule: Rule, ordered_decisions: List[str]) -> List[Dict[str, Any]]:
    """Compile a single rule's conditions into an ordered path."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in rule.conditions:
        grouped[a.var].append({"op": a.op, "value": a.value})
    return [{"var": v, "tests": grouped[v]} for v in ordered_decisions if v in grouped]


def build_decision_graph(policy_ir: Dict[str, Any]) -> Dict[str, Any]:
    """Build a complete decision graph from a policy_ir dict.

    Args:
        policy_ir: The intermediate representation with variables, conditional_rules, constraints.

    Returns:
        Decision graph dict with decision_nodes, node_schema, leaf_actions, compiled_paths.
    """
    rules = parse_rules(policy_ir)
    schema = policy_ir["variables"]

    # Collect and order decision variables
    decision_vars: List[str] = []
    for r in rules:
        for c in r.conditions:
            if c.var not in decision_vars:
                decision_vars.append(c.var)

    freq = Counter(c.var for r in rules for c in r.conditions)
    ordered_decisions = sorted(decision_vars, key=lambda v: _variable_priority(v, schema, freq))

    leaf_actions = sorted(set(normalize_action(r.action_type, r.action_value) for r in rules))

    compiled_paths = [
        {
            "policy_id": r.policy_id,
            "path": _compile_path(r, ordered_decisions),
            "leaf_action": normalize_action(r.action_type, r.action_value),
            "metadata": r.metadata,
        }
        for r in rules
    ]

    return {
        "module": "2D_decision_graph_generation",
        "decision_nodes": ordered_decisions,
        "node_schema": {v: schema[v] for v in ordered_decisions},
        "leaf_actions": leaf_actions,
        "compiled_paths": compiled_paths,
        "excluded": {
            "constraints_count": len(policy_ir.get("constraints", []))
        },
        "reproducibility": {
            "deterministic": True
        },
    }
