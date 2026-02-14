"""Z3 helpers for constraint encoding — ported from Validation.z3_utils."""
from typing import Any, Dict, List

from z3 import Bool, Int, Real, String, Solver, sat

from .schemas import VariableSchema


def z3_var(name: str, vtype: str):
    """Create a Z3 variable from a schema type string."""
    if vtype == "bool":
        return Bool(name)
    if vtype == "int":
        return Int(name)
    if vtype == "float":
        return Real(name)
    if vtype == "enum":
        return String(name)
    raise ValueError(f"Unsupported type: {vtype}")


def encode_test(z3v, test: Dict[str, Any]):
    """Encode a single {op, value} test as a Z3 constraint."""
    op = test["op"]
    val = test["value"]

    if op == "==":
        return z3v == val
    if op == "!=":
        return z3v != val
    if op == "<=":
        return z3v <= val
    if op == ">=":
        return z3v >= val
    if op == ">":
        return z3v > val
    if op == "<":
        return z3v < val
    raise ValueError(f"Unsupported operator: {op}")


def encode_path(solver: Solver, path: List[Dict[str, Any]], z3vars: Dict[str, Any]):
    """Add all conditions in a compiled path to the solver."""
    for step in path:
        var = step["var"]
        for test in step["tests"]:
            solver.add(encode_test(z3vars[var], test))


def build_z3_vars(variables: Dict[str, VariableSchema]) -> Dict[str, Any]:
    """Build Z3 variable dict from the bundle's variable schema."""
    return {name: z3_var(name, info.type) for name, info in variables.items()}


def normalize_action(action_type: str, action_value: Any) -> str:
    """Normalize to 'type:value' format."""
    return f"{action_type}:{action_value}"
