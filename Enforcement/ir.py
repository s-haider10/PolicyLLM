"""Z3 helpers for constraint encoding — ported from Validation.z3_utils.

Z3 is imported lazily so Enforcement can be imported even when libz3
is missing or wrong architecture (e.g. macOS); SMT checks will run when
Z3 is available or can be deferred.
"""
from typing import Any, Dict, List, Optional

from .schemas import VariableSchema

_z3: Any = None
_z3_error: Optional[Exception] = None


def _get_z3():
    """Lazy-load z3; raises if unavailable."""
    global _z3, _z3_error
    if _z3_error is not None:
        raise _z3_error
    if _z3 is not None:
        return _z3
    try:
        from z3 import Bool, Int, Real, String, Solver, sat
        _z3 = {"Bool": Bool, "Int": Int, "Real": Real, "String": String, "Solver": Solver, "sat": sat}
        return _z3
    except Exception as e:
        _z3_error = e
        raise e


def z3_var(name: str, vtype: str):
    """Create a Z3 variable from a schema type string."""
    z = _get_z3()
    if vtype == "bool":
        return z["Bool"](name)
    if vtype == "int":
        return z["Int"](name)
    if vtype == "float":
        return z["Real"](name)
    if vtype == "enum":
        return z["String"](name)
    raise ValueError(f"Unsupported type: {vtype}")


def encode_test(z3v, test: Dict[str, Any]):
    """Encode a single {op, value} test as a Z3 constraint."""
    op = test["op"]
    val = test["value"]
    if op == "exists":
        op = "=="

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


def encode_path(solver, path: List[Dict[str, Any]], z3vars: Dict[str, Any]):
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
