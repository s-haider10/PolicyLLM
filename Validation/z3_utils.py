"""Shared Z3 utilities for constraint encoding and solving."""
from typing import Any, Dict, List

from z3 import Bool, Int, Real, String, Solver, sat


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


def build_z3_vars(schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build a dict of Z3 variables from a variable schema."""
    return {name: z3_var(name, info["type"]) for name, info in schema.items()}


def solve_paths(
    path_a: List[Dict[str, Any]],
    path_b: List[Dict[str, Any]],
    schema: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Check if two paths can fire simultaneously. Returns a witness dict or None."""
    solver = Solver()
    z3vars = build_z3_vars(schema)

    encode_path(solver, path_a, z3vars)
    encode_path(solver, path_b, z3vars)

    if solver.check() != sat:
        return None

    model = solver.model()
    witness: Dict[str, Any] = {}
    for v, z3v in z3vars.items():
        val = model.eval(z3v, model_completion=True)
        if val is None:
            continue
        vtype = schema[v]["type"]
        if vtype == "bool":
            witness[v] = bool(val)
        elif vtype == "int":
            witness[v] = val.as_long()
        elif vtype == "float":
            witness[v] = float(val.numerator_as_long()) / float(val.denominator_as_long())
        elif vtype == "enum":
            witness[v] = val.as_string()
    return witness
