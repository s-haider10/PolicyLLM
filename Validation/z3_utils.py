"""Shared Z3 utilities for constraint encoding and solving.

Z3 is imported lazily so that:
- Process can start even if the bundled libz3 is wrong architecture (e.g. macOS arm64 vs x86_64).
- When the Python binding fails to load libz3, we fall back to the z3 CLI (SMT-LIB2) so conflict
  detection still runs on most devices.
"""
import subprocess
import tempfile
from typing import Any, Dict, List

# Lazy Z3 import: only load when first needed; allows fallback (e.g. subprocess) on load failure
_z3 = None
_z3_cli_available: bool | None = None


def _get_z3():
    """Import and return z3 module. Raises on failure (e.g. missing lib or wrong arch)."""
    global _z3
    if _z3 is not None:
        return _z3
    try:
        from z3 import Bool, Int, Real, String, Solver, sat
        _z3 = {
            "Bool": Bool,
            "Int": Int,
            "Real": Real,
            "String": String,
            "Solver": Solver,
            "sat": sat,
        }
        return _z3
    except Exception as e:
        _z3 = False  # mark attempted so we don't retry in-process
        raise e


def _z3_cli_available_check() -> bool:
    """Return True if z3 CLI is on PATH and works (for SMT-LIB2 fallback)."""
    global _z3_cli_available
    if _z3_cli_available is not None:
        return _z3_cli_available
    try:
        r = subprocess.run(
            ["z3", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _z3_cli_available = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _z3_cli_available = False
    return _z3_cli_available


def _smt2_escape(s: str) -> str:
    """Escape string for SMT-LIB2."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _path_to_smt2_asserts(path: List[Dict[str, Any]], schema: Dict[str, Dict[str, Any]]) -> List[str]:
    """Generate SMT-LIB2 assert lines for one path."""
    lines: List[str] = []
    for step in path:
        var = step["var"]
        if var not in schema:
            continue
        vtype = schema[var]["type"]
        for test in step["tests"]:
            op = test["op"]
            val = test["value"]
            if op == "exists":
                op = "=="
            if vtype == "bool":
                vname = _canon_smt2_var(var)
                if op == "==":
                    lines.append(f'(assert (= {vname} {"true" if val else "false"}))')
                elif op == "!=":
                    lines.append(f'(assert (not (= {vname} {"true" if val else "false"})))')
                else:
                    raise ValueError(f"Unsupported bool op: {op}")
            elif vtype == "int":
                vname = _canon_smt2_var(var)
                if op == "==":
                    lines.append(f"(assert (= {vname} {int(val)}))")
                elif op == "!=":
                    lines.append(f"(assert (not (= {vname} {int(val)})))")
                elif op == "<=":
                    lines.append(f"(assert (<= {vname} {int(val)}))")
                elif op == ">=":
                    lines.append(f"(assert (>= {vname} {int(val)}))")
                elif op == "<":
                    lines.append(f"(assert (< {vname} {int(val)}))")
                elif op == ">":
                    lines.append(f"(assert (> {vname} {int(val)}))")
                else:
                    raise ValueError(f"Unsupported int op: {op}")
            elif vtype == "float":
                vname = _canon_smt2_var(var)
                fval = float(val)
                if op == "==":
                    lines.append(f"(assert (= {vname} {fval}))")
                elif op == "!=":
                    lines.append(f"(assert (not (= {vname} {fval})))")
                elif op == "<=":
                    lines.append(f"(assert (<= {vname} {fval}))")
                elif op == ">=":
                    lines.append(f"(assert (>= {vname} {fval}))")
                elif op == "<":
                    lines.append(f"(assert (< {vname} {fval}))")
                elif op == ">":
                    lines.append(f"(assert (> {vname} {fval}))")
                else:
                    raise ValueError(f"Unsupported float op: {op}")
            elif vtype == "enum":
                vname = _canon_smt2_var(var)
                sval = _smt2_escape(str(val))
                if op == "==" or op == "exists":
                    lines.append(f'(assert (= {vname} "{sval}"))')
                elif op == "!=":
                    lines.append(f'(assert (not (= {vname} "{sval}")))')
                else:
                    raise ValueError(f"Unsupported enum op: {op}")
    return lines


def _canon_smt2_var(name: str) -> str:
    """SMT-LIB2 symbol: use |name| if contains hyphen so it's valid."""
    if "-" in name or name in ("exists", "not", "and", "or", "assert", "true", "false"):
        return f"|{name}|"
    return name


def _build_smt2(path_a: List[Dict[str, Any]], path_b: List[Dict[str, Any]], schema: Dict[str, Dict[str, Any]]) -> str:
    """Build a full SMT-LIB2 script that checks path_a AND path_b satisfiability."""
    declares: List[str] = []
    for name, info in schema.items():
        vname = _canon_smt2_var(name)
        t = info["type"]
        if t == "bool":
            declares.append(f"(declare-const {vname} Bool)")
        elif t == "int":
            declares.append(f"(declare-const {vname} Int)")
        elif t == "float":
            declares.append(f"(declare-const {vname} Real)")
        elif t == "enum":
            declares.append(f'(declare-const {vname} String)')
    asserts_a = _path_to_smt2_asserts(path_a, schema)
    asserts_b = _path_to_smt2_asserts(path_b, schema)
    lines = ["(set-option :produce-models true)"] + declares + asserts_a + asserts_b + ["(check-sat)", "(get-model)"]
    return "\n".join(lines)


def _parse_z3_model(stdout: str, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse Z3 get-model output into a witness dict (var name -> value)."""
    import re
    witness: Dict[str, Any] = {}
    # Z3 model: (model (define-fun |var| () Type value) ...)
    for name in schema:
        cname = _canon_smt2_var(name)
        esc = re.escape(cname)
        t = schema[name]["type"]
        if t == "bool":
            m = re.search(rf"\(define-fun\s+{esc}\s+\(\)\s+Bool\s+(true|false)", stdout)
            if m:
                witness[name] = m.group(1) == "true"
        elif t == "int":
            m = re.search(rf"\(define-fun\s+{esc}\s+\(\)\s+Int\s+(-?\d+)", stdout)
            if m:
                witness[name] = int(m.group(1))
        elif t == "float":
            m = re.search(rf"\(define-fun\s+{esc}\s+\(\)\s+Real\s+([\d./-]+)", stdout)
            if m:
                try:
                    witness[name] = float(m.group(1))
                except ValueError:
                    pass
        elif t == "enum":
            m = re.search(rf'\(define-fun\s+{esc}\s+\(\)\s+String\s+"((?:[^"\\]|\\.)*)"', stdout)
            if m:
                witness[name] = m.group(1).replace('\\"', '"')
    return witness


def solve_paths_cli(
    path_a: List[Dict[str, Any]],
    path_b: List[Dict[str, Any]],
    schema: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Use z3 CLI (SMT-LIB2) to check if path_a and path_b can both fire. Returns witness or None."""
    if not _z3_cli_available_check():
        return None
    smt2 = _build_smt2(path_a, path_b, schema)
    try:
        r = subprocess.run(
            ["z3", "-smt2", "-in"],
            input=smt2,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or "sat" not in r.stdout:
        return None
    return _parse_z3_model(r.stdout, schema)


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


def build_z3_vars(schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build a dict of Z3 variables from a variable schema."""
    return {name: z3_var(name, info["type"]) for name, info in schema.items()}


def solve_paths(
    path_a: List[Dict[str, Any]],
    path_b: List[Dict[str, Any]],
    schema: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Check if two paths can fire simultaneously. Returns a witness dict or None.
    Uses Python z3 binding if available; otherwise falls back to z3 CLI (SMT-LIB2).
    """
    try:
        z = _get_z3()
    except Exception:
        return solve_paths_cli(path_a, path_b, schema)

    Solver = z["Solver"]
    sat = z["sat"]
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
