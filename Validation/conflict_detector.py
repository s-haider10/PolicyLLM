"""Z3-based conflict detection between compiled policy paths."""
import itertools
from typing import Any, Dict, List

from .z3_utils import solve_paths


def detect_conflicts(
    decision_graph: Dict[str, Any],
    policy_ir: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect logical conflicts between policy paths using Z3.

    For each pair of paths with different leaf actions, checks whether
    there exists an input assignment that fires both simultaneously.

    Args:
        decision_graph: Output from build_decision_graph().
        policy_ir: The intermediate representation (needed for variable schema).

    Returns:
        Conflict report dict with logical_conflicts, stats, and reproducibility info.
    """
    paths = decision_graph["compiled_paths"]
    schema = policy_ir["variables"]

    logical_conflicts: List[Dict[str, Any]] = []

    for p1, p2 in itertools.combinations(paths, 2):
        if p1["leaf_action"] == p2["leaf_action"]:
            continue

        witness = solve_paths(p1["path"], p2["path"], schema)
        if witness is not None:
            logical_conflicts.append({
                "type": "logical",
                "policies": [p1["policy_id"], p2["policy_id"]],
                "actions": [p1["leaf_action"], p2["leaf_action"]],
                "witness": witness,
                "metadata": {
                    "p1": p1["metadata"],
                    "p2": p2["metadata"],
                },
            })

    return {
        "module": "2E_conflict_detection_z3",
        "logical_conflicts": logical_conflicts,
        "stats": {
            "num_policies": len(paths),
            "logical_conflicts": len(logical_conflicts),
        },
        "reproducibility": {
            "engine": "z3",
            "complete": True,
            "deterministic": True,
        },
    }
