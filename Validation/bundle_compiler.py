"""Bundle compiler: merge all Validation outputs into a single compiled_policy_bundle.json."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


def compile_bundle(
    policy_ir: Dict[str, Any],
    decision_graph: Dict[str, Any],
    conflict_report: Dict[str, Any],
    resolution_report: Dict[str, Any],
    canonical_action_map: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compile all Validation outputs into the unified policy bundle.

    Args:
        policy_ir: Output of build_policy_ir().
        decision_graph: Output of build_decision_graph().
        conflict_report: Output of detect_conflicts().
        resolution_report: Output of resolve_conflicts().
        canonical_action_map: Optional list of {policy_id, canonical_actions} from schema_discovery.

    Returns:
        compiled_policy_bundle dict ready for JSON serialization.
    """
    cfp = resolution_report.get("conflict_free_plan", {})

    bundle = {
        "schema_version": "1.0",
        "variables": policy_ir.get("variables", {}),
        "conditional_rules": policy_ir.get("conditional_rules", []),
        "constraints": policy_ir.get("constraints", []),
        "decision_nodes": decision_graph.get("decision_nodes", []),
        "node_schema": decision_graph.get("node_schema", {}),
        "leaf_actions": decision_graph.get("leaf_actions", []),
        "compiled_paths": decision_graph.get("compiled_paths", []),
        "dominance_rules": cfp.get("dominance_rules", []),
        "escalations": resolution_report.get("escalations", []),
        "canonical_action_map": canonical_action_map or [],
        "priority_lattice": resolution_report.get("priority_lattice", {
            "regulatory": 1,
            "core_values": 2,
            "company": 3,
            "department": 4,
            "situational": 5,
        }),
        "bundle_metadata": {
            "generated_on": datetime.utcnow().isoformat(),
            "generator": "PolicyLLM-BundleCompiler-v1",
            "policy_count": len(policy_ir.get("conditional_rules", [])),
            "rule_count": len(policy_ir.get("conditional_rules", [])),
            "constraint_count": len(policy_ir.get("constraints", [])),
            "path_count": len(decision_graph.get("compiled_paths", [])),
        },
    }

    return bundle


def _is_z3_load_error(exc: BaseException) -> bool:
    """True if the exception is from Z3 failing to load (e.g. wrong arch, missing lib)."""
    msg = str(exc).lower()
    if "libz3" in msg or "z3exception" in msg or "incompatible architecture" in msg:
        return True
    if getattr(exc, "__module__", "") and "z3" in str(getattr(exc, "__module__", "")):
        return True
    return False


def _run_conflict_detection_subprocess(
    policy_ir: Dict[str, Any],
    decision_graph: Dict[str, Any],
    arch: str,
    project_root: str,
    python_exe: str,
) -> Dict[str, Any] | None:
    """Run conflict detection in a subprocess with the given arch. Returns conflict_report or None."""
    payload = {"policy_ir": policy_ir, "decision_graph": decision_graph}
    try:
        proc = subprocess.run(
            ["arch", arch, python_exe, "-m", "Validation.conflict_detector_runner"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _run_conflict_detection(
    decision_graph: Dict[str, Any],
    policy_ir: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Z3 conflict detection in-process, or via subprocess on arch mismatch, or return empty report."""
    num_paths = len(decision_graph.get("compiled_paths", []))
    empty_report = {
        "module": "2E_conflict_detection_z3",
        "logical_conflicts": [],
        "stats": {"num_policies": num_paths, "logical_conflicts": 0},
        "reproducibility": {"engine": "z3", "complete": False, "deterministic": True},
    }

    try:
        from .conflict_detector import detect_conflicts
        return detect_conflicts(decision_graph, policy_ir)
    except Exception as e:
        if not _is_z3_load_error(e) or platform.system() != "Darwin":
            return empty_report
        project_root = os.getcwd()
        python_exe = sys.executable
        # On macOS, try subprocess with alternate arch (e.g. arm64 when process is x86_64 under Rosetta)
        machine = platform.machine().lower()
        archs = ["arm64", "x86_64"] if machine == "arm64" else ["x86_64", "arm64"]
        for arch in archs:
            try:
                subprocess.run(["arch", arch, "true"], capture_output=True, timeout=2)
            except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                continue
            except Exception:
                continue
            report = _run_conflict_detection_subprocess(
                policy_ir, decision_graph, arch, project_root, python_exe
            )
            if report is not None:
                return report
        return empty_report


def compile_from_policies(policies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """End-to-end: policies list -> compiled_policy_bundle.

    Convenience function that runs the full Validation pipeline.
    """
    from .policy_ir_builder import build_policy_ir
    from .decision_graph import build_decision_graph
    from .resolution import resolve_conflicts

    policy_ir = build_policy_ir(policies)
    decision_graph = build_decision_graph(policy_ir)
    conflict_report = _run_conflict_detection(decision_graph, policy_ir)
    resolution_report = resolve_conflicts(conflict_report, decision_graph)

    # Build canonical action map from policies that have it
    canonical_action_map = []
    for p in policies:
        if "canonical_actions" in p:
            canonical_action_map.append({
                "policy_id": p.get("policy_id", ""),
                "canonical_actions": p["canonical_actions"],
            })

    return compile_bundle(
        policy_ir=policy_ir,
        decision_graph=decision_graph,
        conflict_report=conflict_report,
        resolution_report=resolution_report,
        canonical_action_map=canonical_action_map,
    )


def write_bundle(bundle: Dict[str, Any], output_path: str) -> None:
    """Write compiled bundle to a JSON file."""
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
