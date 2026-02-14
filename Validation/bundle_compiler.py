"""Bundle compiler: merge all Validation outputs into a single compiled_policy_bundle.json."""
import json
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


def compile_from_policies(policies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """End-to-end: policies list -> compiled_policy_bundle.

    Convenience function that runs the full Validation pipeline.
    """
    from .policy_ir_builder import build_policy_ir
    from .decision_graph import build_decision_graph
    from .conflict_detector import detect_conflicts
    from .resolution import resolve_conflicts

    policy_ir = build_policy_ir(policies)
    decision_graph = build_decision_graph(policy_ir)
    conflict_report = detect_conflicts(decision_graph, policy_ir)
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
