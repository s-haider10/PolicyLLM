"""Priority resolution: resolve conflicts using a priority lattice."""
from typing import Any, Dict, List, Set, Tuple

PRIORITY_RANK: Dict[str, int] = {
    "regulatory": 1,
    "core_values": 2,
    "company": 3,
    "department": 4,
    "situational": 5,
}


def _normalize_priority(meta: Dict[str, Any]) -> str:
    """Map metadata to a canonical priority level."""
    if meta.get("regulatory_linkage"):
        return "regulatory"
    p = (meta.get("priority") or "company").lower().strip()
    if p in PRIORITY_RANK:
        return p
    ALIASES = {
        "legal": "regulatory", "law": "regulatory", "reg": "regulatory",
        "values": "core_values", "ethics": "core_values", "privacy": "core_values", "safety": "core_values",
        "dept": "department", "team": "department",
        "promo": "situational", "temporary": "situational",
    }
    return ALIASES.get(p, "company")


def _rank(meta: Dict[str, Any]) -> int:
    return PRIORITY_RANK[_normalize_priority(meta)]


def _owner_of(meta: Dict[str, Any]) -> str:
    return meta.get("owner", "unknown_owner")


def _action_relation(a1: str, a2: str) -> str:
    """Determine if two conflicting actions should compose or override."""
    if ("approval" in a1 and "refund" in a2) or ("approval" in a2 and "refund" in a1):
        return "compose"
    return "override"


def _evidence_of(conf: Dict[str, Any]) -> Dict[str, Any]:
    if "witness" in conf:
        return {"witness": conf["witness"]}
    if "witness_note" in conf:
        return {"note": conf["witness_note"]}
    if "note" in conf:
        return {"note": conf["note"]}
    return {}


def resolve_conflicts(
    conflict_report: Dict[str, Any],
    decision_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve conflicts using the priority lattice.

    Args:
        conflict_report: Output from detect_conflicts().
        decision_graph: Output from build_decision_graph().

    Returns:
        Resolution report with auto_resolutions, escalations, dominance_rules.
    """
    pid_to_meta: Dict[str, Dict[str, Any]] = {
        p["policy_id"]: p.get("metadata", {})
        for p in decision_graph["compiled_paths"]
    }

    auto_resolutions: List[Dict[str, Any]] = []
    escalations: List[Dict[str, Any]] = []
    dominance_rules: List[Dict[str, Any]] = []
    seen_rules: Set[Tuple] = set()

    def _resolve_pair(conf: Dict[str, Any], conf_type: str):
        p1, p2 = conf["policies"]
        a1, a2 = conf["actions"]
        meta1 = conf.get("metadata", {}).get("p1") or pid_to_meta.get(p1, {})
        meta2 = conf.get("metadata", {}).get("p2") or pid_to_meta.get(p2, {})
        pr1, pr2 = _rank(meta1), _rank(meta2)
        rel = _action_relation(a1, a2)

        if conf_type == "semantic":
            escalations.append({
                "conflict_type": "semantic",
                "policies": [p1, p2],
                "actions": [a1, a2],
                "priority": f"{_normalize_priority(meta1)}|{_normalize_priority(meta2)}",
                "owners_to_notify": sorted({_owner_of(meta1), _owner_of(meta2)}),
                "evidence": _evidence_of(conf),
                "recommended_next_step": "llm_validation_or_human_review",
            })
            return

        if pr1 != pr2:
            winner = p1 if pr1 < pr2 else p2
            loser = p2 if winner == p1 else p1
            win_meta = meta1 if winner == p1 else meta2
            lose_meta = meta2 if winner == p1 else meta1

            auto_resolutions.append({
                "conflict_type": "logical",
                "policies": [p1, p2],
                "winner": winner,
                "loser": loser,
                "winner_priority": _normalize_priority(win_meta),
                "loser_priority": _normalize_priority(lose_meta),
                "action_relation": rel,
                "rationale": "priority_lattice",
                "evidence": _evidence_of(conf),
            })

            key = (tuple(sorted([p1, p2])), winner, rel)
            if key not in seen_rules:
                seen_rules.add(key)
                dominance_rules.append({
                    "when": {"policies_fire": sorted([p1, p2])},
                    "then": {
                        "mode": "compose" if rel == "compose" else "override",
                        "enforce": winner,
                        "notes": (
                            "compose: treat approval as gating step before refund"
                            if rel == "compose"
                            else "override: winner action replaces loser action"
                        ),
                    },
                })
            return

        # Same priority — escalate
        escalations.append({
            "conflict_type": "logical",
            "policies": [p1, p2],
            "actions": [a1, a2],
            "priority": _normalize_priority(meta1),
            "owners_to_notify": sorted({_owner_of(meta1), _owner_of(meta2)}),
            "evidence": _evidence_of(conf),
            "recommended_next_step": "human_review",
        })

    for conf in conflict_report.get("logical_conflicts", []):
        _resolve_pair(conf, "logical")

    for conf in conflict_report.get("semantic_conflicts", []):
        conf2 = dict(conf)
        p1, p2 = conf2["policies"]
        conf2["metadata"] = {"p1": pid_to_meta.get(p1, {}), "p2": pid_to_meta.get(p2, {})}
        _resolve_pair(conf2, "semantic")

    return {
        "module": "2F_priority_resolution",
        "priority_lattice": PRIORITY_RANK,
        "auto_resolutions": auto_resolutions,
        "escalations": escalations,
        "conflict_free_plan": {
            "dominance_rules": dominance_rules,
            "notes": [
                "Logical conflicts are resolved deterministically by priority lattice.",
                "Same-priority logical conflicts are escalated to policy owners with evidence.",
                "Semantic conflicts are not auto-resolved; they require LLM validation or human review.",
                "If action_relation=compose, downstream enforcement should treat approval as a gating step for refund.",
            ],
        },
        "stats": {
            "num_auto_resolutions": len(auto_resolutions),
            "num_escalations": len(escalations),
            "num_dominance_rules": len(dominance_rules),
        },
        "reproducibility": {
            "deterministic": True,
        },
    }
