"""Load and validate compiled_policy_bundle.json, build in-memory indexes."""
import json
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .schemas import (
    CompiledPolicyBundle,
    CompiledPath,
    ConditionalRule,
    Constraint,
    DominanceRule,
    EscalationEntry,
)


class BundleIndex:
    """In-memory indexes over a loaded bundle for O(1) lookups."""

    def __init__(self, bundle: CompiledPolicyBundle):
        self.rules_by_domain: Dict[str, List[ConditionalRule]] = defaultdict(list)
        self.rules_by_policy_id: Dict[str, ConditionalRule] = {}
        self.paths_by_domain: Dict[str, List[CompiledPath]] = defaultdict(list)
        self.paths_by_policy_id: Dict[str, CompiledPath] = {}
        self.constraints_by_scope: Dict[str, List[Constraint]] = defaultdict(list)
        self.dominance_lookup: Dict[FrozenSet[str], DominanceRule] = {}
        self.escalation_lookup: Dict[FrozenSet[str], EscalationEntry] = {}

        for rule in bundle.conditional_rules:
            domain = rule.metadata.domain
            self.rules_by_domain[domain].append(rule)
            self.rules_by_policy_id[rule.policy_id] = rule

        for path in bundle.compiled_paths:
            domain = path.metadata.domain
            self.paths_by_domain[domain].append(path)
            self.paths_by_policy_id[path.policy_id] = path

        for constraint in bundle.constraints:
            self.constraints_by_scope[constraint.scope].append(constraint)

        for dr in bundle.dominance_rules:
            key = frozenset(dr.when.get("policies_fire", []))
            self.dominance_lookup[key] = dr

        for esc in bundle.escalations:
            key = frozenset(esc.policies)
            self.escalation_lookup[key] = esc


def validate_bundle_integrity(bundle: CompiledPolicyBundle) -> List[str]:
    """Return warnings for non-fatal integrity issues."""
    warnings: List[str] = []

    # Check that all variables referenced in conditions exist
    var_names = set(bundle.variables.keys())
    for rule in bundle.conditional_rules:
        for cond in rule.conditions:
            if cond.var not in var_names:
                warnings.append(f"Rule {rule.policy_id} references undefined variable '{cond.var}'")

    # Check that decision_nodes reference valid variables
    for node in bundle.decision_nodes:
        if node not in var_names:
            warnings.append(f"Decision node '{node}' not in variables")

    # Check policy_ids in dominance_rules exist
    rule_ids = {r.policy_id for r in bundle.conditional_rules}
    for dr in bundle.dominance_rules:
        for pid in dr.when.get("policies_fire", []):
            if pid not in rule_ids:
                warnings.append(f"Dominance rule references unknown policy '{pid}'")

    return warnings


def load_bundle(path: str) -> Tuple[CompiledPolicyBundle, BundleIndex]:
    """Load and validate a compiled policy bundle from disk.

    Args:
        path: Path to compiled_policy_bundle.json.

    Returns:
        Tuple of (validated bundle, indexed bundle).

    Raises:
        pydantic.ValidationError: On schema mismatch.
        FileNotFoundError: If path doesn't exist.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    bundle = CompiledPolicyBundle.model_validate(raw)
    warnings = validate_bundle_integrity(bundle)
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for w in warnings:
            logger.warning("Bundle integrity: %s", w)

    index = BundleIndex(bundle)
    return bundle, index
