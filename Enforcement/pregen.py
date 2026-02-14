"""Pre-generation: query classification, rule retrieval, dominance resolution."""
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .bundle_loader import BundleIndex
from .schemas import (
    CompiledPath,
    CompiledPolicyBundle,
    ConditionalRule,
    Constraint,
    DominanceRule,
    EnforcementContext,
)

# ---------------------------------------------------------------------------
# Keyword dictionaries for classification
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "refund": ["refund", "return", "money back", "exchange", "receipt", "store credit", "reimburse"],
    "privacy": ["pii", "personal data", "gdpr", "privacy", "disclose", "data protection", "confidential"],
    "escalation": ["manager", "supervisor", "escalate", "complaint", "appeal", "higher authority"],
    "security": ["password", "authentication", "breach", "encrypt", "access control", "firewall"],
    "hr": ["employee", "leave", "payroll", "hiring", "termination", "benefits", "vacation"],
}

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "refund_request": ["want refund", "return item", "get my money", "return this", "send back"],
    "policy_inquiry": ["what is the policy", "rules for", "am i allowed", "can i", "is it possible"],
    "complaint": ["not happy", "unacceptable", "file complaint", "terrible", "disappointed"],
}


def classify_query(
    query: str,
    bundle: CompiledPolicyBundle,
    llm_client: Optional[Any] = None,
) -> Tuple[str, str, float]:
    """Classify a user query into (domain, intent, confidence).

    Uses keyword matching first. Falls back to LLM if confidence is low
    and an LLM client is available.
    """
    lower = query.lower()
    tokens = set(lower.split())

    # Score each domain
    best_domain = "unknown"
    best_domain_score = 0.0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in lower)
        score = matches / len(keywords) if keywords else 0.0
        if score > best_domain_score:
            best_domain_score = score
            best_domain = domain

    # Also check against domains present in the bundle
    bundle_domains = {r.metadata.domain for r in bundle.conditional_rules}
    if best_domain not in bundle_domains and best_domain != "unknown":
        # Try to find a domain from the bundle that matches
        for bd in bundle_domains:
            if bd in lower:
                best_domain = bd
                best_domain_score = 0.6
                break

    # Score intents
    best_intent = "unknown"
    best_intent_score = 0.0
    for intent, keywords in INTENT_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in lower)
        score = matches / len(keywords) if keywords else 0.0
        if score > best_intent_score:
            best_intent_score = score
            best_intent = intent

    confidence = max(best_domain_score, best_intent_score)

    # LLM fallback
    if confidence < 0.6 and llm_client is not None:
        try:
            available_domains = list(bundle_domains)
            prompt = (
                f"Classify this user query into one of these domains: {available_domains}.\n"
                f"Also classify the intent as one of: refund_request, policy_inquiry, complaint, other.\n"
                f"Query: {query}\n"
                f'Respond as JSON: {{"domain": "...", "intent": "...", "confidence": 0.0-1.0}}'
            )
            from pydantic import BaseModel

            class ClassifyOut(BaseModel):
                domain: str
                intent: str
                confidence: float

            result = llm_client.invoke_json(prompt, schema=ClassifyOut)
            if result.get("confidence", 0) > confidence:
                best_domain = result.get("domain", best_domain)
                best_intent = result.get("intent", best_intent)
                confidence = result.get("confidence", confidence)
        except Exception:
            pass

    return best_domain, best_intent, confidence


def retrieve_rules(
    domain: str,
    bundle_index: BundleIndex,
    effective_date: Optional[str] = None,
) -> Tuple[List[ConditionalRule], List[CompiledPath], List[Constraint]]:
    """Retrieve applicable rules, paths, and constraints for a domain."""
    rules = list(bundle_index.rules_by_domain.get(domain, []))

    # Temporal filtering
    if effective_date:
        try:
            cutoff = date.fromisoformat(effective_date)
        except (ValueError, TypeError):
            cutoff = date.today()
        filtered_rules = []
        for r in rules:
            eff = r.metadata.eff_date
            if eff is None:
                filtered_rules.append(r)
            else:
                try:
                    if date.fromisoformat(eff) <= cutoff:
                        filtered_rules.append(r)
                except (ValueError, TypeError):
                    filtered_rules.append(r)
        rules = filtered_rules

    # Get paths for retrieved rules
    rule_pids = {r.policy_id for r in rules}
    paths = [p for p in bundle_index.paths_by_domain.get(domain, []) if p.policy_id in rule_pids]

    # Constraints: always + domain-scoped
    constraints = list(bundle_index.constraints_by_scope.get("always", []))
    constraints.extend(bundle_index.constraints_by_scope.get(domain, []))

    return rules, paths, constraints


def apply_dominance(
    rules: List[ConditionalRule],
    paths: List[CompiledPath],
    bundle_index: BundleIndex,
    priority_lattice: Dict[str, int],
) -> Tuple[List[ConditionalRule], List[CompiledPath], List[DominanceRule]]:
    """Resolve conflicts among retrieved rules using dominance rules and priority lattice."""
    applied: List[DominanceRule] = []
    losers: set = set()
    pids = sorted(r.policy_id for r in rules)

    for i, p1 in enumerate(pids):
        for p2 in pids[i + 1:]:
            key: FrozenSet[str] = frozenset({p1, p2})
            dr = bundle_index.dominance_lookup.get(key)
            if dr:
                mode = dr.then.get("mode", "override")
                enforced = dr.then.get("enforce", "")
                if mode == "override":
                    loser = p2 if enforced == p1 else p1
                    losers.add(loser)
                applied.append(dr)
            else:
                # Fall back to priority lattice
                r1 = bundle_index.rules_by_policy_id.get(p1)
                r2 = bundle_index.rules_by_policy_id.get(p2)
                if r1 and r2:
                    rank1 = priority_lattice.get(r1.metadata.priority, 3)
                    rank2 = priority_lattice.get(r2.metadata.priority, 3)
                    if rank1 != rank2:
                        loser = p2 if rank1 < rank2 else p1
                        losers.add(loser)

    filtered_rules = [r for r in rules if r.policy_id not in losers]
    filtered_paths = [p for p in paths if p.policy_id not in losers]
    return filtered_rules, filtered_paths, applied


def build_context(
    query: str,
    bundle: CompiledPolicyBundle,
    bundle_index: BundleIndex,
    session_id: Optional[str] = None,
    llm_client: Optional[Any] = None,
    effective_date: Optional[str] = None,
) -> EnforcementContext:
    """Full pregen pipeline: classify, retrieve, resolve, assemble context."""
    sid = session_id or str(uuid.uuid4())
    domain, intent, confidence = classify_query(query, bundle, llm_client)
    rules, paths, constraints = retrieve_rules(domain, bundle_index, effective_date or str(date.today()))
    filtered_rules, filtered_paths, applied = apply_dominance(
        rules, paths, bundle_index, bundle.priority_lattice,
    )

    # Collect escalation contacts
    esc_contacts: List[str] = []
    rule_pids = {r.policy_id for r in filtered_rules}
    for esc in bundle.escalations:
        if set(esc.policies) & rule_pids:
            esc_contacts.extend(esc.owners_to_notify)
    esc_contacts = sorted(set(esc_contacts))

    return EnforcementContext(
        session_id=sid,
        query=query,
        domain=domain,
        intent=intent,
        domain_confidence=confidence,
        applicable_rules=filtered_rules,
        applicable_constraints=constraints,
        applicable_paths=filtered_paths,
        dominance_applied=applied,
        escalation_contacts=esc_contacts,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
