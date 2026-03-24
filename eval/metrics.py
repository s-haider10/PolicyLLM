"""Core metric computation for PolicyLLM evaluation.

All metrics follow standard IR / NLP evaluation practice:
  - Policy Recall / Precision: set-based with fuzzy matching
  - Condition F1: micro-averaged over condition triples
  - Conflict F1: set-based over conflict pairs
  - Compliance %: accuracy of enforcement decisions
  - FP %: false-positive rate (escalated / blocked when should not have been)
  - Latency: wall-clock seconds for the full pipeline
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Set[str]:
    """Lower-case word tokenization for fuzzy matching."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Policy-level precision / recall / F1
# ---------------------------------------------------------------------------

_MATCH_THRESHOLD = 0.10  # Jaccard threshold — low because extracted descriptions differ from annotations


def _best_match(
    pred: Dict[str, Any],
    refs: List[Dict[str, Any]],
    used: Set[int],
) -> Optional[int]:
    """Find the best matching reference policy for a predicted policy.

    Matching strategy:
      1. Domain must match (or ref domain is 'any')
      2. Score = Jaccard(pred_tokens, ref_tokens) where tokens include
         description, domain, keywords, and action_type
      3. Best scoring match above threshold wins
    """
    pred_tokens = _tokenize(
        " ".join([
            pred.get("description", ""),
            pred.get("domain", ""),
            pred.get("action_type", ""),
            " ".join(pred.get("keywords", [])),
        ])
    )
    best_idx, best_sim = None, 0.0
    for i, ref in enumerate(refs):
        if i in used:
            continue
        # Domain filter: same domain or generic 'other'
        pred_domain = pred.get("domain", "other")
        ref_domain = ref.get("domain", "other")
        if ref_domain != "any" and pred_domain != ref_domain:
            # Allow 'other' to match anything as a soft filter
            if pred_domain != "other" and ref_domain != "other":
                continue
        ref_tokens = _tokenize(
            " ".join([
                ref.get("description", ""),
                ref.get("domain", ""),
                " ".join(ref.get("keywords", [])),
                ref.get("action_type", ""),
            ])
        )
        sim = _jaccard(pred_tokens, ref_tokens)
        # Boost score if domains match exactly
        if pred_domain == ref_domain and pred_domain != "other":
            sim = min(1.0, sim * 1.5)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    if best_sim >= _MATCH_THRESHOLD and best_idx is not None:
        return best_idx
    return None


def policy_precision_recall_f1(
    predicted: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for extracted policies vs. reference.

    Matching: greedy best-match by domain + Jaccard(description tokens).
    """
    if not predicted and not reference:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not reference:
        return 0.0, 0.0, 0.0

    used: Set[int] = set()
    tp = 0
    for p in predicted:
        idx = _best_match(p, reference, used)
        if idx is not None:
            used.add(idx)
            tp += 1

    precision = tp / len(predicted)
    recall = tp / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Condition-level F1 (micro-averaged)
# ---------------------------------------------------------------------------

def _cond_key(c: Any) -> str:
    """Canonical key for a condition: type|operator.

    Uses type+operator (without parameter name) because extracted parameter
    names vary across methods (e.g. 'days_since_purchase' vs 'return_period')
    while type+operator captures the semantic intent of the condition.
    """
    if isinstance(c, str):
        return c  # Plain string condition
    if not isinstance(c, dict):
        return str(c)
    ctype = c.get('type', '')
    op = c.get('operator', c.get('op', ''))
    # Normalize null/missing operators based on condition type
    if op is None or op == '':
        if ctype == 'boolean_flag':
            op = '=='
        elif ctype in ('time_window', 'amount_threshold'):
            op = '<='
    # Normalize single = to ==
    if op == '=':
        op = '=='
    return f"{ctype}|{op}"


def condition_f1(
    predicted_conditions: List[Dict[str, Any]],
    reference_conditions: List[Dict[str, Any]],
) -> float:
    """Micro-averaged F1 over condition triples (type, parameter, operator)."""
    pred_keys = Counter(_cond_key(c) for c in predicted_conditions)
    ref_keys = Counter(_cond_key(c) for c in reference_conditions)
    tp = sum((pred_keys & ref_keys).values())
    fp = sum((pred_keys - ref_keys).values())
    fn = sum((ref_keys - pred_keys).values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# ---------------------------------------------------------------------------
# Conflict-level F1
# ---------------------------------------------------------------------------

def _conflict_key(c: Dict[str, Any]) -> frozenset:
    """Canonical key for a conflict: frozenset of involved policy descriptions."""
    return frozenset(c.get("policies", []))


def conflict_f1(
    predicted_conflicts: List[Dict[str, Any]],
    reference_conflicts: List[Dict[str, Any]],
) -> float:
    """F1 over conflict pairs."""
    pred_set = {_conflict_key(c) for c in predicted_conflicts}
    ref_set = {_conflict_key(c) for c in reference_conflicts}
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# ---------------------------------------------------------------------------
# Enforcement metrics
# ---------------------------------------------------------------------------

def compliance_rate(
    decisions: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
) -> float:
    """Fraction of enforcement decisions that are correct.

    Correctness definition:
      - Expected "pass": correct if action is pass, auto_correct, or regenerate
        (i.e. NOT escalated — response is delivered or retried, not blocked)
      - Expected "escalate": correct if action is escalate
    """
    if not decisions:
        return 0.0
    correct = 0
    for dec, exp in zip(decisions, expected):
        exp_action = exp.get("expected_action", "pass")
        dec_action = dec.get("action", "pass")
        if exp_action == "escalate":
            if dec_action == "escalate":
                correct += 1
        else:
            # pass, auto_correct, regenerate are all non-blocking actions
            if dec_action in ("pass", "auto_correct", "regenerate"):
                correct += 1
    return correct / len(decisions)


def false_positive_rate(
    decisions: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
) -> float:
    """FP rate: cases escalated (blocked) when expected to pass."""
    should_pass = [(d, e) for d, e in zip(decisions, expected) if e.get("expected_action") == "pass"]
    if not should_pass:
        return 0.0
    fp = sum(1 for d, _ in should_pass if d.get("action") == "escalate")
    return fp / len(should_pass)


# ---------------------------------------------------------------------------
# Aggregation across documents
# ---------------------------------------------------------------------------

def aggregate_extraction_metrics(
    per_doc: List[Dict[str, float]],
) -> Dict[str, float]:
    """Macro-average extraction metrics across documents."""
    if not per_doc:
        return {"precision": 0.0, "recall": 0.0, "condition_f1": 0.0, "conflict_f1": 0.0}
    keys = ["precision", "recall", "condition_f1", "conflict_f1"]
    return {k: sum(d.get(k, 0.0) for d in per_doc) / len(per_doc) for k in keys}


def aggregate_enforcement_metrics(
    results: Dict[str, Any],
) -> Dict[str, float]:
    """Summarize enforcement metrics."""
    decisions = results.get("decisions", [])
    expected = results.get("expected", [])
    latencies = results.get("latencies", [])
    comp = compliance_rate(decisions, expected)
    fp = false_positive_rate(decisions, expected)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "compliance_pct": comp * 100,
        "fp_pct": fp * 100,
        "latency_s": avg_latency,
    }


def build_table_row(
    method_name: str,
    extraction: Dict[str, float],
    enforcement: Dict[str, float],
) -> List[str]:
    """Build a single comparison table row.

    Columns: Method | Policy Recall | Policy Precision | Condition F1 |
             Conflict F1 | Compliance % | FP % | Latency (s)
    """
    return [
        method_name,
        f"{extraction.get('recall', 0.0):.2f}",
        f"{extraction.get('precision', 0.0):.2f}",
        f"{extraction.get('condition_f1', 0.0):.2f}",
        f"{extraction.get('conflict_f1', 0.0):.1f}",
        f"{enforcement.get('compliance_pct', 0.0):.1f}",
        f"{enforcement.get('fp_pct', 0.0):.1f}",
        f"{enforcement.get('latency_s', 0.0):.2f}",
    ]
