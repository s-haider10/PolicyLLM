"""Hallucination metrics following ACL-standard evaluation methodology.

References:
  - SummaC (Laban et al., TACL 2022): NLI-based consistency scoring
  - Q² (Honovich et al., EMNLP 2021): QA-based faithfulness
  - AIS (Rashkin et al., ACL 2023): Attributable to Identified Sources
  - HaluEval (Li et al., EMNLP 2023): Hallucination evaluation framework
  - Maynez et al. (ACL 2020): Intrinsic vs extrinsic hallucination taxonomy
  - FaithDial (Dziri et al., TACL 2022): Faithfulness in dialogue

Metrics computed:
  1. Faithfulness (NLI):  LLM-as-NLI entailment judge (SummaC-style)
  2. Q² Score:            QA-consistency — generate Qs from response, verify vs source
  3. Hallucination Rate:  Binary per-response, fraction containing ≥1 hallucinated claim
  4. Intrinsic Halluc:    Rate of source-contradicting claims (Maynez taxonomy)
  5. Extrinsic Halluc:    Rate of unverifiable claims (Maynez taxonomy)
  6. AIS Score:           Sentence-level attribution to source (Rashkin et al.)
"""
from __future__ import annotations

import logging
import math
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "that", "this",
    "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "you", "your", "he", "him", "his", "she", "her",
})


def _tokenize(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOP


def _sentence_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10]


def _claim_overlap(claim: str, response: str, threshold: float = 0.40) -> bool:
    """Token-overlap check if a claim is reflected in the response."""
    ct = _tokenize(claim)
    rt = _tokenize(response)
    if not ct:
        return False
    return len(ct & rt) / len(ct) >= threshold


# ---------------------------------------------------------------------------
# 1. NLI-based Faithfulness (SummaC-style)
# ---------------------------------------------------------------------------

def nli_faithfulness(
    response: str,
    source_text: str,
    grounded_facts: List[str],
    llm_client: Any,
) -> float:
    """LLM-as-NLI judge: score each response sentence as entailed / neutral / contradicted.

    Returns average entailment score in [0, 1].
    """
    sentences = _sentence_split(response)
    if not sentences:
        return 1.0  # empty response => nothing to hallucinate

    source_excerpt = source_text[:4000]
    facts_str = "\n".join(f"- {f}" for f in grounded_facts)

    prompt = (
        "You are an NLI faithfulness judge evaluating response sentences against a source document.\n\n"
        f"SOURCE DOCUMENT (excerpt):\n{source_excerpt}\n\n"
        f"KNOWN GROUND-TRUTH FACTS:\n{facts_str}\n\n"
        f"RESPONSE TO EVALUATE:\n{response}\n\n"
        "For EACH sentence in the response, classify as:\n"
        '  "entailed" (1.0): fully supported by the source\n'
        '  "neutral" (0.5): not contradicted but not directly verifiable\n'
        '  "contradicted" (0.0): directly contradicts the source\n\n'
        "Return JSON:\n"
        '{"sentences": [{"text": "...", "label": "entailed|neutral|contradicted", "score": 0.0-1.0}], '
        '"overall_score": 0.0-1.0}'
    )
    try:
        result = llm_client.invoke_json(prompt)
        return float(result.get("overall_score", 0.5))
    except Exception as e:
        logger.warning("NLI faithfulness failed: %s", e)
        return 0.5


# ---------------------------------------------------------------------------
# 2. Q²-style QA Consistency
# ---------------------------------------------------------------------------

def qa_consistency(
    response: str,
    source_text: str,
    grounded_facts: List[str],
    llm_client: Any,
) -> float:
    """Q² metric: generate questions from response, verify answers against source.

    High Q² = response claims are consistent with source when probed.
    """
    if not response.strip():
        return 1.0

    source_excerpt = source_text[:3000]
    facts_str = "\n".join(f"- {f}" for f in grounded_facts)

    prompt = (
        "You are evaluating factual consistency using QA-based verification (Q² method).\n\n"
        f"SOURCE DOCUMENT (excerpt):\n{source_excerpt}\n\n"
        f"KNOWN FACTS:\n{facts_str}\n\n"
        f"RESPONSE:\n{response}\n\n"
        "Step 1: Generate 3-5 factual questions that could be answered by the RESPONSE.\n"
        "Step 2: For each question, check if the answer from the RESPONSE matches the SOURCE.\n"
        "Step 3: Score consistency.\n\n"
        "Return JSON:\n"
        '{"questions": [{"question": "...", "response_answer": "...", "source_answer": "...", '
        '"consistent": true/false}], "consistency_score": 0.0-1.0}'
    )
    try:
        result = llm_client.invoke_json(prompt)
        return float(result.get("consistency_score", 0.5))
    except Exception as e:
        logger.warning("Q² consistency failed: %s", e)
        return 0.5


# ---------------------------------------------------------------------------
# 3. Hallucination Rate (HaluEval-style binary)
# ---------------------------------------------------------------------------

def hallucination_detected(
    response: str,
    hallucinated_claims: List[str],
    grounded_facts: List[str],
    source_text: str,
    llm_client: Optional[Any] = None,
) -> Tuple[bool, float]:
    """HaluEval-style binary detection: does the response contain ANY hallucination?

    Uses both token-overlap heuristic and LLM judge for robustness.
    Returns (has_hallucination: bool, confidence: float).
    """
    # Token-overlap heuristic pass
    heuristic_halluc = any(_claim_overlap(h, response) for h in hallucinated_claims)

    # LLM judge pass (if available)
    llm_halluc = False
    llm_confidence = 0.5
    if llm_client is not None and response.strip():
        prompt = (
            "You are a hallucination detector. Given source facts, determine if the response "
            "contains ANY fabricated, unsupported, or incorrect claims.\n\n"
            f"SOURCE FACTS:\n" + "\n".join(f"- {f}" for f in grounded_facts) + "\n\n"
            f"KNOWN HALLUCINATIONS (for reference):\n" + "\n".join(f"- {h}" for h in hallucinated_claims) + "\n\n"
            f"RESPONSE:\n{response}\n\n"
            'Return JSON: {"has_hallucination": true/false, "confidence": 0.0-1.0, '
            '"detected_hallucinations": ["list of hallucinated claims found"]}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            llm_halluc = result.get("has_hallucination", False)
            llm_confidence = float(result.get("confidence", 0.5))
        except Exception:
            pass

    # Combine: hallucination if either method flags it
    combined = heuristic_halluc or llm_halluc
    # Confidence: average of heuristic (binary) and LLM confidence
    heur_conf = 0.9 if heuristic_halluc else 0.3
    combined_conf = (heur_conf + llm_confidence) / 2

    return combined, combined_conf


# ---------------------------------------------------------------------------
# 4 & 5. Intrinsic / Extrinsic Hallucination (Maynez et al. taxonomy)
# ---------------------------------------------------------------------------

def classify_hallucinations(
    response: str,
    grounded_facts: List[str],
    hallucinated_claims: List[str],
    source_text: str,
    llm_client: Optional[Any] = None,
) -> Dict[str, float]:
    """Classify hallucinations as intrinsic (contradicts source) or extrinsic (unverifiable).

    Returns {"intrinsic_rate": float, "extrinsic_rate": float}.
    """
    sentences = _sentence_split(response)
    if not sentences:
        return {"intrinsic_rate": 0.0, "extrinsic_rate": 0.0}

    if llm_client is not None:
        source_excerpt = source_text[:3000]
        prompt = (
            "Classify each sentence in the RESPONSE using Maynez et al. (2020) taxonomy:\n"
            "  - FAITHFUL: supported by the source document\n"
            "  - INTRINSIC: contradicts information in the source document\n"
            "  - EXTRINSIC: contains information not verifiable from the source (fabricated)\n\n"
            f"SOURCE (excerpt):\n{source_excerpt}\n\n"
            f"RESPONSE:\n{response}\n\n"
            "Return JSON:\n"
            '{"classifications": [{"sentence": "...", "label": "faithful|intrinsic|extrinsic"}], '
            '"intrinsic_count": int, "extrinsic_count": int, "total_sentences": int}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            total = max(int(result.get("total_sentences", len(sentences))), 1)
            intr = int(result.get("intrinsic_count", 0))
            extr = int(result.get("extrinsic_count", 0))
            return {
                "intrinsic_rate": intr / total,
                "extrinsic_rate": extr / total,
            }
        except Exception as e:
            logger.warning("Hallucination classification failed: %s", e)

    # Fallback: heuristic-based
    source_tokens = _tokenize(source_text)
    grounded_tokens = _tokenize(" ".join(grounded_facts))
    intrinsic = 0
    extrinsic = 0
    for sent in sentences:
        st = _tokenize(sent)
        if not st:
            continue
        src_overlap = len(st & source_tokens) / len(st)
        gnd_overlap = len(st & grounded_tokens) / len(st) if grounded_tokens else 0
        # Check if any hallucinated claim is reflected
        has_halluc_claim = any(_claim_overlap(h, sent) for h in hallucinated_claims)
        if has_halluc_claim and src_overlap > 0.2:
            intrinsic += 1  # Contradicts source (uses source words but wrong meaning)
        elif src_overlap < 0.15 and gnd_overlap < 0.15:
            extrinsic += 1  # Unverifiable
    n = len(sentences)
    return {"intrinsic_rate": intrinsic / n, "extrinsic_rate": extrinsic / n}


# ---------------------------------------------------------------------------
# 6. AIS Score (Rashkin et al.)
# ---------------------------------------------------------------------------

def ais_score(
    response: str,
    source_text: str,
    grounded_facts: List[str],
    llm_client: Optional[Any] = None,
) -> float:
    """AIS (Attributable to Identified Sources): sentence-level attribution.

    Each sentence is scored 1 if attributable to the source, 0 otherwise.
    Returns the average across sentences.
    """
    sentences = _sentence_split(response)
    if not sentences:
        return 1.0

    if llm_client is not None:
        source_excerpt = source_text[:3000]
        prompt = (
            "For each sentence in the RESPONSE, determine if it is ATTRIBUTABLE to the "
            "SOURCE document (AIS framework, Rashkin et al. 2023).\n"
            "A sentence is attributable if a reader could verify it using the source.\n\n"
            f"SOURCE (excerpt):\n{source_excerpt}\n\n"
            f"RESPONSE:\n{response}\n\n"
            "Return JSON:\n"
            '{"sentences": [{"text": "...", "attributable": true/false}], '
            '"ais_score": 0.0-1.0}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            return float(result.get("ais_score", 0.5))
        except Exception as e:
            logger.warning("AIS scoring failed: %s", e)

    # Fallback: token overlap
    source_tokens = _tokenize(source_text)
    attributed = 0
    for sent in sentences:
        st = _tokenize(sent)
        if not st:
            attributed += 1
            continue
        overlap = len(st & source_tokens) / len(st)
        if overlap >= 0.25:
            attributed += 1
    return attributed / len(sentences)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    if not scores:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1 - ci
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return sum(scores) / n, means[lo_idx], means[min(hi_idx, n_bootstrap - 1)]


# ---------------------------------------------------------------------------
# Full per-scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_response(
    response: str,
    grounded_facts: List[str],
    hallucinated_claims: List[str],
    source_text: str,
    llm_client: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute all hallucination metrics for a single response.

    Uses a SINGLE comprehensive LLM judge call to minimize API usage
    while computing all 6 metrics. Falls back to heuristics if no LLM.
    """
    if not response.strip():
        # Empty response: no hallucinations but also no useful content
        return {
            "faithfulness": 0.5,
            "qa_consistency": 0.0,
            "hallucination_detected": 0.0,
            "intrinsic_halluc": 0.0,
            "extrinsic_halluc": 0.0,
            "ais_score": 0.5,
        }

    # ── Single comprehensive LLM judge (saves 4x API calls) ──────────
    if llm_client is not None:
        source_excerpt = source_text[:3500]
        facts_str = "\n".join(f"  - {f}" for f in grounded_facts)
        halluc_str = "\n".join(f"  - {h}" for h in hallucinated_claims)

        prompt = (
            "You are an expert hallucination evaluation judge for NLP research.\n"
            "Evaluate the RESPONSE against the SOURCE document using standard metrics.\n\n"
            f"SOURCE DOCUMENT (excerpt):\n{source_excerpt}\n\n"
            f"GROUND-TRUTH FACTS (from source):\n{facts_str}\n\n"
            f"KNOWN HALLUCINATED CLAIMS (should NOT appear):\n{halluc_str}\n\n"
            f"RESPONSE TO EVALUATE:\n{response}\n\n"
            "Score each metric:\n"
            "1. faithfulness (0.0-1.0): NLI entailment — fraction of response claims supported by source\n"
            "2. qa_consistency (0.0-1.0): Q² — if you ask questions about the response, do source answers match?\n"
            "3. has_hallucination (true/false): does response contain ANY fabricated/unsupported claim?\n"
            "4. intrinsic_rate (0.0-1.0): fraction of sentences contradicting the source\n"
            "5. extrinsic_rate (0.0-1.0): fraction of sentences with unverifiable claims\n"
            "6. ais_score (0.0-1.0): fraction of sentences attributable to identified source passages\n\n"
            "Be strict but fair. A response that correctly says 'this information is not in the document' "
            "is faithful and should score high on faithfulness.\n\n"
            "Return JSON:\n"
            '{"faithfulness": 0.0-1.0, "qa_consistency": 0.0-1.0, '
            '"has_hallucination": true/false, "intrinsic_rate": 0.0-1.0, '
            '"extrinsic_rate": 0.0-1.0, "ais_score": 0.0-1.0, '
            '"reasoning": "brief explanation"}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            return {
                "faithfulness": float(result.get("faithfulness", 0.5)),
                "qa_consistency": float(result.get("qa_consistency", 0.5)),
                "hallucination_detected": 1.0 if result.get("has_hallucination", True) else 0.0,
                "intrinsic_halluc": float(result.get("intrinsic_rate", 0.0)),
                "extrinsic_halluc": float(result.get("extrinsic_rate", 0.0)),
                "ais_score": float(result.get("ais_score", 0.5)),
            }
        except Exception as e:
            logger.warning("Comprehensive LLM judge failed: %s — falling back to heuristics", e)

    # ── Heuristic fallback (no LLM) ──────────────────────────────────
    metrics: Dict[str, float] = {}
    sentences = _sentence_split(response)
    src_tk = _tokenize(source_text)
    grounded_tk = _tokenize(" ".join(grounded_facts))

    # Faithfulness (token overlap)
    if sentences:
        faith_count = sum(
            1 for s in sentences
            if len(_tokenize(s) & src_tk) / max(len(_tokenize(s)), 1) >= 0.25
        )
        metrics["faithfulness"] = faith_count / len(sentences)
    else:
        metrics["faithfulness"] = 0.5

    # Q² (grounded fact coverage)
    if grounded_facts:
        covered = sum(1 for f in grounded_facts if _claim_overlap(f, response, 0.35))
        metrics["qa_consistency"] = covered / len(grounded_facts)
    else:
        metrics["qa_consistency"] = 1.0

    # Hallucination detected (token overlap with hallucinated claims)
    has_halluc = any(_claim_overlap(h, response) for h in hallucinated_claims)
    metrics["hallucination_detected"] = 1.0 if has_halluc else 0.0

    # Intrinsic / Extrinsic (heuristic)
    intrinsic = 0
    extrinsic = 0
    for sent in (sentences or []):
        st = _tokenize(sent)
        if not st:
            continue
        src_ovl = len(st & src_tk) / len(st)
        has_halluc_claim = any(_claim_overlap(h, sent) for h in hallucinated_claims)
        if has_halluc_claim:
            intrinsic += 1
        elif src_ovl < 0.15:
            extrinsic += 1
    n_sent = max(len(sentences), 1)
    metrics["intrinsic_halluc"] = intrinsic / n_sent
    metrics["extrinsic_halluc"] = extrinsic / n_sent

    # AIS (token overlap with source)
    if sentences:
        attr = sum(
            1 for s in sentences
            if len(_tokenize(s) & src_tk) / max(len(_tokenize(s)), 1) >= 0.25
        )
        metrics["ais_score"] = attr / len(sentences)
    else:
        metrics["ais_score"] = 0.5

    return metrics


# ---------------------------------------------------------------------------
# Aggregation with CIs
# ---------------------------------------------------------------------------

def aggregate_with_ci(
    per_scenario: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Macro-average with bootstrap 95% CIs.

    Returns {metric_name: {"mean": float, "ci_lo": float, "ci_hi": float}}.
    """
    if not per_scenario:
        keys = ["faithfulness", "qa_consistency", "hallucination_detected",
                "intrinsic_halluc", "extrinsic_halluc", "ais_score"]
        return {k: {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0} for k in keys}

    keys = list(per_scenario[0].keys())
    result = {}
    for k in keys:
        scores = [d.get(k, 0.0) for d in per_scenario]
        mean, lo, hi = bootstrap_ci(scores)
        result[k] = {"mean": round(mean, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}
    return result
