"""Baseline implementations for ACL evaluation.

Each baseline implements two phases:
  1. extract(sections) -> List[policy_dict]  (extraction)
  2. enforce(query, response, bundle) -> decision_dict  (enforcement)

Baselines are designed as ablations / alternatives of the PolicyLLM pipeline
so they can be evaluated on the exact same data with comparable metrics.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaselineExtractor(ABC):
    """Extract policies from regularized document sections."""
    name: str = "base"

    @abstractmethod
    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """Return a list of policy dicts with at least: domain, description, conditions, action_type."""
        ...


class BaselineEnforcer(ABC):
    """Enforce policies against a query/response pair."""
    name: str = "base"

    @abstractmethod
    def enforce(
        self,
        query: str,
        response: str,
        bundle: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return dict with: action (str), violations (list), score (float), latency_s (float)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# 1. Keyword overlap + rules  (no LLM, no embeddings)
# ─────────────────────────────────────────────────────────────────────────────

_POLICY_SIGNALS = [
    r"\b(?:must|shall|may\s+not|prohibited|required|mandatory|forbidden)\b",
    r"\b(?:penalty|violation|compliance|obligation|entitled|refund|return)\b",
    r"\b(?:within\s+\d+\s+days|no\s+later\s+than|prior\s+to)\b",
]
_COMPILED_SIGNALS = [re.compile(p, re.IGNORECASE) for p in _POLICY_SIGNALS]

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "refund": ["return", "refund", "exchange", "store credit", "receipt", "defective"],
    "privacy": ["personal data", "pii", "confidential", "privacy", "gdpr", "hipaa", "health information"],
    "security": ["security", "unauthorized", "access", "safeguard", "threat", "virus"],
    "hr": ["employee", "conduct", "workplace", "disciplinary", "grievance"],
    "escalation": ["dispute", "arbitration", "escalation", "appeal", "settlement"],
}

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "password": re.compile(r"(?i)\bpassword\s*[:=]\s*\S+"),
}
_GUARANTEE_PATTERN = re.compile(r"(?i)\b(?:I\s+guarantee|I\s+promise|we\s+will\s+definitely)\b")


def _detect_domain(text: str) -> str:
    lower = text.lower()
    scores = {}
    for domain, kws in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in kws if kw in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def _extract_conditions_regex(text: str) -> List[Dict[str, Any]]:
    """Extract conditions via regex heuristics."""
    conditions: List[Dict[str, Any]] = []
    # Time windows
    for m in re.finditer(r"(?i)within\s+(\d+)\s+(days?|months?|years?)", text):
        val = int(m.group(1))
        unit = m.group(2).rstrip("s")
        conditions.append({"type": "time_window", "parameter": f"days_since_purchase", "operator": "<=", "value": val, "unit": unit})
    # Boolean flags
    for m in re.finditer(r"(?i)(receipt|tags?|proof|original)\s+(?:is\s+)?(?:required|must|needed)", text):
        param = m.group(1).lower().replace(" ", "_")
        conditions.append({"type": "boolean_flag", "parameter": f"{param}_required", "operator": "==", "value": True})
    # Amount thresholds
    for m in re.finditer(r"(?i)(?:not\s+more\s+than|up\s+to|maximum)\s+\$?([\d,]+)", text):
        try:
            val = float(m.group(1).replace(",", ""))
            conditions.append({"type": "amount_threshold", "parameter": "refund_amount", "operator": "<=", "value": val})
        except ValueError:
            pass
    return conditions


class KeywordRulesExtractor(BaselineExtractor):
    """Pure regex/keyword extraction — no LLM, no embeddings."""
    name = "Keyword overlap + rules"

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        policies: List[Dict[str, Any]] = []
        for sec in sections:
            paras = sec.get("paragraphs", [])
            text = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in paras)
            signal_count = sum(1 for pat in _COMPILED_SIGNALS if pat.search(text))
            if signal_count < 1:
                continue
            domain = _detect_domain(text)
            conditions = _extract_conditions_regex(text)
            # Determine action type from keywords
            lower = text.lower()
            if any(kw in lower for kw in ["prohibited", "forbidden", "may not", "must not"]):
                action_type = "prohibited"
            elif any(kw in lower for kw in ["required", "must", "shall"]):
                action_type = "required"
            elif any(kw in lower for kw in ["may", "optional", "eligible"]):
                action_type = "conditional"
            else:
                action_type = "other"
            policies.append({
                "domain": domain,
                "description": text[:200].strip(),
                "conditions": conditions,
                "action_type": action_type,
                "keywords": list(set(re.findall(r"\b\w{4,}\b", text.lower())))[:20],
            })
        return policies


class KeywordRulesEnforcer(BaselineEnforcer):
    """Enforcement via keyword/regex scanning only — no LLM."""
    name = "Keyword overlap + rules"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        violations: List[str] = []
        # PII scan
        for name, pat in _PII_PATTERNS.items():
            m = pat.search(response)
            if m:
                violations.append(f"{name}: matched '{m.group()}'")
        # Guarantee/promise
        if _GUARANTEE_PATTERN.search(response):
            violations.append("guarantee_promise: detected commitment language")
        score = 1.0 if not violations else 0.0
        action = "pass" if not violations else "escalate"
        return {"action": action, "violations": violations, "score": score, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Semantic similarity + rules  (sentence-transformers, no LLM API)
# ─────────────────────────────────────────────────────────────────────────────

_SEMANTIC_MODEL = None


def _get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            # Handles ImportError, version conflicts (torch/numpy), etc.
            logger.warning("sentence-transformers unavailable (%s); semantic baseline will use keyword fallback", type(e).__name__)
            return None
    return _SEMANTIC_MODEL


_POLICY_TEMPLATES = [
    "customers must return items within the required time window",
    "receipt is required for refund processing",
    "personal information must be protected and not disclosed",
    "unauthorized access to systems is prohibited",
    "employees must follow workplace conduct standards",
    "disputes shall be resolved through arbitration",
    "penalties apply for non-compliance with regulations",
    "prohibited items cannot be returned",
    "security safeguards must be maintained",
    "intellectual property rights are protected",
]


class SemanticRulesExtractor(BaselineExtractor):
    """Embedding-based extraction: sections similar to policy templates are policies.

    Falls back to keyword extraction when sentence-transformers is unavailable.
    """
    name = "Semantic similarity + rules"

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        try:
            model = _get_semantic_model()
        except Exception:
            model = None
        if model is None:
            # Fallback: use keyword extraction + slight boost
            kw = KeywordRulesExtractor()
            return kw.extract(sections, full_text)
        import numpy as np

        template_embs = model.encode(_POLICY_TEMPLATES, convert_to_numpy=True)
        policies: List[Dict[str, Any]] = []
        for sec in sections:
            paras = sec.get("paragraphs", [])
            text = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in paras)
            if len(text.strip()) < 20:
                continue
            sec_emb = model.encode([text], convert_to_numpy=True)
            # Cosine similarity with templates
            sims = np.dot(sec_emb, template_embs.T).flatten()
            max_sim = float(np.max(sims))
            if max_sim < 0.35:
                continue
            domain = _detect_domain(text)
            conditions = _extract_conditions_regex(text)
            lower = text.lower()
            if any(kw in lower for kw in ["prohibited", "forbidden", "may not"]):
                action_type = "prohibited"
            elif any(kw in lower for kw in ["required", "must", "shall"]):
                action_type = "required"
            else:
                action_type = "conditional"
            policies.append({
                "domain": domain,
                "description": text[:200].strip(),
                "conditions": conditions,
                "action_type": action_type,
                "keywords": list(set(re.findall(r"\b\w{4,}\b", text.lower())))[:20],
                "similarity": max_sim,
            })
        return policies


class SemanticRulesEnforcer(BaselineEnforcer):
    """Enforcement via semantic similarity + keyword rules.

    Falls back to keyword-only enforcement when sentence-transformers is unavailable.
    """
    name = "Semantic similarity + rules"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        violations: List[str] = []
        # PII scan (hard gate, same as keyword)
        for pii_name, pat in _PII_PATTERNS.items():
            m = pat.search(response)
            if m:
                violations.append(f"{pii_name}: matched '{m.group()}'")
        if _GUARANTEE_PATTERN.search(response):
            violations.append("guarantee_promise: detected")
        # Semantic: check if response contradicts policy templates
        try:
            model = _get_semantic_model()
            if model is not None and bundle:
                import numpy as np
                constraints = bundle.get("constraints", [])
                if constraints:
                    constraint_texts = [c.get("constraint", "") for c in constraints if isinstance(c, dict)]
                    if constraint_texts:
                        resp_emb = model.encode([response], convert_to_numpy=True)
                        con_embs = model.encode(constraint_texts, convert_to_numpy=True)
                        sims = np.dot(resp_emb, con_embs.T).flatten()
                        for i, sim in enumerate(sims):
                            if sim > 0.5:
                                violations.append(f"semantic_constraint_violation: {constraint_texts[i]} (sim={sim:.2f})")
        except Exception:
            pass  # Fall back to keyword-only enforcement

        score = 1.0 if not violations else max(0.0, 1.0 - 0.3 * len(violations))
        action = "pass" if not violations else "escalate"
        return {"action": action, "violations": violations, "score": score, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMT-only (Z3 verification, no neural components)
# ─────────────────────────────────────────────────────────────────────────────

class SMTOnlyExtractor(BaselineExtractor):
    """Template/regex extraction + Z3 for conflict detection — no LLM."""
    name = "SMT-only (Z3, no neural)"

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        # Reuse keyword extraction (same quality without LLM)
        kw = KeywordRulesExtractor()
        return kw.extract(sections, full_text)


class SMTOnlyEnforcer(BaselineEnforcer):
    """Enforcement using only Z3 SMT verification — no judge, no regex, no coverage."""
    name = "SMT-only (Z3, no neural)"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        if bundle is None:
            return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}
        try:
            from Enforcement.schemas import CompiledPolicyBundle, VariableSchema
            from Enforcement.postgen.smt import extract_facts_from_response, verify_facts_against_rules
            variables = {}
            for k, v in bundle.get("variables", {}).items():
                variables[k] = VariableSchema(**v) if isinstance(v, dict) else v

            # Extract facts without LLM (symbolic only — pass llm_client=None)
            facts = extract_facts_from_response(response, variables, llm_client=None)

            from Enforcement.schemas import ConditionalRule, CompiledPath, Constraint
            rules = [ConditionalRule(**r) for r in bundle.get("conditional_rules", [])]
            paths = [CompiledPath(**p) for p in bundle.get("compiled_paths", [])]
            constraints = [Constraint(**c) for c in bundle.get("constraints", [])]

            result = verify_facts_against_rules(facts, rules, paths, constraints, variables)
            action = "pass" if result.passed else "escalate"
            return {
                "action": action,
                "violations": [str(v) for v in result.violations],
                "score": result.score,
                "latency_s": time.time() - t0,
            }
        except Exception as e:
            logger.warning("SMT-only enforcement failed: %s", e)
            return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vanilla LLM (single call, no structured extraction, no verification)
# ─────────────────────────────────────────────────────────────────────────────

class VanillaLLMExtractor(BaselineExtractor):
    """Single LLM call to list policies — no multi-pass, no structure."""
    name = "Vanilla LLM (no policy)"

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        if self._llm is None:
            return []
        # Truncate text to fit context
        text = full_text[:8000]
        prompt = (
            "List all policies, rules, and obligations in this document. "
            "Return JSON array of objects. Each object should have: "
            "domain (string), description (string), action_type (string), "
            'conditions (array of {{"type": "time_window|boolean_flag|amount_threshold", '
            '"parameter": "name", "operator": "<=|==|>="}}).\n\n'
            f"Document:\n{text}\n\n"
            "Return JSON array only."
        )
        try:
            result = self._llm.invoke_json(prompt)
            policies = []
            if isinstance(result, list):
                policies = result
            elif isinstance(result, dict) and "policies" in result:
                policies = result["policies"]
            return [{"domain": p.get("domain", "other"), "description": p.get("description", ""),
                     "conditions": p.get("conditions", []),
                     "action_type": p.get("action_type", "other"),
                     "keywords": []} for p in policies]
        except Exception as e:
            logger.warning("Vanilla LLM extraction failed: %s", e)
            return []


class VanillaLLMEnforcer(BaselineEnforcer):
    """No enforcement — always pass (simulates no policy system)."""
    name = "Vanilla LLM (no policy)"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        # No verification at all — always pass
        return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 5. System prompt injection (policy text in system prompt, no structured checks)
# ─────────────────────────────────────────────────────────────────────────────

class SystemPromptExtractor(BaselineExtractor):
    """System-prompt-guided extraction — single LLM pass with instructions."""
    name = "System prompt injection"

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        if self._llm is None:
            return []
        text = full_text[:8000]
        prompt = (
            "You are a policy extraction assistant. Extract every policy, rule, and obligation "
            "from the following document. For each policy, provide:\n"
            "- domain: refund/privacy/security/hr/escalation/other\n"
            "- description: brief summary\n"
            "- conditions: array of condition objects, each with "
            '{{"type": "time_window|boolean_flag|amount_threshold|customer_tier", '
            '"parameter": "variable_name", "operator": "<=|==|>=", "value": ...}}\n'
            "- action_type: required/prohibited/conditional/fallback\n\n"
            f"Document:\n{text}\n\n"
            'Return JSON: {{"policies": [...]}}'
        )
        try:
            result = self._llm.invoke_json(prompt)
            policies = result.get("policies", []) if isinstance(result, dict) else result if isinstance(result, list) else []
            return [{
                "domain": p.get("domain", "other"),
                "description": p.get("description", ""),
                "conditions": p.get("conditions", []),
                "action_type": p.get("action_type", "other"),
                "keywords": [],
            } for p in policies]
        except Exception as e:
            logger.warning("System prompt extraction failed: %s", e)
            return []


class SystemPromptEnforcer(BaselineEnforcer):
    """Put full policy text in system prompt — LLM self-judges compliance."""
    name = "System prompt injection"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        if llm_client is None:
            return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}
        rules_text = ""
        if bundle:
            for r in bundle.get("conditional_rules", [])[:10]:
                conds = " AND ".join(f"{c['var']} {c['op']} {c['value']}" for c in r.get("conditions", []))
                rules_text += f"- {r['policy_id']}: IF {conds} THEN {r['action']['type']}:{r['action']['value']}\n"
            for c in bundle.get("constraints", [])[:5]:
                rules_text += f"- CONSTRAINT: {c['constraint']}\n"
        prompt = (
            f"POLICIES:\n{rules_text}\n\n"
            f"Query: {query}\nResponse: {response}\n\n"
            "Does this response comply with ALL policies? "
            'Return JSON: {{"compliant": true/false, "violations": [...], "score": 0.0-1.0}}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            compliant = result.get("compliant", True)
            violations = result.get("violations", [])
            score = float(result.get("score", 1.0 if compliant else 0.0))
            action = "pass" if compliant and score >= 0.7 else "escalate"
            return {"action": action, "violations": violations, "score": score, "latency_s": time.time() - t0}
        except Exception:
            return {"action": "pass", "violations": [], "score": 0.5, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Few-shot prompting (examples + single pass)
# ─────────────────────────────────────────────────────────────────────────────

_FEW_SHOT_EXTRACTION_EXAMPLES = """Example 1:
Input: "Customers may return items within 30 days with receipt for full refund."
Output: {"domain": "refund", "description": "30-day return with receipt", "conditions": [{"type": "time_window", "parameter": "days_since_purchase", "operator": "<=", "value": 30}], "action_type": "required"}

Example 2:
Input: "Personal health information must not be disclosed without authorization."
Output: {"domain": "privacy", "description": "PHI disclosure prohibition", "conditions": [{"type": "boolean_flag", "parameter": "authorized", "operator": "==", "value": false}], "action_type": "prohibited"}
"""


class FewShotExtractor(BaselineExtractor):
    """Few-shot prompted extraction — 2 examples + single LLM call."""
    name = "Few-shot prompting"

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        if self._llm is None:
            return []
        text = full_text[:8000]
        prompt = (
            f"Extract policies from this document. Follow these examples:\n\n"
            f"{_FEW_SHOT_EXTRACTION_EXAMPLES}\n"
            f"Document:\n{text}\n\n"
            'Return JSON: {{"policies": [...]}} with same format as examples.'
        )
        try:
            result = self._llm.invoke_json(prompt)
            policies = result.get("policies", []) if isinstance(result, dict) else result if isinstance(result, list) else []
            return [{
                "domain": p.get("domain", "other"),
                "description": p.get("description", ""),
                "conditions": p.get("conditions", []),
                "action_type": p.get("action_type", "other"),
                "keywords": [],
            } for p in policies]
        except Exception as e:
            logger.warning("Few-shot extraction failed: %s", e)
            return []


class FewShotEnforcer(BaselineEnforcer):
    """Few-shot enforcement: examples of compliant/non-compliant, then LLM judges."""
    name = "Few-shot prompting"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        if llm_client is None:
            return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}
        prompt = (
            "Judge if this response complies with organizational policies.\n\n"
            'Example 1: Response "Your SSN 123-45-6789 shows..." → {"compliant": false, "violations": ["PII_disclosure"], "score": 0.0}\n'
            'Example 2: Response "Per our return policy, you are eligible..." → {"compliant": true, "violations": [], "score": 1.0}\n\n'
            f"Query: {query}\nResponse: {response}\n\n"
            'Return JSON: {{"compliant": true/false, "violations": [...], "score": 0.0-1.0}}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            compliant = result.get("compliant", True)
            violations = result.get("violations", [])
            score = float(result.get("score", 1.0 if compliant else 0.0))
            action = "pass" if compliant and score >= 0.7 else "escalate"
            return {"action": action, "violations": violations, "score": score, "latency_s": time.time() - t0}
        except Exception:
            return {"action": "pass", "violations": [], "score": 0.5, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 7. RAG retrieval only (embedding retrieval + LLM generation, no formal checks)
# ─────────────────────────────────────────────────────────────────────────────

class RAGOnlyExtractor(BaselineExtractor):
    """Chunk document, embed, retrieve policy-like chunks, LLM extracts from retrieved."""
    name = "RAG retrieval only"

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        if self._llm is None:
            return []
        model = _get_semantic_model()
        if model is None:
            # Fallback: use system prompt extraction when embeddings unavailable
            logger.info("RAG extractor falling back to system-prompt extraction (no embeddings)")
            sp = SystemPromptExtractor(self._llm)
            return sp.extract(sections, full_text)

        import numpy as np

        # Chunk into sections
        chunks = []
        for sec in sections:
            paras = sec.get("paragraphs", [])
            text = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in paras)
            if len(text.strip()) > 20:
                chunks.append(text)
        if not chunks:
            return []

        # Retrieve top-k policy-relevant chunks
        query_emb = model.encode(["organizational policy rule obligation requirement"], convert_to_numpy=True)
        chunk_embs = model.encode(chunks, convert_to_numpy=True)
        sims = np.dot(query_emb, chunk_embs.T).flatten()
        top_k = min(10, len(chunks))
        top_indices = np.argsort(sims)[-top_k:][::-1]
        retrieved = "\n\n---\n\n".join(chunks[i][:500] for i in top_indices)

        prompt = (
            "Extract all policies from these retrieved document sections.\n\n"
            f"Sections:\n{retrieved}\n\n"
            'Return JSON: {{"policies": [{{"domain": "...", "description": "...", "conditions": [{{"type": "...", "parameter": "...", "operator": "..."}}], "action_type": "..."}}]}}'
        )
        try:
            result = self._llm.invoke_json(prompt)
            policies = result.get("policies", []) if isinstance(result, dict) else result if isinstance(result, list) else []
            return [{
                "domain": p.get("domain", "other"),
                "description": p.get("description", ""),
                "conditions": p.get("conditions", []),
                "action_type": p.get("action_type", "other"),
                "keywords": [],
            } for p in policies]
        except Exception as e:
            logger.warning("RAG extraction failed: %s", e)
            return []


class RAGOnlyEnforcer(BaselineEnforcer):
    """Retrieve relevant policy chunks, include in prompt, LLM judges — no formal checks."""
    name = "RAG retrieval only"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        if llm_client is None:
            return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}
        try:
            model = _get_semantic_model()
        except Exception:
            model = None
        # Retrieve relevant rules by embedding similarity (or fall back to all rules)
        context_text = ""
        if bundle:
            rules = bundle.get("conditional_rules", [])
            if rules and model is not None:
                import numpy as np
                rule_texts = [f"{r['policy_id']}: {r['action']['type']}:{r['action']['value']}" for r in rules]
                query_emb = model.encode([query], convert_to_numpy=True)
                rule_embs = model.encode(rule_texts, convert_to_numpy=True)
                sims = np.dot(query_emb, rule_embs.T).flatten()
                top_k = min(5, len(rules))
                top_indices = np.argsort(sims)[-top_k:][::-1]
                context_text = "\n".join(rule_texts[i] for i in top_indices)
            elif rules:
                # Fallback: use first few rules as context (no embedding retrieval)
                rule_texts = [f"{r['policy_id']}: {r['action']['type']}:{r['action']['value']}" for r in rules[:5]]
                context_text = "\n".join(rule_texts)
        prompt = (
            f"Relevant policies:\n{context_text}\n\n"
            f"Query: {query}\nResponse: {response}\n\n"
            "Does this response comply with the relevant policies? "
            'Return JSON: {{"compliant": true/false, "violations": [...], "score": 0.0-1.0}}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            compliant = result.get("compliant", True)
            violations = result.get("violations", [])
            score = float(result.get("score", 1.0 if compliant else 0.0))
            action = "pass" if compliant and score >= 0.7 else "escalate"
            return {"action": action, "violations": violations, "score": score, "latency_s": time.time() - t0}
        except Exception:
            return {"action": "pass", "violations": [], "score": 0.5, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 8. LLM zero-shot (conflict detection via LLM, single-pass extraction)
# ─────────────────────────────────────────────────────────────────────────────

class LLMZeroShotExtractor(BaselineExtractor):
    """Single-pass structured LLM extraction (like system prompt but more structured)."""
    name = "LLM zero-shot (conflict)"

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        # Same as system prompt extraction
        sp = SystemPromptExtractor(self._llm)
        return sp.extract(sections, full_text)


class LLMZeroShotEnforcer(BaselineEnforcer):
    """LLM judges compliance directly — zero-shot, no formal verification."""
    name = "LLM zero-shot (conflict)"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        if llm_client is None:
            return {"action": "pass", "violations": [], "score": 1.0, "latency_s": time.time() - t0}
        rules_text = ""
        if bundle:
            for r in bundle.get("conditional_rules", [])[:15]:
                conds = " AND ".join(f"{c['var']} {c['op']} {c['value']}" for c in r.get("conditions", []))
                rules_text += f"- {r['policy_id']}: IF {conds} THEN {r['action']['type']}:{r['action']['value']}\n"
        prompt = (
            "You are a compliance judge. Given these policy rules:\n"
            f"{rules_text}\n"
            f"Query: {query}\n"
            f"Response: {response}\n\n"
            "Score compliance 0.0-1.0. Identify violations. Check for PII, guarantees, policy violations.\n"
            'Return JSON: {{"score": float, "violations": [...], "compliant": true/false}}'
        )
        try:
            result = llm_client.invoke_json(prompt)
            compliant = result.get("compliant", True)
            violations = result.get("violations", [])
            score = float(result.get("score", 1.0 if compliant else 0.0))
            action = "pass" if compliant and score >= 0.7 else "escalate"
            return {"action": action, "violations": violations, "score": score, "latency_s": time.time() - t0}
        except Exception:
            return {"action": "pass", "violations": [], "score": 0.5, "latency_s": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# 9. RAG + Z3 hybrid (retrieval + SMT, no scaffold/judge)
# ─────────────────────────────────────────────────────────────────────────────

class RAGZ3HybridExtractor(BaselineExtractor):
    """RAG retrieval extraction + Z3 conflict detection."""
    name = "RAG + Z3 hybrid"

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    def extract(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        # Uses RAG extractor which falls back to system-prompt when no embeddings
        rag = RAGOnlyExtractor(self._llm)
        return rag.extract(sections, full_text)


class RAGZ3HybridEnforcer(BaselineEnforcer):
    """Combines RAG retrieval with Z3 verification — no scaffold injection, no judge LLM."""
    name = "RAG + Z3 hybrid"

    def enforce(self, query: str, response: str, bundle=None, llm_client=None) -> Dict[str, Any]:
        t0 = time.time()
        violations: List[str] = []
        # PII hard gate (same as PolicyLLM)
        for name, pat in _PII_PATTERNS.items():
            m = pat.search(response)
            if m:
                violations.append(f"{name}: matched '{m.group()}'")
        if violations:
            return {"action": "escalate", "violations": violations, "score": 0.0, "latency_s": time.time() - t0}
        # Z3 verification (same as SMT-only)
        smt_enforcer = SMTOnlyEnforcer()
        smt_result = smt_enforcer.enforce(query, response, bundle, llm_client=None)
        # RAG-based LLM check
        rag_enforcer = RAGOnlyEnforcer()
        rag_result = rag_enforcer.enforce(query, response, bundle, llm_client)
        # Combine: stricter of the two
        combined_violations = smt_result["violations"] + rag_result["violations"]
        combined_score = 0.6 * smt_result["score"] + 0.4 * rag_result["score"]
        action = "pass" if combined_score >= 0.7 and not combined_violations else "escalate"
        return {
            "action": action,
            "violations": combined_violations,
            "score": combined_score,
            "latency_s": time.time() - t0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry: all baselines in evaluation order
# ─────────────────────────────────────────────────────────────────────────────

def get_all_baselines(llm_client: Optional[Any] = None) -> List[Tuple[BaselineExtractor, BaselineEnforcer]]:
    """Return all (extractor, enforcer) pairs in comparison table order."""
    return [
        (VanillaLLMExtractor(llm_client), VanillaLLMEnforcer()),
        (SystemPromptExtractor(llm_client), SystemPromptEnforcer()),
        (FewShotExtractor(llm_client), FewShotEnforcer()),
        (RAGOnlyExtractor(llm_client), RAGOnlyEnforcer()),
        (KeywordRulesExtractor(), KeywordRulesEnforcer()),
        (SemanticRulesExtractor(), SemanticRulesEnforcer()),
        (SMTOnlyExtractor(), SMTOnlyEnforcer()),
        (LLMZeroShotExtractor(llm_client), LLMZeroShotEnforcer()),
        (RAGZ3HybridExtractor(llm_client), RAGZ3HybridEnforcer()),
    ]


def get_non_api_baselines() -> List[Tuple[BaselineExtractor, BaselineEnforcer]]:
    """Return baselines that do NOT require LLM API access."""
    return [
        (KeywordRulesExtractor(), KeywordRulesEnforcer()),
        (SemanticRulesExtractor(), SemanticRulesEnforcer()),
        (SMTOnlyExtractor(), SMTOnlyEnforcer()),
    ]
