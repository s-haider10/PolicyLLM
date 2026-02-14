"""Fast regex/keyword pattern matching on LLM responses."""
import re
from typing import List, Optional

from ..schemas import Constraint, RegexResult

# Default forbidden patterns (always applied)
DEFAULT_FORBIDDEN_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "credit_card": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    "password_disclosure": r"(?i)\bpassword\s*[:=]\s*\S+",
    "guarantee_promise": r"(?i)\bI\s+(?:guarantee|promise)\s+(?:you|that)\b",
    "unconditional_commit": r"(?i)\bwe\s+will\s+definitely\b",
}


def compile_constraint_patterns(constraints: List[Constraint]) -> dict[str, str]:
    """Convert constraint strings into regex patterns."""
    patterns: dict[str, str] = {}
    for c in constraints:
        text = c.constraint
        if "disclose_pii" in text.lower() or "pii" in text.lower():
            # PII-related constraints map to PII patterns (already in defaults)
            pass
        elif text.startswith("NOT(") and text.endswith(")"):
            inner = text[4:-1]
            # Create a case-insensitive literal search for the action
            safe_inner = inner.replace("_", r"[\s_]")
            patterns[f"constraint_{c.policy_id}"] = rf"(?i)\b{safe_inner}\b"
    return patterns


def run_regex_check(
    response_text: str,
    constraints: List[Constraint],
    extra_patterns: Optional[dict[str, str]] = None,
) -> RegexResult:
    """Run all regex patterns against the response text.

    Returns RegexResult with passed=True if no patterns match.
    """
    all_patterns = dict(DEFAULT_FORBIDDEN_PATTERNS)
    all_patterns.update(compile_constraint_patterns(constraints))
    if extra_patterns:
        all_patterns.update(extra_patterns)

    flags: List[str] = []
    for name, pattern in all_patterns.items():
        try:
            match = re.search(pattern, response_text)
            if match:
                flags.append(f"{name}: matched '{match.group()}' at pos {match.start()}")
        except re.error:
            continue

    passed = len(flags) == 0
    return RegexResult(passed=passed, flags=flags, score=1.0 if passed else 0.0)
