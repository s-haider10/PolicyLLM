"""Full enforcement pipeline orchestration."""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .audit import AuditLogger, build_audit_entry
from .bundle_loader import BundleIndex
from .duringgen import build_injection_bundle, format_full_prompt
from .postgen.judge import run_judge_check
from .postgen.regex import run_regex_check
from .postgen.smt import run_smt_check
from .pregen import build_context
from .schemas import (
    ComplianceAction,
    ComplianceDecision,
    CompiledPolicyBundle,
    CoverageResult,
    EnforcementContext,
    JudgeResult,
    PostGenReport,
    RegexResult,
    SMTResult,
)
from .scoring import build_compliance_decision, compute_coverage

logger = logging.getLogger(__name__)


@dataclass
class EnforcementConfig:
    max_retries: int = 2
    auto_correct_max_attempts: int = 1
    judge_enabled: bool = True
    smt_enabled: bool = True
    regex_enabled: bool = True
    timeout_ms: float = 30000
    generation_temperature: float = 0.0
    generation_max_tokens: int = 2048


def _run_postgen(
    response_text: str,
    context: EnforcementContext,
    bundle: CompiledPolicyBundle,
    llm_client: Any,
    judge_llm_client: Any,
    config: EnforcementConfig,
) -> PostGenReport:
    """Run all post-generation checks."""
    # Regex
    if config.regex_enabled:
        try:
            regex_result = run_regex_check(response_text, context.applicable_constraints)
        except Exception as e:
            logger.warning("Regex check failed: %s", e)
            regex_result = RegexResult(passed=True, flags=[], score=1.0)
    else:
        regex_result = RegexResult(passed=True, flags=[], score=1.0)

    # SMT (hard dependency — let it raise)
    if config.smt_enabled:
        smt_result = run_smt_check(response_text, context, bundle, llm_client)
    else:
        smt_result = SMTResult(passed=True, violations=[], score=1.0)

    # Coverage
    coverage_result = compute_coverage(context, response_text)

    # Judge
    if config.judge_enabled:
        try:
            judge_result = run_judge_check(response_text, context, judge_llm_client)
        except Exception as e:
            logger.warning("Judge check failed: %s", e)
            judge_result = JudgeResult(score=0.5, issues=["judge_unavailable"])
    else:
        judge_result = JudgeResult(score=1.0, issues=[])

    return PostGenReport(
        regex_result=regex_result,
        smt_result=smt_result,
        judge_result=judge_result,
        coverage_result=coverage_result,
    )


def enforce(
    query: str,
    bundle: CompiledPolicyBundle,
    bundle_index: BundleIndex,
    llm_client: Any,
    judge_llm_client: Optional[Any] = None,
    config: Optional[EnforcementConfig] = None,
    session_id: Optional[str] = None,
    generate_fn: Optional[Callable[[Dict[str, str]], str]] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> ComplianceDecision:
    """Full enforcement pipeline.

    Args:
        query: User query string.
        bundle: Loaded CompiledPolicyBundle.
        bundle_index: BundleIndex for fast lookups.
        llm_client: LLM client for generation and fact extraction.
        judge_llm_client: Optional separate client for judge (defaults to llm_client).
        config: EnforcementConfig overrides.
        session_id: Optional session ID (auto-generated if None).
        generate_fn: Optional callback: {"system": str, "user": str} -> response str.
        audit_logger: Optional AuditLogger for structured logging.

    Returns:
        ComplianceDecision with score, action, violations, evidence.
    """
    t0 = time.time()
    cfg = config or EnforcementConfig()
    judge_client = judge_llm_client or llm_client

    # --- PREGEN ---
    context = build_context(query, bundle, bundle_index, session_id, llm_client)
    logger.info("Pregen: domain=%s intent=%s rules=%d", context.domain, context.intent, len(context.applicable_rules))

    if not context.applicable_rules and context.domain == "unknown":
        duration_ms = (time.time() - t0) * 1000
        decision = ComplianceDecision(
            score=1.0,
            action=ComplianceAction.PASS,
            violations=[],
            evidence={"note": "no applicable policies found"},
            audit_trail={"duration_ms": duration_ms},
            llm_response="",
        )
        if audit_logger:
            entry = build_audit_entry(context, None, decision, duration_ms)
            audit_logger.log(entry)
        return decision

    # --- DURINGGEN ---
    injection = build_injection_bundle(context, bundle)
    prompt = format_full_prompt(query, injection)

    # --- GENERATE ---
    if generate_fn:
        response = generate_fn(prompt)
    else:
        full_prompt = prompt["system"] + "\n\n" + prompt["user"] if prompt["system"] else prompt["user"]
        try:
            result = llm_client.invoke_json(full_prompt)
            response = result if isinstance(result, str) else str(result)
        except Exception:
            response = ""

    # --- POSTGEN ---
    report = _run_postgen(response, context, bundle, llm_client, judge_client, cfg)
    decision = build_compliance_decision(report, response, context)
    logger.info("Postgen: score=%.3f action=%s", decision.score, decision.action.value)

    # --- ACTION ROUTING ---
    retries = 0
    while decision.action in (ComplianceAction.AUTO_CORRECT, ComplianceAction.REGENERATE):
        if decision.action == ComplianceAction.AUTO_CORRECT and retries < cfg.auto_correct_max_attempts:
            # Append violation hints
            hint = "\n".join(f"FIX: {v}" for v in decision.violations[:5])
            corrected_prompt = {
                "system": prompt["system"],
                "user": prompt["user"] + f"\n\nPrevious issues to fix:\n{hint}",
            }
            if generate_fn:
                new_response = generate_fn(corrected_prompt)
            else:
                full = corrected_prompt["system"] + "\n\n" + corrected_prompt["user"] if corrected_prompt["system"] else corrected_prompt["user"]
                try:
                    r = llm_client.invoke_json(full)
                    new_response = r if isinstance(r, str) else str(r)
                except Exception:
                    break

            new_report = _run_postgen(new_response, context, bundle, llm_client, judge_client, cfg)
            new_decision = build_compliance_decision(new_report, new_response, context)
            if new_decision.score >= 0.95:
                new_decision.corrected_response = new_response
                decision = new_decision
                report = new_report
                break
            retries += 1
            decision = new_decision
            report = new_report

        elif decision.action == ComplianceAction.REGENERATE and retries < cfg.max_retries:
            # Tighten scaffold with DO NOT directives
            do_nots = "\n".join(f"DO NOT: {v}" for v in decision.violations[:5])
            tighter_prompt = {
                "system": prompt["system"],
                "user": prompt["user"] + f"\n\nSTRICT CONSTRAINTS:\n{do_nots}",
            }
            if generate_fn:
                new_response = generate_fn(tighter_prompt)
            else:
                full = tighter_prompt["system"] + "\n\n" + tighter_prompt["user"] if tighter_prompt["system"] else tighter_prompt["user"]
                try:
                    r = llm_client.invoke_json(full)
                    new_response = r if isinstance(r, str) else str(r)
                except Exception:
                    break

            new_report = _run_postgen(new_response, context, bundle, llm_client, judge_client, cfg)
            new_decision = build_compliance_decision(new_report, new_response, context)
            retries += 1
            if new_decision.action == ComplianceAction.PASS:
                decision = new_decision
                report = new_report
                break
            decision = new_decision
            report = new_report
        else:
            # All retries exhausted — escalate
            decision.action = ComplianceAction.ESCALATE
            break

    duration_ms = (time.time() - t0) * 1000
    decision.audit_trail["duration_ms"] = duration_ms

    # --- AUDIT ---
    if audit_logger:
        entry = build_audit_entry(context, report, decision, duration_ms)
        audit_logger.log(entry)

    return decision
