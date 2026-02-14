"""Structured JSONL audit logging with SHA256 hash chain."""
import hashlib
import json
import os
from typing import Optional

from .schemas import (
    AuditEntry,
    ComplianceAction,
    ComplianceDecision,
    EnforcementContext,
    PostGenReport,
)


class AuditLogger:
    """Append-only JSONL audit log with hash chain for tamper detection."""

    def __init__(self, log_path: str = "audit/enforcement.jsonl"):
        self.log_path = log_path
        self._prev_hash: Optional[str] = None

    def log(self, entry: AuditEntry) -> str:
        """Append entry to audit log. Returns the entry hash."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        entry_json = entry.model_dump_json()
        hash_input = (self._prev_hash or "") + entry_json
        entry_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        record = {
            "entry_hash": entry_hash,
            "prev_hash": self._prev_hash,
            **json.loads(entry_json),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

        self._prev_hash = entry_hash
        return entry_hash

    def verify_integrity(self) -> bool:
        """Replay hash chain and verify all entries."""
        if not os.path.exists(self.log_path):
            return True

        prev_hash: Optional[str] = None
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stored_hash = record.pop("entry_hash", "")
                stored_prev = record.pop("prev_hash", None)

                if stored_prev != prev_hash:
                    return False

                entry_json = json.dumps(record, ensure_ascii=False, sort_keys=True)
                # Reconstruct AuditEntry to get canonical JSON
                entry = AuditEntry.model_validate(record)
                canonical_json = entry.model_dump_json()
                computed = hashlib.sha256(
                    ((prev_hash or "") + canonical_json).encode("utf-8")
                ).hexdigest()

                if computed != stored_hash:
                    return False

                prev_hash = stored_hash
        return True


def build_audit_entry(
    context: EnforcementContext,
    report: Optional[PostGenReport],
    decision: ComplianceDecision,
    duration_ms: float,
) -> AuditEntry:
    """Construct an AuditEntry from pipeline artifacts."""
    scaffold_text = "|".join(
        step.var for path in context.applicable_paths for step in path.path
    )
    scaffold_hash = hashlib.sha256(scaffold_text.encode("utf-8")).hexdigest()
    response_hash = hashlib.sha256(decision.llm_response.encode("utf-8")).hexdigest()

    return AuditEntry(
        session_id=context.session_id,
        timestamp=context.timestamp,
        query=context.query,
        domain=context.domain,
        intent=context.intent,
        retrieved_policy_ids=[r.policy_id for r in context.applicable_rules],
        scaffold_hash=scaffold_hash,
        llm_response_hash=response_hash,
        postgen_report=report,
        compliance_score=decision.score,
        final_action=decision.action,
        owners_notified=context.escalation_contacts if decision.action == ComplianceAction.ESCALATE else [],
        duration_ms=duration_ms,
    )
