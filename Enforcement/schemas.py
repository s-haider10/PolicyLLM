"""All Pydantic models for the Enforcement module — single source of truth."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bundle types (loaded from compiled_policy_bundle.json)
# ---------------------------------------------------------------------------

class VariableSchema(BaseModel):
    type: str  # "bool" | "int" | "float" | "enum"
    description: str = ""
    values: Optional[List[Any]] = None  # only for enum type


class IRCondition(BaseModel):
    var: str
    op: str  # "==" | "!=" | "<=" | ">=" | ">" | "<"
    value: Any


class IRAction(BaseModel):
    type: str
    value: Any


class RuleMetadata(BaseModel):
    domain: str = "other"
    priority: str = "company"
    owner: Optional[str] = None
    source: str = ""
    eff_date: Optional[str] = None
    regulatory_linkage: List[str] = Field(default_factory=list)


class ConditionalRule(BaseModel):
    policy_id: str
    conditions: List[IRCondition]
    action: IRAction
    metadata: RuleMetadata


class Constraint(BaseModel):
    policy_id: str
    constraint: str
    scope: str = "always"
    metadata: RuleMetadata


class PathStep(BaseModel):
    var: str
    tests: List[Dict[str, Any]]


class CompiledPath(BaseModel):
    policy_id: str
    path: List[PathStep]
    leaf_action: str
    metadata: RuleMetadata


class DominanceRule(BaseModel):
    when: Dict[str, List[str]]
    then: Dict[str, str]


class EscalationEntry(BaseModel):
    conflict_type: str
    policies: List[str]
    actions: List[str]
    priority: str = ""
    owners_to_notify: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    recommended_next_step: str = ""


class CanonicalActionMap(BaseModel):
    policy_id: str
    canonical_actions: List[str]


class BundleMetadata(BaseModel):
    generated_on: str = ""
    generator: str = ""
    policy_count: int = 0
    rule_count: int = 0
    constraint_count: int = 0
    path_count: int = 0


class CompiledPolicyBundle(BaseModel):
    schema_version: str = "1.0"
    variables: Dict[str, VariableSchema]
    conditional_rules: List[ConditionalRule]
    constraints: List[Constraint] = Field(default_factory=list)
    decision_nodes: List[str] = Field(default_factory=list)
    node_schema: Dict[str, VariableSchema] = Field(default_factory=dict)
    leaf_actions: List[str] = Field(default_factory=list)
    compiled_paths: List[CompiledPath] = Field(default_factory=list)
    dominance_rules: List[DominanceRule] = Field(default_factory=list)
    escalations: List[EscalationEntry] = Field(default_factory=list)
    canonical_action_map: List[CanonicalActionMap] = Field(default_factory=list)
    priority_lattice: Dict[str, int] = Field(default_factory=lambda: {
        "regulatory": 1, "core_values": 2, "company": 3,
        "department": 4, "situational": 5,
    })
    bundle_metadata: BundleMetadata = Field(default_factory=BundleMetadata)


# ---------------------------------------------------------------------------
# Runtime types
# ---------------------------------------------------------------------------

class EnforcementContext(BaseModel):
    session_id: str
    query: str
    domain: Optional[str] = None
    intent: Optional[str] = None
    domain_confidence: float = 0.0
    applicable_rules: List[ConditionalRule] = Field(default_factory=list)
    applicable_constraints: List[Constraint] = Field(default_factory=list)
    applicable_paths: List[CompiledPath] = Field(default_factory=list)
    dominance_applied: List[DominanceRule] = Field(default_factory=list)
    escalation_contacts: List[str] = Field(default_factory=list)
    timestamp: str = ""


class InjectionBundle(BaseModel):
    system_prompt_additions: str = ""
    scaffold_steps: List[str] = Field(default_factory=list)
    priority_guidance: str = ""
    invariant_constraints: List[str] = Field(default_factory=list)
    generation_params: Dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0.0, "max_tokens": 2048,
    })


class RegexResult(BaseModel):
    passed: bool
    flags: List[str] = Field(default_factory=list)
    score: float  # 1.0 if passed, 0.0 if any flag


class SMTResult(BaseModel):
    passed: bool
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    score: float  # 1.0 if passed, 0.0 if any violation


class JudgeResult(BaseModel):
    score: float  # 0.0 to 1.0
    issues: List[str] = Field(default_factory=list)
    explanation: str = ""


class CoverageResult(BaseModel):
    score: float
    nodes_required: List[str] = Field(default_factory=list)
    nodes_covered: List[str] = Field(default_factory=list)


class PostGenReport(BaseModel):
    regex_result: RegexResult
    smt_result: SMTResult
    judge_result: JudgeResult
    coverage_result: CoverageResult


class ComplianceAction(str, Enum):
    PASS = "pass"
    AUTO_CORRECT = "auto_correct"
    REGENERATE = "regenerate"
    ESCALATE = "escalate"


class ComplianceDecision(BaseModel):
    score: float
    action: ComplianceAction
    violations: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: Dict[str, Any] = Field(default_factory=dict)
    llm_response: str = ""
    corrected_response: Optional[str] = None


class AuditEntry(BaseModel):
    session_id: str
    timestamp: str
    query: str
    domain: Optional[str] = None
    intent: Optional[str] = None
    retrieved_policy_ids: List[str] = Field(default_factory=list)
    scaffold_hash: str = ""
    llm_response_hash: str = ""
    postgen_report: Optional[PostGenReport] = None
    compliance_score: float = 0.0
    final_action: ComplianceAction = ComplianceAction.PASS
    owners_notified: List[str] = Field(default_factory=list)
    duration_ms: float = 0.0
