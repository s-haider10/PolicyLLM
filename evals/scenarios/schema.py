"""Pydantic models for evaluation scenarios."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExpectedRegex(BaseModel):
    """Expected regex check outcomes."""
    should_flag: List[str] = Field(default_factory=list, description="Pattern names that should be flagged (e.g. ssn, email)")
    should_pass: bool = Field(default=True, description="Whether regex check should pass overall")


class ExpectedSMT(BaseModel):
    """Expected SMT check outcomes."""
    should_pass: bool = Field(default=True, description="Whether SMT check should find no violations")
    expected_violations: List[str] = Field(default_factory=list, description="Expected SMT violation strings")


class ExpectedJudge(BaseModel):
    """Expected judge check outcomes."""
    min_score: float = Field(default=0.0, description="Minimum expected judge score")
    max_score: float = Field(default=1.0, description="Maximum expected judge score")


class ExpectedCoverage(BaseModel):
    """Expected coverage check outcomes."""
    min_score: float = Field(default=0.0, description="Minimum expected coverage score")


class EvalScenario(BaseModel):
    """A single evaluation scenario."""
    id: str = Field(description="Unique scenario identifier")
    name: str = Field(description="Human-readable scenario name")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering (e.g. pii, regex, smt)")
    query: str = Field(description="User query to enforce")
    response: Optional[str] = Field(default=None, description="Pre-generated response (if None, LLM generates one)")
    expected_action: str = Field(description="Expected compliance action: pass, auto_correct, regenerate, escalate")
    expected_score_min: Optional[float] = Field(default=None, description="Minimum expected compliance score")
    expected_score_max: Optional[float] = Field(default=None, description="Maximum expected compliance score")
    expected_violations: List[str] = Field(default_factory=list, description="Expected violation substrings")
    expected_regex: Optional[ExpectedRegex] = Field(default=None)
    expected_smt: Optional[ExpectedSMT] = Field(default=None)
    expected_judge: Optional[ExpectedJudge] = Field(default=None)
    expected_coverage: Optional[ExpectedCoverage] = Field(default=None)
    determinism_runs: int = Field(default=1, description="Number of runs for determinism check (>1 to test consistency)")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional scenario-specific data")


class EvalSuite(BaseModel):
    """A collection of evaluation scenarios."""
    name: str = Field(description="Suite name")
    bundle_path: str = Field(description="Path to the compiled_policy_bundle.json to use")
    description: str = Field(default="", description="Suite description")
    scenarios: List[EvalScenario] = Field(description="List of scenarios")
