"""Load and filter evaluation suites from JSON files."""
import json
from pathlib import Path
from typing import List, Optional, Set

from .schema import EvalSuite, EvalScenario


def load_suite(path: str) -> EvalSuite:
    """Load an EvalSuite from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalSuite.model_validate(data)


def load_scenarios_jsonl(path: str, bundle_path: str, name: str = "jsonl_suite") -> EvalSuite:
    """Load scenarios from a JSONL file (one scenario per line)."""
    scenarios = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(EvalScenario.model_validate(json.loads(line)))
    return EvalSuite(name=name, bundle_path=bundle_path, scenarios=scenarios)


def filter_scenarios(
    suite: EvalSuite,
    tags: Optional[Set[str]] = None,
    ids: Optional[Set[str]] = None,
) -> EvalSuite:
    """Return a new suite with only scenarios matching the given tags or ids."""
    filtered = suite.scenarios
    if tags:
        filtered = [s for s in filtered if tags & set(s.tags)]
    if ids:
        filtered = [s for s in filtered if s.id in ids]
    return EvalSuite(
        name=suite.name,
        bundle_path=suite.bundle_path,
        description=suite.description,
        scenarios=filtered,
    )
