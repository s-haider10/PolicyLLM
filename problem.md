# Hard-Coded Data Checkpoints and Paths in PolicyLLM Pipeline

**Issue Status:** CONFIRMED  
**Severity:** HIGH - Affects maintainability, testability, and deployment flexibility  
**Date:** February 14, 2026

---

## Executive Summary

The PolicyLLM pipeline contains numerous hard-coded file paths, directory names, and data checkpoint locations throughout the codebase. This creates significant issues for:
- Multi-tenant deployments
- Parallel pipeline execution
- Testing in different environments
- CI/CD integration
- Production deployment flexibility

---

## 1. Hard-Coded Output Directories

### 1.1 Main Pipeline Defaults

**Location:** `main.py`

```python
# Line 122
extract_out = args.out or "out"

# Line 221
p_extract.add_argument("--out", default="out", help="Output directory")

# Line 250
p_run.add_argument("--out", default="out", help="Output directory for extraction artifacts")
```

**Impact:** All extraction outputs default to `./out/` directory. When running multiple pipeline instances, outputs collide unless explicitly overridden.

### 1.2 Extractor CLI

**Location:** `Extractor/src/cli.py`

```python
# Line 10
parser.add_argument("--out", dest="out_dir", default="out", help="Output directory for JSON artifacts")
```

**Impact:** Same issue - hard-coded `out` directory conflicts with concurrent runs.

### 1.3 Stage5 Checkpoint Directory

**Location:** `Extractor/src/pipeline.py`

```python
# Line 210
stage5_dir = os.path.join(output_dir, "stage5")

# Line 229
stage5_dir = os.path.join(output_dir, "stage5")

# Line 231
path = os.path.join(stage5_dir, f"{doc_id}-{batch_id}-stage5.jsonl")
```

**Impact:** The `stage5/` subdirectory structure is hard-coded. Cannot be configured or changed without modifying source code.

---

## 2. Hard-Coded Configuration File Paths

### 2.1 Default Config Path

**Location:** `main.py`

```python
# Line 222
p_extract.add_argument("--config", default="Extractor/configs/config.example.yaml", help="Extractor YAML config")

# Line 252
p_run.add_argument("--config", default="Extractor/configs/config.example.yaml", help="Extractor YAML config")
```

**Location:** `Extractor/src/cli.py`

```python
# Line 13
parser.add_argument("--config", dest="config_path", default="configs/config.example.yaml", help="Path to YAML config")
```

**Impact:** 
- Assumes specific directory structure (`Extractor/configs/` or `configs/`)
- Makes it difficult to use different configs for different environments (dev/staging/prod)
- Breaks when running from different working directories

### 2.2 Test Config Paths

**Location:** `tests/test_e2e_pipeline.py`

```python
# Line 54
config = load_config("Extractor/configs/config.chatgpt.yaml")
```

**Location:** `tests/test_extraction.py`

```python
# Line 34
config = load_config("Extractor/configs/config.chatgpt.yaml")
```

**Impact:** Tests assume specific config file location, making them fragile to directory structure changes.

---

## 3. Hard-Coded Audit Log Paths

### 3.1 Enforcement Audit Logging

**Location:** `Enforcement/audit.py`

```python
# Line 19
def __init__(self, log_path: str = "audit/enforcement.jsonl"):
```

**Location:** `Enforcement/cli.py`

```python
# Line 15
parser.add_argument("--audit-log", default="audit/enforcement.jsonl", help="Audit log path")
```

**Location:** `main.py`

```python
# Line 241
p_enforce.add_argument("--audit-log", default="audit/enforcement.jsonl", help="Audit log path")

# Line 257
p_run.add_argument("--audit-log", default="audit/enforcement.jsonl", help="Audit log path")
```

**Impact:**
- All enforcement runs write to same `audit/enforcement.jsonl` by default
- Concurrent enforcement operations will corrupt the audit log
- No isolation between different tenants/sessions
- Violates audit integrity requirements for multi-tenant systems

---

## 4. Hard-Coded Test Data Paths

### 4.1 Test Input Files

**Location:** `tests/test_e2e_pipeline.py`

```python
# Line 43
policy_doc = "tests/data/sample_policy.md"

# Line 148
with open("tests/data/test_queries.json", "r") as f:
```

**Location:** `tests/test_extraction.py`

```python
# Line 35
policy_file = "tests/data/sample_policy.md"
```

**Location:** `tests/test_enforcement.py`

```python
# Line 23
with open("tests/data/test_queries.json", "r") as f:
```

**Impact:** Tests fail if run from different working directories or if test data is reorganized.

### 4.2 Test Output Directories

**Location:** `tests/test_e2e_pipeline.py`

```python
# Line 38
test_dir = "tests/output/e2e"
```

**Location:** `tests/test_extraction.py`

```python
# Line 28
test_dir = "tests/output/extraction"
```

**Impact:** 
- Test outputs are not isolated per test run
- Parallel test execution causes conflicts
- Leftover test data from failed runs may affect subsequent tests

### 4.3 Test Fixture Paths

**Location:** Multiple test files

```python
# tests/test_enforcement.py Line 48
bundle, bundle_index = load_bundle("Enforcement/tests/fixtures/test_bundle.json")

# tests/test_validation_dag.py Line 21
with open("Enforcement/tests/fixtures/test_bundle.json", "r") as f:

# Enforcement/tests/test_postgen.py Line 21
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")

# Enforcement/tests/test_bundle_loader.py Line 11
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")

# Enforcement/tests/test_orchestrator.py Line 10
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")

# Enforcement/tests/test_duringgen.py Line 17
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")

# Enforcement/tests/test_scoring.py Line 30
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")

# Enforcement/tests/test_pregen.py Line 15
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")

# Enforcement/tests/test_e2e.py Line 12
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_bundle.json")
```

**Impact:** Tests are tightly coupled to specific fixture locations.

---

## 5. Hard-Coded Bundle Output Names

### 5.1 Validation Module

**Location:** `Validation/cli.py`

```python
# Line 26
parser.add_argument("--out", default="compiled_policy_bundle.json", help="Output bundle path")
```

**Location:** `main.py`

```python
# Line 168
bundle_path = args.bundle_out or os.path.join(extract_out, "compiled_policy_bundle.json")
```

**Impact:** Default bundle filename is hard-coded, making it difficult to maintain multiple versions or configurations.

---

## 6. Hard-Coded Intermediate Data Structure

### 6.1 JSONL Output Naming Convention

**Location:** `Extractor/src/pipeline.py`

```python
# Line 408
policies_path = os.path.join(output_dir, f"{canonical.doc_id}-{batch_id}.jsonl")

# Line 427
index_path = os.path.join(output_dir, f"{canonical.doc_id}-{batch_id}-index.json")
```

**Impact:** 
- Filename pattern is hard-coded: `{doc_id}-{batch_id}.jsonl`
- Cannot customize naming scheme without code changes
- Difficult to implement custom organization strategies (e.g., by tenant, date, version)

### 6.2 Stage5 File Naming

**Location:** `Extractor/src/pipeline.py`

```python
# Line 220
dest_name = f"{doc_id}-{batch_id}-{os.path.basename(src)}"

# Line 231
path = os.path.join(stage5_dir, f"{doc_id}-{batch_id}-stage5.jsonl")
```

**Impact:** Stage5 checkpoint filenames follow hard-coded pattern, limiting flexibility.

---

## 7. Missing Environment-Based Configuration

### Problem

The codebase has no centralized configuration system for file paths and checkpoint locations. Each module and CLI defines its own defaults, leading to:

1. **Inconsistency:** Different modules use different path resolution strategies
2. **No environment support:** Cannot easily switch between dev/staging/prod configurations
3. **No tenant isolation:** Multi-tenant deployments require manual path overrides everywhere
4. **Testing difficulties:** Tests cannot easily use isolated temporary directories

---

## 8. Specific Issues by Pipeline Stage

### Stage 1: Extraction

**Hard-coded elements:**
- Output directory: `out/`
- Config file: `Extractor/configs/config.example.yaml`
- Stage5 subdirectory: `stage5/`
- Output filename pattern: `{doc_id}-{batch_id}.jsonl`
- Index filename pattern: `{doc_id}-{batch_id}-index.json`

### Stage 2: Validation

**Hard-coded elements:**
- Output bundle name: `compiled_policy_bundle.json`
- No intermediate checkpoint configuration

### Stage 3: Enforcement

**Hard-coded elements:**
- Audit log path: `audit/enforcement.jsonl`
- No session-specific output directories

---

## 9. Real-World Impact Examples

### Example 1: Concurrent Pipeline Execution

```bash
# Terminal 1
python main.py run doc1.pdf --query "Can I get a refund?"

# Terminal 2 (simultaneously)
python main.py run doc2.pdf --query "What's the return policy?"
```

**Problem:** Both processes write to:
- `out/` directory
- `audit/enforcement.jsonl`
- Outputs collide, audit log corrupted

### Example 2: Multi-Tenant Deployment

```python
# Tenant A
enforce(query="...", bundle=bundle_a, audit_logger=AuditLogger())

# Tenant B (concurrent)
enforce(query="...", bundle=bundle_b, audit_logger=AuditLogger())
```

**Problem:** Both tenants write to `audit/enforcement.jsonl`, violating data isolation requirements.

### Example 3: Testing in CI/CD

```bash
# Parallel pytest execution
pytest tests/ -n 4
```

**Problem:**
- Multiple test processes write to `tests/output/e2e/`, `tests/output/extraction/`
- Tests interfere with each other
- Non-deterministic failures

### Example 4: Running from Different Directory

```bash
cd /tmp
python /path/to/ConstitutionCreator/main.py extract input.pdf
```

**Problem:**
- Config file not found: `Extractor/configs/config.example.yaml`
- Output goes to unexpected location
- Audit directory created in wrong place

---

## 10. Recommended Solutions

### 10.1 Centralized Path Configuration

Create a `paths.py` configuration module:

```python
from pathlib import Path
from typing import Optional
import os

class PathConfig:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.base_dir = base_dir or Path.cwd()
        self.tenant_id = tenant_id or os.getenv("POLICYLLM_TENANT_ID", "default")
        self.session_id = session_id or os.getenv("POLICYLLM_SESSION_ID")
    
    def get_extraction_output_dir(self) -> Path:
        path = self.base_dir / "extraction" / self.tenant_id
        if self.session_id:
            path = path / self.session_id
        return path
    
    def get_audit_log_path(self) -> Path:
        return self.base_dir / "audit" / self.tenant_id / "enforcement.jsonl"
    
    def get_bundle_path(self, name: str = "bundle.json") -> Path:
        return self.base_dir / "bundles" / self.tenant_id / name
```

### 10.2 Environment Variable Support

Support environment variables for all path configurations:

```bash
export POLICYLLM_BASE_DIR=/var/lib/policyllm
export POLICYLLM_TENANT_ID=customer_acme
export POLICYLLM_SESSION_ID=session_12345
export POLICYLLM_CONFIG_FILE=/etc/policyllm/config.yaml
```

### 10.3 Per-Session Isolation

Generate unique session IDs and use them for all checkpoints:

```python
import uuid
from datetime import datetime

session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
output_dir = f"out/{session_id}"
audit_log = f"audit/{session_id}/enforcement.jsonl"
```

### 10.4 Test Isolation

Use pytest tmpdir fixtures consistently:

```python
def test_extraction(tmp_path):
    output_dir = tmp_path / "extraction"
    config = load_config(tmp_path / "test_config.yaml")
    pipeline.run_pipeline(input_path="...", output_dir=str(output_dir), ...)
```

### 10.5 Configuration Hierarchy

Implement configuration precedence:
1. Command-line arguments (highest priority)
2. Environment variables
3. Config file settings
4. Reasonable defaults (lowest priority)

---

## 11. Migration Plan

### Phase 1: Add PathConfig Module
- Create centralized path configuration system
- Maintain backward compatibility with existing defaults

### Phase 2: Refactor Main Pipeline
- Update `main.py`, `pipeline.py` to use PathConfig
- Add environment variable support
- Keep existing CLI arguments functioning

### Phase 3: Update Tests
- Use pytest `tmp_path` fixtures consistently
- Remove hard-coded test output directories
- Ensure test isolation

### Phase 4: Update Documentation
- Document new configuration options
- Provide migration guide for existing deployments
- Add multi-tenant deployment examples

---

## 12. Priority Assessment

### Critical (P0)
- **Audit log path isolation:** Security/compliance risk in multi-tenant scenarios
- **Output directory conflicts:** Breaks concurrent execution

### High (P1)
- **Config file path flexibility:** Deployment and testing difficulties
- **Stage5 checkpoint structure:** Integration with monitoring/debugging tools

### Medium (P2)
- **Test isolation:** CI/CD reliability issues
- **Bundle naming:** Version management complexity

### Low (P3)
- **Filename patterns:** Organizational preferences
- **Documentation:** User convenience

---

## Conclusion

The PolicyLLM pipeline has **extensive hard-coding of data checkpoints and file paths** throughout the codebase. This is a confirmed issue that significantly impacts:

1. **Scalability:** Cannot run multiple pipeline instances safely
2. **Multi-tenancy:** No data isolation between tenants
3. **Testability:** Tests interfere with each other
4. **Deployment:** Assumes specific directory structures
5. **Maintainability:** Changes require modifications across multiple files

**Recommendation:** Implement a centralized path configuration system with environment variable support and session-based isolation as a high-priority refactoring effort.
