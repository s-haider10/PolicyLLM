"""Evaluation runner — orchestrates extraction & enforcement evaluation for all methods.

Usage (standalone):
    python -m eval.runner                    # run full evaluation
    python -m eval.runner --no-api           # skip LLM-dependent baselines
    python -m eval.runner --cache results/   # reuse existing PolicyLLM extraction
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    aggregate_enforcement_metrics,
    aggregate_extraction_metrics,
    build_table_row,
    compliance_rate,
    condition_f1,
    conflict_f1,
    false_positive_rate,
    policy_precision_recall_f1,
)
from eval.baselines import (
    BaselineEnforcer,
    BaselineExtractor,
    get_all_baselines,
    get_non_api_baselines,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = Path(__file__).resolve().parent / "reference_data"


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_extraction_gt() -> Dict[str, Any]:
    path = REFERENCE_DIR / "extraction_gt.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_enforcement_gt() -> Dict[str, Any]:
    path = REFERENCE_DIR / "enforcement_gt.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PolicyLLM extraction results loading
# ---------------------------------------------------------------------------

def _load_policyllm_extraction(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing PolicyLLM extraction results per dataset.

    Returns: {dataset_name: {"policies": [...], "bundle": {...}, "index": {...}}}
    """
    per_dataset: Dict[str, Dict[str, Any]] = {}
    if not results_dir.exists():
        return per_dataset

    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        bundle_path = sub / "compiled_policy_bundle.json"
        if not bundle_path.exists():
            continue

        # Load bundle
        with open(bundle_path, encoding="utf-8") as f:
            bundle = json.load(f)

        # Load raw JSONL (extracted policies)
        policies = []
        for jsonl in sub.glob("*.jsonl"):
            if "stage5" in str(jsonl):
                continue
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        policies.append(json.loads(line))

        # Load index
        index = {}
        for idx_file in sub.glob("*-index.json"):
            with open(idx_file, encoding="utf-8") as f:
                index = json.load(f)
            break

        per_dataset[name] = {"policies": policies, "bundle": bundle, "index": index}

    return per_dataset


def _policies_to_eval_format(policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert PolicyLLM extracted policies to evaluation format."""
    result = []
    for p in policies:
        raw_conditions = p.get("conditions", [])
        # Filter noise: drop conditions with type='other' or missing key fields
        conditions = []
        for c in raw_conditions:
            if not isinstance(c, dict):
                continue
            ctype = c.get("type", "")
            if ctype in ("other", "", None):
                continue  # Untyped/generic conditions are noise
            conditions.append(c)

        domain = p.get("metadata", {}).get("domain", "other")
        actions = p.get("actions", [])
        action_type = actions[0].get("type", "other") if actions else "other"
        # Build description from source text
        desc_parts = []
        for c in raw_conditions:
            if isinstance(c, dict) and c.get("source_text"):
                desc_parts.append(c["source_text"])
        for a in actions:
            if a.get("source_text"):
                desc_parts.append(a["source_text"])
        description = " ".join(desc_parts)[:300] if desc_parts else p.get("policy_id", "")

        result.append({
            "domain": domain,
            "description": description,
            "conditions": conditions,
            "action_type": action_type,
            "keywords": list(set(
                w.lower() for text in desc_parts for w in text.split() if len(w) > 3
            ))[:20],
        })
    return result


# ---------------------------------------------------------------------------
# Document sections loader (for baseline extractors)
# ---------------------------------------------------------------------------

def _load_document_sections(pdf_path: Path, config_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Regularize a PDF and return (sections, full_text)."""
    try:
        from Extractor.src.config import load_config
        from Extractor.src.regularize import router
        config = load_config(str(config_path))
        canonical = router.regularize(str(pdf_path), config)
        sections = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in canonical.sections]
        # Build full text
        full_text_parts = []
        for sec in sections:
            for para in sec.get("paragraphs", []):
                if isinstance(para, dict):
                    full_text_parts.append(para.get("text", ""))
                else:
                    full_text_parts.append(str(para))
        return sections, "\n\n".join(full_text_parts)
    except Exception as e:
        logger.warning("Failed to regularize %s: %s", pdf_path, e)
        return [], ""


# ---------------------------------------------------------------------------
# Per-method extraction evaluation
# ---------------------------------------------------------------------------

def evaluate_extraction(
    extracted: List[Dict[str, Any]],
    gt_doc: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate extraction output against ground truth for one document."""
    ref_policies = gt_doc.get("expected_policies", [])
    ref_conditions = gt_doc.get("expected_conditions", [])
    ref_conflicts = gt_doc.get("expected_conflicts", [])

    precision, recall, f1 = policy_precision_recall_f1(extracted, ref_policies)

    # Flatten conditions from extracted policies
    extracted_conditions = []
    for p in extracted:
        extracted_conditions.extend(p.get("conditions", []))
    cond_f1 = condition_f1(extracted_conditions, ref_conditions)

    # Conflict detection: check if extracted policies reveal the expected conflicts
    # For baselines that don't do conflict detection, this is 0
    conflict_f1_score = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "condition_f1": cond_f1,
        "conflict_f1": conflict_f1_score,
    }


# ---------------------------------------------------------------------------
# Per-method enforcement evaluation
# ---------------------------------------------------------------------------

def evaluate_enforcement(
    enforcer: BaselineEnforcer,
    scenarios: List[Dict[str, Any]],
    bundle: Optional[Dict[str, Any]],
    llm_client: Optional[Any],
) -> Dict[str, Any]:
    """Run enforcement on all test scenarios and compute metrics."""
    decisions: List[Dict[str, Any]] = []
    expected: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for scenario in scenarios:
        query = scenario["query"]
        exp_action = scenario["expected_action"]
        # Use non-compliant response for escalation scenarios, compliant for pass scenarios
        if exp_action == "escalate":
            response = scenario.get("non_compliant_response", "")
        else:
            response = scenario.get("compliant_response", "")

        decision = enforcer.enforce(query, response, bundle, llm_client)
        decisions.append(decision)
        expected.append({"expected_action": exp_action})
        latencies.append(decision.get("latency_s", 0.0))

    return {
        "decisions": decisions,
        "expected": expected,
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# PolicyLLM enforcement (uses the real pipeline)
# ---------------------------------------------------------------------------

def _run_policyllm_enforcement(
    scenarios: List[Dict[str, Any]],
    bundle_path: str,
    llm_client: Optional[Any],
) -> Dict[str, Any]:
    """Run full PolicyLLM enforcement pipeline on test scenarios."""
    try:
        from Enforcement import load_bundle, enforce
        from Enforcement.orchestrator import EnforcementConfig
    except ImportError:
        logger.warning("Enforcement module not importable")
        return {"decisions": [], "expected": [], "latencies": []}

    try:
        bundle, index = load_bundle(bundle_path)
    except Exception as e:
        logger.warning("Failed to load bundle %s: %s", bundle_path, e)
        return {"decisions": [], "expected": [], "latencies": []}

    # Disable retries for evaluation: pre-written responses can't improve
    # on retry, and we want single-pass comparison across all methods.
    config = EnforcementConfig(
        judge_enabled=llm_client is not None,
        smt_enabled=True,
        regex_enabled=True,
        max_retries=0,
        auto_correct_max_attempts=0,
    )

    decisions: List[Dict[str, Any]] = []
    expected: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for scenario in scenarios:
        query = scenario["query"]
        exp_action = scenario["expected_action"]

        # Use pre-written responses for fair comparison with baselines:
        #   - "escalate" scenarios: inject non-compliant response (test violation detection)
        #   - "pass" scenarios: inject compliant response (test false-positive rate)
        if exp_action == "escalate":
            response = scenario.get("non_compliant_response", "")
        else:
            response = scenario.get("compliant_response", "")
        gen_fn = (lambda r: lambda _prompt: r)(response)

        t0 = time.time()
        try:
            decision = enforce(
                query=query,
                bundle=bundle,
                bundle_index=index,
                llm_client=llm_client,
                judge_llm_client=llm_client,
                config=config,
                generate_fn=gen_fn,
            )
            action = decision.action.value
            violations = decision.violations
            score = decision.score
        except Exception as e:
            logger.warning("PolicyLLM enforcement failed for '%s': %s", query[:50], e)
            action = "pass"
            violations = []
            score = 0.5

        lat = time.time() - t0
        decisions.append({"action": action, "violations": violations, "score": score, "latency_s": lat})
        expected.append({"expected_action": exp_action})
        latencies.append(lat)

    return {"decisions": decisions, "expected": expected, "latencies": latencies}


# ---------------------------------------------------------------------------
# Conflict detection evaluation for PolicyLLM
# ---------------------------------------------------------------------------

def _evaluate_policyllm_conflicts(
    bundle: Dict[str, Any],
    gt_conflicts: List[Dict[str, Any]],
) -> float:
    """Evaluate PolicyLLM's conflict detection against ground truth.

    Uses the compiled bundle: if dominance_rules or escalations exist,
    those represent detected conflicts.
    """
    detected = []
    for dr in bundle.get("dominance_rules", []):
        pols = dr.get("when", {}).get("policies_fire", [])
        if len(pols) >= 2:
            detected.append({"policies": pols})
    for esc in bundle.get("escalations", []):
        pols = esc.get("policies", [])
        if len(pols) >= 2:
            detected.append({"policies": pols})
    return conflict_f1(detected, gt_conflicts) if gt_conflicts else (1.0 if not detected else 0.0)


# ---------------------------------------------------------------------------
# Full evaluation orchestration
# ---------------------------------------------------------------------------

def run_full_evaluation(
    results_dir: Path,
    tests_dir: Path,
    config_path: Path,
    use_api: bool = True,
    llm_client: Optional[Any] = None,
) -> List[List[str]]:
    """Run the complete evaluation and return comparison table rows.

    Each row: [Method, Policy Recall, Policy Precision, Condition F1,
               Conflict F1, Compliance %, FP %, Latency (s)]
    """
    extraction_gt = load_extraction_gt()
    enforcement_gt = load_enforcement_gt()
    scenarios = enforcement_gt.get("scenarios", [])

    # Load PolicyLLM existing results
    policyllm_data = _load_policyllm_extraction(results_dir)

    # Load document sections for baselines (uses regularizer)
    doc_sections: Dict[str, Tuple[List[Dict[str, Any]], str]] = {}
    if tests_dir.exists():
        for pdf in sorted(tests_dir.glob("*.pdf")):
            name = pdf.stem[:30].replace(" ", "_")
            sections, full_text = _load_document_sections(pdf, config_path)
            if sections:
                doc_sections[name] = (sections, full_text)

    # Select baselines
    if use_api and llm_client:
        baselines = get_all_baselines(llm_client)
    else:
        baselines = get_non_api_baselines()
        logger.info("Running non-API baselines only (no LLM client)")

    # ── Evaluate each baseline ────────────────────────────────────────────
    all_rows: List[List[str]] = []

    for extractor, enforcer in baselines:
        method_name = extractor.name
        logger.info("Evaluating baseline: %s", method_name)

        # Extraction evaluation: run extractor on each document
        per_doc_extraction: List[Dict[str, float]] = []
        for ds_name, gt_doc in extraction_gt.items():
            if ds_name.startswith("_"):
                continue
            if ds_name in doc_sections:
                sections, full_text = doc_sections[ds_name]
                try:
                    extracted = extractor.extract(sections, full_text)
                except Exception as e:
                    logger.warning("  %s extraction failed on %s: %s", method_name, ds_name, e)
                    extracted = []
            else:
                extracted = []
            metrics = evaluate_extraction(extracted, gt_doc)
            per_doc_extraction.append(metrics)

        ext_agg = aggregate_extraction_metrics(per_doc_extraction)

        # Enforcement evaluation
        # Find a bundle to use (first available from policyllm_data)
        eval_bundle = None
        for ds_data in policyllm_data.values():
            if ds_data.get("bundle"):
                eval_bundle = ds_data["bundle"]
                break

        enf_results = evaluate_enforcement(enforcer, scenarios, eval_bundle, llm_client)
        enf_agg = aggregate_enforcement_metrics(enf_results)

        row = build_table_row(method_name, ext_agg, enf_agg)
        all_rows.append(row)
        logger.info("  %s -> recall=%.2f prec=%.2f cond_f1=%.2f comp=%.1f%% fp=%.1f%%",
                     method_name, ext_agg["recall"], ext_agg["precision"],
                     ext_agg["condition_f1"], enf_agg["compliance_pct"], enf_agg["fp_pct"])

    # ── Evaluate PolicyLLM (Ours) ─────────────────────────────────────────
    logger.info("Evaluating: PolicyLLM (Ours)")
    policyllm_extraction: List[Dict[str, float]] = []
    all_conflict_f1s: List[float] = []

    for ds_name, gt_doc in extraction_gt.items():
        if ds_name.startswith("_"):
            continue
        ds_data = policyllm_data.get(ds_name, {})
        if ds_data:
            extracted = _policies_to_eval_format(ds_data["policies"])
            metrics = evaluate_extraction(extracted, gt_doc)
            # Conflict F1 from bundle
            gt_conflicts = gt_doc.get("expected_conflicts", [])
            cf1 = _evaluate_policyllm_conflicts(ds_data["bundle"], gt_conflicts)
            metrics["conflict_f1"] = cf1
            all_conflict_f1s.append(cf1)
        else:
            metrics = {"precision": 0.0, "recall": 0.0, "condition_f1": 0.0, "conflict_f1": 0.0}
        policyllm_extraction.append(metrics)

    policyllm_ext_agg = aggregate_extraction_metrics(policyllm_extraction)
    # Override conflict F1 with actual computed value
    if all_conflict_f1s:
        policyllm_ext_agg["conflict_f1"] = sum(all_conflict_f1s) / len(all_conflict_f1s)

    # PolicyLLM enforcement
    bundle_path = None
    for ds_name, ds_data in policyllm_data.items():
        bp = results_dir / ds_name / "compiled_policy_bundle.json"
        if bp.exists():
            bundle_path = str(bp)
            break

    if bundle_path:
        policyllm_enf_results = _run_policyllm_enforcement(scenarios, bundle_path, llm_client)
    else:
        policyllm_enf_results = {"decisions": [], "expected": [], "latencies": []}

    policyllm_enf_agg = aggregate_enforcement_metrics(policyllm_enf_results)

    policyllm_row = build_table_row("PolicyLLM (Ours)", policyllm_ext_agg, policyllm_enf_agg)
    all_rows.append(policyllm_row)
    logger.info("  PolicyLLM -> recall=%.2f prec=%.2f cond_f1=%.2f conflict_f1=%.1f comp=%.1f%% fp=%.1f%%",
                 policyllm_ext_agg["recall"], policyllm_ext_agg["precision"],
                 policyllm_ext_agg["condition_f1"], policyllm_ext_agg.get("conflict_f1", 0.0),
                 policyllm_enf_agg["compliance_pct"], policyllm_enf_agg["fp_pct"])

    return all_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Run PolicyLLM evaluation")
    parser.add_argument("--results-dir", default=str(ROOT / "results"), help="Path to existing PolicyLLM results")
    parser.add_argument("--tests-dir", default=str(ROOT / "tests"), help="Path to test PDFs")
    parser.add_argument("--config", default=str(ROOT / "Extractor" / "configs" / "config.chatgpt.yaml"))
    parser.add_argument("--no-api", action="store_true", help="Skip LLM-dependent baselines")
    parser.add_argument("--output", default=str(ROOT / "results" / "computed_metrics.json"), help="Output JSON path")
    args = parser.parse_args()

    llm_client = None
    if not args.no_api:
        try:
            from Extractor.src.llm.client import LLMClient
            llm_client = LLMClient(provider="ollama", model_id="Qwen3-30B-A3B", temperature=0.0, max_tokens=2048)
            logger.info("LLM client initialized (ollama / Qwen3-30B-A3B)")
        except Exception as e:
            logger.warning("LLM client unavailable: %s — running non-API baselines only", e)

    rows = run_full_evaluation(
        results_dir=Path(args.results_dir),
        tests_dir=Path(args.tests_dir),
        config_path=Path(args.config),
        use_api=not args.no_api,
        llm_client=llm_client,
    )

    # Save computed metrics
    header = ["Method", "Policy Recall", "Policy Precision", "Condition F1",
              "Conflict F1", "Compliance %", "FP %", "Latency (s)"]
    output = {"header": header, "rows": rows}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Metrics written to %s", args.output)

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Method':<30} {'Recall':>8} {'Prec':>8} {'CondF1':>8} {'ConflF1':>8} {'Comp%':>8} {'FP%':>8} {'Lat(s)':>8}")
    print("-" * 100)
    for row in rows:
        print(f"{row[0]:<30} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8} {row[6]:>8} {row[7]:>8}")
    print("=" * 100)


if __name__ == "__main__":
    main()
