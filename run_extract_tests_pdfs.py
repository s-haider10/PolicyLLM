#!/usr/bin/env python3
"""Run PolicyLLM extract + validate on every PDF in tests/, evaluate all
baselines, and generate the ACL comparison table from **computed** metrics.

No metric is hardcoded — every number in the output tables comes from
actual pipeline execution and evaluation against reference annotations.

Usage:
    python run_extract_tests_pdfs.py                   # full run: extract + validate + eval
    python run_extract_tests_pdfs.py --tables-only      # build tables from existing results
    python run_extract_tests_pdfs.py --no-api           # skip LLM-dependent baselines
    python run_extract_tests_pdfs.py --eval-only        # skip extraction, run evaluation only
"""
import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
RESULTS_DIR = ROOT / "results"
CONFIG = ROOT / "Extractor" / "configs" / "config.chatgpt.yaml"


# ---------------------------------------------------------------------------
# Metric helpers (for per-dataset summary — unchanged)
# ---------------------------------------------------------------------------

def _read_metrics(out_dir: Path) -> dict | None:
    """Read per-dataset metrics from index + bundle in out_dir."""
    index_files = list(out_dir.glob("*-index.json"))
    index_path = index_files[0] if index_files else None
    bundle_path = out_dir / "compiled_policy_bundle.json"
    if not index_path or not bundle_path.exists():
        return None
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    with open(bundle_path, encoding="utf-8") as f:
        bundle = json.load(f)
    meta = bundle.get("bundle_metadata", {})
    return {
        "dataset": out_dir.name,
        "num_policies": index.get("num_policies", 0),
        "policy_count_bundle": meta.get("policy_count", 0),
        "num_rules": meta.get("rule_count", 0),
        "num_constraints": meta.get("constraint_count", 0),
        "num_paths": meta.get("path_count", 0),
        "domains": index.get("domains", {}),
        "flagged_pct": index.get("flagged_pct", 0),
    }


# ---------------------------------------------------------------------------
# Table generation — now reads from computed_metrics.json
# ---------------------------------------------------------------------------

def _write_tables(rows: list[dict], comparison_rows: list[list[str]], comparison_header: list[str]) -> None:
    """Write per-dataset CSV + ACL comparison table (MD + CSV) into results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_docs = len(rows)
    total_policies = sum(r["num_policies"] for r in rows)
    total_rules = sum(r["num_rules"] for r in rows)
    total_constraints = sum(r["num_constraints"] for r in rows)
    total_paths = sum(r["num_paths"] for r in rows)
    all_domains = set()
    for r in rows:
        all_domains.update((r.get("domains") or {}).keys())
    compile_rate = (total_rules / total_policies * 100) if total_policies else 0
    avg_constraints = total_constraints / n_docs if n_docs else 0

    # ── 1) per_dataset_results.csv ──────────────────────────────────────
    csv_path = RESULTS_DIR / "per_dataset_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "num_policies", "num_rules",
                        "num_constraints", "num_paths", "flagged_pct", "domains"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["domains"] = json.dumps(row.get("domains") or {})
            w.writerow(row)
    print(f"  -> {csv_path}")

    # ── 2) ACL_metrics_comparison.csv ───────────────────────────────────
    acl_csv = RESULTS_DIR / "ACL_metrics_comparison.csv"
    with open(acl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(comparison_header)
        w.writerows(comparison_rows)
    print(f"  -> {acl_csv}")

    # ── 3) ACL_metrics_comparison.md (main deliverable) ────────────────
    md = []
    md.append("# PolicyLLM — Evaluation Results")
    md.append("")
    md.append("## 1  Per-Dataset Extraction & Validation")
    md.append("")
    md.append(f"Evaluated on **{n_docs}** real-world policy documents spanning "
              f"**{len(all_domains)}** domains ({', '.join(sorted(all_domains))}).")
    md.append("")
    md.append("| Dataset | Policies | Rules | Constraints | Paths | Flagged % | Domains |")
    md.append("|---------|----------|-------|-------------|-------|-----------|---------|")
    for r in rows:
        doms = r.get("domains") or {}
        dom_str = ", ".join(f"{k} ({v})" for k, v in sorted(doms.items()))
        md.append(
            f"| {r['dataset']} | {r['num_policies']} | {r['num_rules']} "
            f"| {r['num_constraints']} | {r['num_paths']} "
            f"| {r['flagged_pct']:.0f} | {dom_str} |"
        )
    md.append(f"| **Total** | **{total_policies}** | **{total_rules}** "
              f"| **{total_constraints}** | **{total_paths}** | — | {len(all_domains)} domains |")
    md.append("")
    avg_flagged = sum(r.get("flagged_pct", 0) for r in rows) / n_docs if n_docs else 0
    md.append(f"- **Rule expansion rate** (compiled rules / extracted policies): {compile_rate:.1f}% "
              f"(>100 % means some policies produce multiple rules)")
    md.append(f"- **Avg constraints per document**: {avg_constraints:.1f}")
    md.append(f"- **Avg flagged-as-policy rate**: {avg_flagged:.1f}%")
    md.append("")

    md.append("## 2  Comparison vs. Baselines (ACL)")
    md.append("")
    md.append("All metrics below are **computed from actual pipeline runs** on the same document set. "
              "No numbers are hardcoded. Baselines are implemented as ablations and alternatives "
              "of the PolicyLLM pipeline (see `eval/baselines.py`). Each method was evaluated "
              "against expert-annotated ground truth (`eval/reference_data/`).")
    md.append("")
    md.append("| " + " | ".join(comparison_header) + " |")
    md.append("|" + "|".join(["---"] * len(comparison_header)) + "|")
    for row in comparison_rows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    md.append("")

    md.append("### Methodology")
    md.append("")
    md.append("- **Policy Recall / Precision**: Measured against expert-annotated reference policies per document. "
              "Matching uses domain + Jaccard similarity (threshold 0.30) on description tokens.")
    md.append("- **Condition F1**: Micro-averaged over condition triples (type, parameter, operator) "
              "compared to reference conditions.")
    md.append("- **Conflict F1**: Set-based F1 over detected conflict pairs vs. reference conflicts.")
    md.append("- **Compliance %**: Fraction of enforcement decisions matching expected action on "
              f"{len(comparison_rows[0]) if comparison_rows else '?'}-scenario test set "
              "(see `eval/reference_data/enforcement_gt.json`).")
    md.append("- **FP %**: False positive rate — escalated when should have passed.")
    md.append("- **Latency**: Average wall-clock seconds per enforcement decision.")
    md.append("")

    md.append("### Reproducing these results")
    md.append("")
    md.append("```bash")
    md.append("# Full evaluation (requires OpenAI API key for LLM-dependent baselines)")
    md.append("python run_extract_tests_pdfs.py --eval-only")
    md.append("")
    md.append("# Non-API baselines only (Keyword, Semantic, SMT-only)")
    md.append("python run_extract_tests_pdfs.py --eval-only --no-api")
    md.append("```")
    md.append("")

    md.append("## 3  Pipeline Summary")
    md.append("")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Documents processed | {n_docs} |")
    md.append(f"| Total policies extracted | {total_policies} |")
    md.append(f"| Total compiled rules | {total_rules} |")
    md.append(f"| Total constraints | {total_constraints} |")
    md.append(f"| Total decision paths | {total_paths} |")
    md.append(f"| Unique domains | {len(all_domains)} ({', '.join(sorted(all_domains))}) |")
    md.append(f"| Compilation rate | {compile_rate:.1f}% |")
    md.append("")
    md.append("---")
    md.append("*Auto-generated by `run_extract_tests_pdfs.py` — all metrics computed from actual runs*")

    acl_md = RESULTS_DIR / "ACL_metrics_comparison.md"
    with open(acl_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"  -> {acl_md}")


# ---------------------------------------------------------------------------
# Pipeline runner (extraction + validation)
# ---------------------------------------------------------------------------

def _run_pipeline(pdfs: list[Path]) -> None:
    """Run extract + validate on each PDF."""
    for pdf in sorted(pdfs):
        name = pdf.stem[:30].replace(" ", "_")
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Extracting: {pdf.name}  ->  {out_dir}/")

        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(ROOT / "main.py"),
             "extract", str(pdf),
             "--out", str(out_dir),
             "--config", str(CONFIG),
             "--batch", "run"],
            cwd=str(ROOT), timeout=900,
        )
        if r.returncode != 0:
            print(f"  !! Extract FAILED for {pdf.name}")
            continue

        jsonl_files = list(out_dir.glob("*.jsonl"))
        jsonl_files = [f for f in jsonl_files if "stage5" not in str(f)]
        if not jsonl_files:
            print(f"  !! No .jsonl produced for {pdf.name}")
            continue

        jsonl_path = jsonl_files[0]
        bundle_path = out_dir / "compiled_policy_bundle.json"
        print(f"  Validating -> {bundle_path}")
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "main.py"),
             "validate", str(jsonl_path),
             "--out", str(bundle_path)],
            cwd=str(ROOT), timeout=180,
        )
        elapsed = time.time() - t0
        if r2.returncode == 0:
            with open(bundle_path, encoding="utf-8") as f:
                meta = json.load(f).get("bundle_metadata", {})
            print(f"  OK  {meta.get('policy_count',0)} rules, "
                  f"{meta.get('constraint_count',0)} constraints, "
                  f"{meta.get('path_count',0)} paths  ({elapsed:.1f}s)")
        else:
            print(f"  !! Validate FAILED for {pdf.name}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Evaluation runner (calls eval/ framework)
# ---------------------------------------------------------------------------

def _run_evaluation(
    no_api: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_provider: str = "chatgpt",
    llm_model: str = "gpt-4o-mini",
    require_api: bool = False,
) -> list[list[str]]:
    """Run the full evaluation framework and return comparison table rows."""
    # Import eval framework
    sys.path.insert(0, str(ROOT))
    from eval.runner import run_full_evaluation
    from eval.baselines import infer_embedding_backend_type

    llm_client = None
    if not no_api:
        try:
            from Extractor.src.llm.client import LLMClient
            llm_client = LLMClient(
                provider=llm_provider,
                model_id=llm_model,
                temperature=0.0,
                max_tokens=2048,
            )
            logger.info("LLM client initialized for evaluation (%s / %s)", llm_provider, llm_model)
        except Exception as e:
            if require_api:
                raise RuntimeError(
                    f"Failed to initialize API LLM client ({llm_provider}/{llm_model}): {e}"
                ) from e
            logger.warning("LLM client unavailable (%s) — running non-API baselines only", e)

    rows = run_full_evaluation(
        results_dir=RESULTS_DIR,
        tests_dir=TESTS_DIR,
        config_path=CONFIG,
        use_api=not no_api,
        llm_client=llm_client,
        embedding_model=embedding_model,
    )

    # Save computed metrics
    header = ["Method", "Policy Recall", "Policy Precision", "Condition F1",
              "Conflict F1", "Compliance %", "FP %", "Latency (s)"]
    output = {
        "header": header,
        "rows": rows,
        "computed": True,
        "run_metadata": {
            "embedding_model_requested": embedding_model,
            "embedding_backend_type": infer_embedding_backend_type(embedding_model),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "api_baselines_enabled": bool(llm_client) and (not no_api),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    metrics_path = RESULTS_DIR / "computed_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Computed metrics saved to {metrics_path}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    ap = argparse.ArgumentParser(
        description="Run PolicyLLM on tests/*.pdf, evaluate baselines, and write ACL comparison tables to results/"
    )
    ap.add_argument("--tables-only", action="store_true",
                    help="Skip extraction & evaluation; build tables from existing computed_metrics.json")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip extraction; run evaluation on existing results")
    ap.add_argument("--no-api", action="store_true",
                    help="Skip LLM-dependent baselines (Keyword, Semantic, SMT-only only)")
    ap.add_argument("--llm-provider", default="chatgpt",
                    help="LLM provider for API baselines (chatgpt|stub|ollama|bedrock_claude|anthropic)")
    ap.add_argument("--llm-model", default="gpt-4o-mini",
                    help="LLM model id for API baselines")
    ap.add_argument("--require-api", action="store_true",
                    help="Fail instead of silently falling back to non-API baselines when API client cannot initialize")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                    help="Embedding model used by RAG baselines during evaluation")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Extract + Validate (unless --tables-only or --eval-only) ──────
    if not args.tables_only and not args.eval_only:
        pdfs = sorted(TESTS_DIR.glob("*.pdf")) if TESTS_DIR.exists() else []
        if not pdfs:
            print("No PDFs found in tests/")
            return 1
        print(f"Found {len(pdfs)} PDFs in tests/")
        _run_pipeline(pdfs)

    # ── Run evaluation (unless --tables-only) ─────────────────────────
    comparison_rows: list[list[str]] = []
    comparison_header = ["Method", "Policy Recall", "Policy Precision", "Condition F1",
                         "Conflict F1", "Compliance %", "FP %", "Latency (s)"]

    if not args.tables_only:
        print("\n" + "=" * 60)
        print("Running evaluation framework...")
        comparison_rows = _run_evaluation(
            no_api=args.no_api,
            embedding_model=args.embedding_model,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            require_api=args.require_api,
        )
    else:
        # Load from cached computed_metrics.json
        metrics_path = RESULTS_DIR / "computed_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
            comparison_rows = data.get("rows", [])
            comparison_header = data.get("header", comparison_header)
            print(f"Loaded computed metrics from {metrics_path}")
        else:
            print(f"No computed_metrics.json found at {metrics_path}.")
            print("Run without --tables-only first to compute metrics.")
            return 1

    # ── Collect per-dataset metrics ───────────────────────────────────
    metrics: list[dict] = []
    for sub in sorted(RESULTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        m = _read_metrics(sub)
        if m:
            metrics.append(m)

    if not metrics:
        print("\nNo result directories with index + bundle found.")
        return 1

    # ── Write tables ──────────────────────────────────────────────────
    print(f"\nBuilding tables from {len(metrics)} dataset(s) ...")
    _write_tables(metrics, comparison_rows, comparison_header)

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"{'Method':<30} {'Recall':>8} {'Prec':>8} {'CondF1':>8} {'ConflF1':>8} {'Comp%':>8} {'FP%':>8} {'Lat(s)':>8}")
    print(f"{'-'*110}")
    for row in comparison_rows:
        print(f"{row[0]:<30} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8} {row[6]:>8} {row[7]:>8}")
    print(f"{'='*110}")

    print(f"\nDone. Tables in {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
