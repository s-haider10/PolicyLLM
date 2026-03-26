"""Run all external benchmark adapters and aggregate summaries."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _run_step(name: str, cmd: List[str], cwd: Path) -> Dict[str, Any]:
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return {"name": name, "status": "ok", "command": cmd}
    except subprocess.CalledProcessError as e:
        return {"name": name, "status": "failed", "command": cmd, "return_code": e.returncode}


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_mode = getattr(args, "response_mode", "asserted_probe")
    baseline_max_chars = int(getattr(args, "baseline_max_chars", 0))
    unfair_tos_bundle_mode = getattr(args, "unfair_tos_bundle_mode", "extracted_reference")
    cuad_selection_strategy = getattr(args, "cuad_selection_strategy", "random")
    max_policyllm_chars = int(getattr(args, "max_policyllm_chars", 30000))
    cuad_chunk_overlap_chars = int(getattr(args, "cuad_chunk_overlap_chars", 1000))

    contract_path = output_dir / "contract_nli_results.json"
    lb_unfair_path = output_dir / "legalbench_unfair_tos_results.json"
    lb_privacy_path = output_dir / "legalbench_privacy_policy_entailment_results.json"
    cuad_path = output_dir / "cuad_results.json"
    cache_dir = output_dir / "cache"
    contract_split_path = output_dir / "contract_nli_split_ids.json"
    lb_unfair_split_path = output_dir / "legalbench_unfair_tos_split_ids.json"
    lb_privacy_split_path = output_dir / "legalbench_privacy_policy_entailment_split_ids.json"

    steps = [
        (
            "contract_nli",
            [
                sys.executable,
                "eval/external_benchmarks/contract_nli_adapter.py",
                "--seed",
                str(args.seed),
                "--max-examples",
                str(args.max_examples),
                "--dev-examples",
                str(args.dev_examples),
                "--split-ids-path",
                str(contract_split_path),
                "--cache-dir",
                str(cache_dir),
                "--llm-provider",
                args.llm_provider,
                "--model",
                args.model,
                "--config",
                args.config,
                "--response-mode",
                response_mode,
                "--baseline-max-chars",
                str(baseline_max_chars),
                "--output",
                str(contract_path),
            ],
        ),
        (
            "legalbench_unfair_tos",
            [
                sys.executable,
                "eval/external_benchmarks/legalbench_adapter.py",
                "--task",
                "unfair_tos",
                "--seed",
                str(args.seed),
                "--max-examples",
                str(args.max_examples),
                "--dev-examples",
                str(args.dev_examples),
                "--split-ids-path",
                str(lb_unfair_split_path),
                "--cache-dir",
                str(cache_dir),
                "--llm-provider",
                args.llm_provider,
                "--model",
                args.model,
                "--config",
                args.config,
                "--response-mode",
                response_mode,
                "--baseline-max-chars",
                str(baseline_max_chars),
                "--unfair-tos-bundle-mode",
                unfair_tos_bundle_mode,
                "--output",
                str(lb_unfair_path),
            ],
        ),
        (
            "legalbench_privacy_policy_entailment",
            [
                sys.executable,
                "eval/external_benchmarks/legalbench_adapter.py",
                "--task",
                "privacy_policy_entailment",
                "--seed",
                str(args.seed),
                "--max-examples",
                str(args.max_examples),
                "--dev-examples",
                str(args.dev_examples),
                "--split-ids-path",
                str(lb_privacy_split_path),
                "--cache-dir",
                str(cache_dir),
                "--llm-provider",
                args.llm_provider,
                "--model",
                args.model,
                "--config",
                args.config,
                "--response-mode",
                response_mode,
                "--baseline-max-chars",
                str(baseline_max_chars),
                "--output",
                str(lb_privacy_path),
            ],
        ),
        (
            "cuad",
            [
                sys.executable,
                "eval/external_benchmarks/cuad_adapter.py",
                "--seed",
                str(args.seed),
                "--num-contracts",
                str(args.num_contracts),
                "--cache-dir",
                str(cache_dir),
                "--llm-provider",
                args.llm_provider,
                "--model",
                args.model,
                "--config",
                args.config,
                "--embedding-model",
                args.embedding_model,
                "--selection-strategy",
                cuad_selection_strategy,
                "--max-policyllm-chars",
                str(max_policyllm_chars),
                "--chunk-overlap-chars",
                str(cuad_chunk_overlap_chars),
                "--output",
                str(cuad_path),
            ],
        ),
    ]

    if getattr(args, "aggregate_only", False):
        statuses = []
        for name, _cmd in steps:
            expected = {
                "contract_nli": contract_path,
                "legalbench_unfair_tos": lb_unfair_path,
                "legalbench_privacy_policy_entailment": lb_privacy_path,
                "cuad": cuad_path,
            }[name]
            statuses.append(
                {
                    "name": name,
                    "status": "ok" if expected.exists() else "missing",
                    "command": _cmd,
                }
            )
    else:
        statuses = [_run_step(name, cmd, ROOT) for name, cmd in steps]

    payloads = {
        "contract_nli": _load_json_if_exists(contract_path),
        "legalbench_unfair_tos": _load_json_if_exists(lb_unfair_path),
        "legalbench_privacy_policy_entailment": _load_json_if_exists(lb_privacy_path),
        "cuad": _load_json_if_exists(cuad_path),
    }

    summary_rows: List[Dict[str, Any]] = []

    c = payloads["contract_nli"]
    if c:
        for method_key, method_name in [
            ("policyllm", "PolicyLLM"),
            ("vanilla", "Vanilla LLM"),
            ("system_prompt", "System Prompt"),
        ]:
            m = c.get("metrics", {}).get(method_key)
            if not m:
                continue
            summary_rows.append(
                {
                    "benchmark": "ContractNLI",
                    "method": method_name,
                    "metric_a": m.get("accuracy", 0.0),
                    "metric_b": m.get("macro_f1", 0.0),
                    "notes": "accuracy/macro_f1",
                }
            )

    for key, name in [
        ("legalbench_unfair_tos", "LegalBench::unfair_tos"),
        ("legalbench_privacy_policy_entailment", "LegalBench::privacy_policy_entailment"),
    ]:
        p = payloads[key]
        if p:
            for method_key, method_name in [
                ("policyllm", "PolicyLLM"),
                ("vanilla", "Vanilla LLM"),
            ]:
                m = p.get("metrics", {}).get(method_key)
                if not m:
                    continue
                summary_rows.append(
                    {
                        "benchmark": name,
                        "method": method_name,
                        "metric_a": m.get("accuracy", 0.0),
                        "metric_b": m.get("macro_f1", 0.0),
                        "notes": "accuracy/macro_f1",
                    }
                )

    q = payloads["cuad"]
    if q:
        for method_key, method_name in [
            ("policyllm", "PolicyLLM"),
            ("keyword_rules", "Keyword + Rules"),
            ("rag", "RAG Extractor"),
        ]:
            m = q.get("metrics", {}).get(method_key)
            if not m:
                continue
            summary_rows.append(
                {
                    "benchmark": "CUAD",
                    "method": method_name,
                    "metric_a": m.get("clause_type_precision", 0.0),
                    "metric_b": m.get("clause_type_recall", 0.0),
                    "notes": "precision/recall",
                }
            )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "statuses": statuses,
        "rows": summary_rows,
        "artifacts": {
            "contract_nli": str(contract_path),
            "legalbench_unfair_tos": str(lb_unfair_path),
            "legalbench_privacy_policy_entailment": str(lb_privacy_path),
            "cuad": str(cuad_path),
        },
    }

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "method", "metric_a", "metric_b", "notes"])
        for row in summary_rows:
            writer.writerow([row["benchmark"], row["method"], row["metric_a"], row["metric_b"], row["notes"]])

    summary_md = output_dir / "summary.md"
    lines = [
        "# External Benchmark Summary",
        "",
        "| Benchmark | Method | Metric A | Metric B | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {row['metric_a']:.4f} | {row['metric_b']:.4f} | {row['notes']} |"
        )

    lines.append("")
    lines.append("## Execution Status")
    lines.append("")
    for status in statuses:
        lines.append(f"- `{status['name']}`: **{status['status']}**")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all external benchmark adapters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-examples", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--num-contracts", type=int, default=10)
    parser.add_argument("--llm-provider", default="chatgpt")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--config", default=str(ROOT / "Extractor" / "configs" / "config.chatgpt.yaml"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--response-mode", default="asserted_probe", choices=["asserted_probe", "generated_response"])
    parser.add_argument("--baseline-max-chars", type=int, default=0)
    parser.add_argument("--unfair-tos-bundle-mode", default="extracted_reference", choices=["extracted_reference", "legacy_static"])
    parser.add_argument("--cuad-selection-strategy", default="random", choices=["random", "max_coverage"])
    parser.add_argument("--max-policyllm-chars", type=int, default=30000)
    parser.add_argument("--cuad-chunk-overlap-chars", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "external"))
    parser.add_argument("--aggregate-only", action="store_true", help="Skip benchmark execution and aggregate existing artifacts only.")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
