"""Run embedding ablation for RAG baselines and generate summary artifacts."""
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


def _run_eval_for_embedding(embedding_model: str, output_path: Path, extra_args: List[str]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "eval.runner",
        "--embedding-model",
        embedding_model,
        "--output",
        str(output_path),
    ] + extra_args
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def _rows_by_method(run_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    return {row[0]: row for row in run_payload.get("rows", [])}


def _require_row(rows: Dict[str, List[str]], method_name: str, embedding_model: str) -> List[str]:
    row = rows.get(method_name)
    if row is None:
        raise RuntimeError(
            f"Missing required row '{method_name}' for embedding model '{embedding_model}'. "
            "This usually means API baselines were not executed (for example missing credentials)."
        )
    return row


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(args.embedding_models)

    per_model_payload: Dict[str, Dict[str, Any]] = {}
    extra_args = []
    if args.no_api:
        extra_args.append("--no-api")
    else:
        extra_args.append("--require-api")
    extra_args.extend(["--llm-provider", args.llm_provider, "--llm-model", args.llm_model])

    for model in models:
        safe_name = model.replace("/", "_").replace("-", "_")
        out_path = output_dir / f"{safe_name}.json"
        per_model_payload[model] = _run_eval_for_embedding(model, out_path, extra_args)

    summary_rows: List[Dict[str, Any]] = []

    # Keep PolicyLLM reference row from first run only.
    first_rows = _rows_by_method(next(iter(per_model_payload.values())))
    policyllm = _require_row(first_rows, "PolicyLLM (Ours)", next(iter(per_model_payload.keys())))

    for model, payload in per_model_payload.items():
        rows = _rows_by_method(payload)
        for method in ("RAG retrieval only", "RAG + Z3 hybrid"):
            row = _require_row(rows, method, model)
            summary_rows.append(
                {
                    "method": method,
                    "embedding_model": model,
                    "policy_recall": row[1],
                    "policy_precision": row[2],
                    "condition_f1": row[3],
                    "compliance_pct": row[5],
                }
            )

    summary_rows.append(
        {
            "method": "PolicyLLM (Ours)",
            "embedding_model": "N/A",
            "policy_recall": policyllm[1],
            "policy_precision": policyllm[2],
            "condition_f1": policyllm[3],
            "compliance_pct": policyllm[5],
        }
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "rows": summary_rows,
    }

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "embedding_model", "policy_recall", "policy_precision", "condition_f1", "compliance_pct"])
        for r in summary_rows:
            writer.writerow([r["method"], r["embedding_model"], r["policy_recall"], r["policy_precision"], r["condition_f1"], r["compliance_pct"]])

    summary_md = output_dir / "summary.md"
    lines = [
        "# Embedding Ablation Summary",
        "",
        "| Method | Embedding | P-Rec | P-Prec | C-F1 | Comp. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['method']} | {r['embedding_model']} | {r['policy_recall']} | {r['policy_precision']} | {r['condition_f1']} | {r['compliance_pct']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding ablation and summarize RAG rows")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "embedding_ablation"))
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=["all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5", "text-embedding-3-small"],
        help="Embedding models to evaluate in ablation order",
    )
    parser.add_argument("--llm-provider", default="chatgpt")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
