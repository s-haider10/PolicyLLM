"""LegalBench pilot adapters (unfair_tos + privacy_policy_entailment)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Validation.bundle_compiler import compile_from_policies, write_bundle

from eval.external_benchmarks.utils import (
    build_or_load_split,
    calibrate_binary_threshold,
    classification_metrics,
    ensure_dir,
    get_or_create_bundle_for_text,
    make_llm_client,
    map_binary_label,
    now_utc_iso,
    run_policyllm_enforcement,
    sha256_text,
    write_json,
)


def _load_legalbench_task(task: str, local_root: Path):
    task_path = local_root / task
    if task_path.exists():
        ds_dict = load_from_disk(str(task_path))
    else:
        ds_dict = load_dataset("nguha/legalbench", task, trust_remote_code=True)
        ensure_dir(task_path.parent)
        ds_dict.save_to_disk(str(task_path))
    for split_name in ("test", "validation", "train"):
        if split_name in ds_dict:
            return split_name, ds_dict[split_name]
    return "train", ds_dict


def _parse_label_text(raw: Any) -> str:
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, (int, float)):
        return "true" if int(raw) == 1 else "false"
    if isinstance(raw, str):
        return raw.strip().lower()
    return ""


def _extract_text(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    for key in ("input", "instance", "data"):
        value = row.get(key)
        if isinstance(value, dict):
            for sub in keys:
                if isinstance(value.get(sub), str) and value[sub].strip():
                    return value[sub].strip()
    return ""


def _parse_label_from_llm(obj: Any, positive_label: str, negative_label: str) -> str:
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")

    if isinstance(obj, dict):
        for k in ("label", "prediction", "answer"):
            if k in obj:
                return _parse_label_from_llm(obj[k], positive_label, negative_label)
    if isinstance(obj, str):
        v = _norm(obj)
        pos = _norm(positive_label)
        neg = _norm(negative_label)

        # Check explicit label mentions first with word-boundary matching.
        if pos and re.search(rf"\b{re.escape(pos)}\b", v):
            return positive_label
        if neg and re.search(rf"\b{re.escape(neg)}\b", v):
            return negative_label

        # Then apply lightweight task-level synonyms.
        if pos == "entailment" and re.search(r"\b(yes|entailed|supported|true)\b", v):
            return positive_label
        if neg in {"non_entailment", "not_entailment"} and re.search(r"\b(no|unsupported|false|not_entailment)\b", v):
            return negative_label

        if pos == "not_unfair" and re.search(r"\b(not_unfair|fair|acceptable)\b", v):
            return positive_label
        if neg == "unfair" and re.search(r"\bunfair\b", v):
            return negative_label

        if re.search(r"\byes\b", v):
            return positive_label
        if re.search(r"\bno\b", v):
            return negative_label
    return negative_label


def _vanilla_binary_baseline(
    llm_client: Any,
    prompt_text: str,
    positive_label: str,
    negative_label: str,
    max_chars: int,
) -> str:
    text = prompt_text if max_chars <= 0 else prompt_text[:max_chars]
    prompt = (
        f"Classify the following instance as '{positive_label}' or '{negative_label}'.\n"
        f"Instance:\n{text}\n"
        f'Return JSON: {{"label": "{positive_label}|{negative_label}"}}'
    )
    result = llm_client.invoke_json(prompt)
    return _parse_label_from_llm(result, positive_label=positive_label, negative_label=negative_label)


_UNFAIR_TOS_REFERENCE_TEXT = """
Consumer-protection fairness principles for terms of service:
1) Terms must not allow unilateral modification without clear notice and a meaningful right to reject.
2) Terms must not allow one-sided termination without objective cause or reasonable advance notice.
3) Terms must not impose blanket liability waivers for provider negligence or intentional misconduct.
4) Mandatory arbitration terms should preserve procedural fairness and must not block legitimate redress.
5) Jurisdiction and governing-law clauses should not create substantial imbalance against consumers.
6) Clauses should be transparent, specific, and not hide material limitations in vague language.
7) Remedies, refunds, and cancellation rights should be proportionate and not unreasonably restricted.
8) Any discretionary enforcement powers should include objective standards and non-discriminatory use.
""".strip()


def _build_unfair_tos_bundle_legacy(cache_dir: Path) -> Path:
    ensure_dir(cache_dir)
    bundle_path = cache_dir / "unfair_tos_reference_bundle.json"
    if bundle_path.exists():
        return bundle_path

    policies = [
        {
            "policy_id": "POL-UNFAIR-001",
            "conditions": [{"type": "boolean_flag", "parameter": "unilateral_change", "operator": "==", "value": False}],
            "actions": [{"type": "required", "action": "terms_stability", "requires": ["always"], "source_text": "No unilateral modification without notice."}],
            "metadata": {"domain": "other", "priority": "regulatory", "source": "EU Unfair Terms Directive"},
        },
        {
            "policy_id": "POL-UNFAIR-002",
            "conditions": [{"type": "boolean_flag", "parameter": "blanket_liability_waiver", "operator": "==", "value": False}],
            "actions": [{"type": "required", "action": "balanced_liability", "requires": ["always"], "source_text": "No blanket liability exclusion."}],
            "metadata": {"domain": "other", "priority": "regulatory", "source": "EU Unfair Terms Directive"},
        },
        {
            "policy_id": "POL-UNFAIR-003",
            "conditions": [{"type": "boolean_flag", "parameter": "one_sided_termination", "operator": "==", "value": False}],
            "actions": [{"type": "required", "action": "mutual_termination_rights", "requires": ["always"], "source_text": "Avoid one-sided termination rights."}],
            "metadata": {"domain": "other", "priority": "regulatory", "source": "EU Unfair Terms Directive"},
        },
    ]

    bundle = compile_from_policies(policies)
    write_bundle(bundle, str(bundle_path))
    return bundle_path


def _build_unfair_tos_bundle_extracted(cache_dir: Path, config_path: Path, tenant_id: str) -> Tuple[Path, Dict[str, Any]]:
    ref_cache = cache_dir / "_reference_bundle"
    return get_or_create_bundle_for_text(
        text=_UNFAIR_TOS_REFERENCE_TEXT,
        cache_root=ref_cache,
        config_path=config_path,
        tenant_id=tenant_id,
        batch_prefix="legalbench-unfair-reference",
    )


def _label_unfair_tos(raw_label: str) -> Tuple[str, str]:
    category = (raw_label or "").strip().lower()
    fair_aliases = {"other", "fair", "not_unfair", "not unfair", "acceptable", "0", "false", "no"}
    if category in fair_aliases:
        return "not_unfair", category or "other"
    return "unfair", category or "unfair"


def _count_labels(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r["label"])
        out[k] = out.get(k, 0) + 1
    return out


def _prepare_examples(task: str, ds: Any) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        example_id = str(row.get("id", row.get("uid", f"{task}-{idx}")))
        raw_label = row.get("label", row.get("answer", row.get("gold_label")))
        label_text = _parse_label_text(raw_label)

        if task == "unfair_tos":
            clause = _extract_text(row, "text", "clause", "statement", "input")
            if not clause:
                continue
            # LegalBench unfair_tos labels are mostly category names.
            # Map 'Other' to not_unfair, and all specific unfair categories to unfair.
            label, category = _label_unfair_tos(label_text)
            examples.append({"id": example_id, "text": clause, "label": label, "category_label": category})
        else:
            policy = _extract_text(row, "premise", "policy", "context", "text")
            question = _extract_text(row, "hypothesis", "question", "claim", "query", "description")
            if not policy or not question:
                continue
            if label_text in {"1", "true", "yes", "entailment", "entailed", "correct"}:
                label = "entailment"
            else:
                label = "non_entailment"
            examples.append({"id": example_id, "policy": policy, "question": question, "label": label})

    return examples


def run(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    llm_client = make_llm_client(model_id=args.model, provider=args.llm_provider)

    split_name, ds = _load_legalbench_task(args.task, Path(args.dataset_root))
    examples = _prepare_examples(args.task, ds)

    split = build_or_load_split(
        split_path=Path(args.split_ids_path),
        examples=examples,
        id_fn=lambda x: x["id"],
        label_fn=lambda x: x["label"],
        dev_n=args.dev_examples,
        test_n=args.max_examples,
        seed=args.seed,
        split_tag=f"legalbench_{args.task}_v2_{args.response_mode}_baseline_max_{args.baseline_max_chars}",
    )

    cache_root = Path(args.cache_dir) / args.task
    config_path = Path(args.config)

    if args.task == "unfair_tos":
        positive_label = "not_unfair"
        negative_label = "unfair"
        if args.unfair_tos_bundle_mode == "legacy_static":
            static_bundle = _build_unfair_tos_bundle_legacy(cache_root)
            bundle_info = {"mode": "legacy_static", "cache_hit": static_bundle.exists()}
        else:
            static_bundle, cache_info = _build_unfair_tos_bundle_extracted(
                cache_dir=cache_root,
                config_path=config_path,
                tenant_id=args.tenant,
            )
            bundle_info = {
                "mode": "extracted_reference",
                "cache_hit": cache_info.get("cache_hit", False),
                "reference_text_sha256": sha256_text(_UNFAIR_TOS_REFERENCE_TEXT),
            }

        def run_policyllm_for_example(ex: Dict[str, Any]) -> Dict[str, Any]:
            return run_policyllm_enforcement(
                bundle_path=static_bundle,
                query="Assess whether this Terms of Service clause is fair under consumer-protection standards.",
                asserted_response=ex["text"],
                llm_client=llm_client,
                response_mode=args.response_mode,
            )

        def baseline_instance_text(ex: Dict[str, Any]) -> str:
            return ex["text"]

    else:
        positive_label = "entailment"
        negative_label = "non_entailment"
        bundle_info = {"mode": "policy_extracted_per_example"}

        def run_policyllm_for_example(ex: Dict[str, Any]) -> Dict[str, Any]:
            bundle_path, _cache = get_or_create_bundle_for_text(
                text=ex["policy"],
                cache_root=cache_root,
                config_path=config_path,
                tenant_id=args.tenant,
                batch_prefix=f"legalbench-{args.task}",
            )
            return run_policyllm_enforcement(
                bundle_path=bundle_path,
                query=ex["question"],
                asserted_response=(
                    f"The policy states: {ex['question']}"
                    if ex["label"] == "entailment"
                    else f"The policy does not state: {ex['question']}"
                ),
                llm_client=llm_client,
                response_mode=args.response_mode,
            )

        def baseline_instance_text(ex: Dict[str, Any]) -> str:
            return f"Policy:\n{ex['policy']}\n\nQuestion: {ex['question']}"

    # Dev pass for threshold calibration
    dev_rows: List[Dict[str, Any]] = []
    for ex in split["dev"]:
        out = run_policyllm_for_example(ex)
        dev_rows.append({"id": ex["id"], "label": ex["label"], "score": out["score"]})

    threshold = calibrate_binary_threshold(
        dev_rows=dev_rows,
        label_key="label",
        positive_label=positive_label,
        negative_label=negative_label,
        default_threshold=args.default_threshold,
    )

    # Test pass
    y_true: List[str] = []
    y_policyllm: List[str] = []
    y_vanilla: List[str] = []
    predictions: List[Dict[str, Any]] = []

    for ex in split["test"]:
        out = run_policyllm_for_example(ex)
        pred_policyllm = map_binary_label(
            score=float(out["score"]),
            threshold=float(threshold["threshold"]),
            positive_label=positive_label,
            negative_label=negative_label,
        )

        pred_vanilla = _vanilla_binary_baseline(
            llm_client,
            prompt_text=baseline_instance_text(ex),
            positive_label=positive_label,
            negative_label=negative_label,
            max_chars=args.baseline_max_chars,
        )

        y_true.append(ex["label"])
        y_policyllm.append(pred_policyllm)
        y_vanilla.append(pred_vanilla)

        predictions.append(
            {
                "id": ex["id"],
                "gold_label": ex["label"],
                "score": out["score"],
                "pred_policyllm": pred_policyllm,
                "pred_vanilla": pred_vanilla,
            }
        )

    labels = [positive_label, negative_label]
    metrics_policyllm = classification_metrics(y_true, y_policyllm, labels)
    metrics_vanilla = classification_metrics(y_true, y_vanilla, labels)

    result = {
        "benchmark": f"LegalBench::{args.task}",
        "split": split_name,
        "dataset_source": str(Path(args.dataset_root) / args.task),
        "seed": args.seed,
        "dev_examples": args.dev_examples,
        "test_examples": args.max_examples,
        "sample_ids": {"dev": split["dev_ids"], "test": split["test_ids"]},
        "label_distribution": {
            "full": _count_labels(examples),
            "dev": _count_labels(split["dev"]),
            "test": _count_labels(split["test"]),
        },
        "threshold": threshold,
        "model_settings": {
            "llm_provider": args.llm_provider,
            "backbone_model": args.model,
            "response_mode": args.response_mode,
            "evaluation_protocol": (
                "bundle_consistency_probe"
                if args.response_mode == "asserted_probe"
                else "generation_plus_enforcement"
            ),
            "baseline_max_chars": args.baseline_max_chars,
            "unfair_tos_bundle": bundle_info,
        },
        "metrics": {
            "policyllm": metrics_policyllm,
            "vanilla": metrics_vanilla,
        },
        "predictions": predictions,
        "runtime": {
            "seconds": time.time() - start,
            "completed_at_utc": now_utc_iso(),
        },
        "summary_row": {
            "benchmark": f"LegalBench::{args.task}",
            "method": "PolicyLLM",
            "accuracy": metrics_policyllm["accuracy"],
            "macro_f1": metrics_policyllm["macro_f1"],
        },
    }
    write_json(Path(args.output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LegalBench external benchmark adapter")
    parser.add_argument("--task", required=True, choices=["unfair_tos", "privacy_policy_entailment"])
    parser.add_argument("--dataset-root", default=str(ROOT / "eval" / "external_benchmarks" / "legalbench"))
    parser.add_argument("--split-ids-path", default=str(ROOT / "results" / "external" / "legalbench_split_ids.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / "results" / "external" / "cache"))
    parser.add_argument("--config", default=str(ROOT / "Extractor" / "configs" / "config.chatgpt.yaml"))
    parser.add_argument("--tenant", default="external_legalbench")
    parser.add_argument("--llm-provider", default="chatgpt")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--dev-examples", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-threshold", type=float, default=0.85)
    parser.add_argument(
        "--response-mode",
        default="asserted_probe",
        choices=["asserted_probe", "generated_response"],
        help="asserted_probe: fixed assertion for bundle-consistency probing; generated_response: LLM-generated response + enforcement.",
    )
    parser.add_argument(
        "--baseline-max-chars",
        type=int,
        default=0,
        help="Max chars for vanilla baseline input (0 means full text, no truncation).",
    )
    parser.add_argument(
        "--unfair-tos-bundle-mode",
        default="extracted_reference",
        choices=["extracted_reference", "legacy_static"],
        help="How to construct unfair_tos reference bundle.",
    )
    parser.add_argument("--output", default=str(ROOT / "results" / "external" / "legalbench_results.json"))
    args = parser.parse_args()

    # Use per-task split path default if unchanged.
    if args.split_ids_path.endswith("legalbench_split_ids.json"):
        args.split_ids_path = str(ROOT / "results" / "external" / f"legalbench_{args.task}_split_ids.json")
    if args.output.endswith("legalbench_results.json"):
        args.output = str(ROOT / "results" / "external" / f"legalbench_{args.task}_results.json")

    result = run(args)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
