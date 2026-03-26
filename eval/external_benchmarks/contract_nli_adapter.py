"""ContractNLI pilot adapter for PolicyLLM external generalization checks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.external_benchmarks.utils import (
    build_or_load_split,
    calibrate_ternary_thresholds,
    classification_metrics,
    ensure_dir,
    make_llm_client,
    map_ternary_label,
    now_utc_iso,
    run_policyllm_enforcement,
    write_json,
)

LABEL_MAP_DEFAULT = {
    0: "entailment",
    1: "contradiction",
    2: "not_mentioned",
}


def _pick_split(ds_dict: Any):
    for name in ("test", "validation", "train"):
        if name in ds_dict:
            return name, ds_dict[name]
    # If already Dataset, treat as single split.
    return "train", ds_dict


def _normalize_label(raw: Any, features: Any = None) -> str:
    if isinstance(raw, str):
        l = raw.strip().lower().replace("neutral", "not_mentioned")
        if l in {"entailment", "contradiction", "not_mentioned"}:
            return l
    if isinstance(raw, int):
        if features is not None:
            try:
                names = getattr(features["label"], "names", None)
                if names and 0 <= raw < len(names):
                    return _normalize_label(names[raw])
            except Exception:
                pass
        return LABEL_MAP_DEFAULT.get(raw, "not_mentioned")
    return "not_mentioned"


def _extract_contract_text(row: Dict[str, Any]) -> str:
    for key in ("premise", "contract", "document", "context", "text"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    # Some variants store nested structures.
    for key in ("input", "instance", "data"):
        val = row.get(key)
        if isinstance(val, dict):
            for sub in ("premise", "contract", "document", "context", "text"):
                if isinstance(val.get(sub), str) and val[sub].strip():
                    return val[sub].strip()
    return ""


def _extract_hypothesis(row: Dict[str, Any]) -> str:
    for key in ("hypothesis", "claim", "statement", "question"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    for key in ("input", "instance", "data"):
        val = row.get(key)
        if isinstance(val, dict):
            for sub in ("hypothesis", "claim", "statement", "question"):
                if isinstance(val.get(sub), str) and val[sub].strip():
                    return val[sub].strip()
    return ""


def _parse_label_from_llm(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("label", "prediction", "answer"):
            if key in obj:
                return _parse_label_from_llm(obj[key])
    if isinstance(obj, str):
        v = obj.strip().lower().replace("neutral", "not_mentioned")
        if "entail" in v:
            return "entailment"
        if "contrad" in v:
            return "contradiction"
        if "not" in v or "mention" in v:
            return "not_mentioned"
    return "not_mentioned"


def _run_vanilla_baseline(llm_client: Any, contract: str, hypothesis: str) -> str:
    contract_text = contract
    prompt = (
        "Given the contract and hypothesis, classify as entailment, contradiction, or not_mentioned.\n"
        f"Contract:\n{contract_text}\n\n"
        f"Hypothesis: {hypothesis}\n"
        'Return JSON: {"label": "entailment|contradiction|not_mentioned"}'
    )
    result = llm_client.invoke_json(prompt)
    return _parse_label_from_llm(result)


def _run_system_prompt_baseline(llm_client: Any, contract: str, hypothesis: str) -> str:
    contract_text = contract
    prompt = (
        "You are a strict contract entailment checker. Use only the contract text and avoid guessing.\n"
        f"CONTRACT:\n{contract_text}\n\n"
        f"HYPOTHESIS: {hypothesis}\n"
        'Return JSON: {"label": "entailment|contradiction|not_mentioned"}'
    )
    result = llm_client.invoke_json(prompt)
    return _parse_label_from_llm(result)


def _maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text[:max_chars]


def _load_contractnli_dataset(local_path: Path, dataset_config: str) -> Tuple[str, Any]:
    if local_path.exists():
        ds_dict = load_from_disk(str(local_path))
    else:
        ds_dict = load_dataset("kiddothe2b/contract-nli", dataset_config)
        ensure_dir(local_path.parent)
        ds_dict.save_to_disk(str(local_path))
    return _pick_split(ds_dict)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    llm_client = make_llm_client(model_id=args.model, provider=args.llm_provider)

    split_name, ds = _load_contractnli_dataset(Path(args.dataset_path), args.dataset_config)
    features = getattr(ds, "features", None)

    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        contract = _extract_contract_text(row)
        hypothesis = _extract_hypothesis(row)
        if not contract or not hypothesis:
            continue
        label_raw = row.get("label", row.get("gold_label", row.get("answer")))
        label = _normalize_label(label_raw, features)
        if label not in {"entailment", "contradiction", "not_mentioned"}:
            continue
        example_id = str(row.get("id", row.get("uid", f"{split_name}-{idx}")))
        examples.append(
            {
                "id": example_id,
                "contract": contract,
                "hypothesis": hypothesis,
                "label": label,
            }
        )

    split = build_or_load_split(
        split_path=Path(args.split_ids_path),
        examples=examples,
        id_fn=lambda x: x["id"],
        label_fn=lambda x: x["label"],
        dev_n=args.dev_examples,
        test_n=args.max_examples,
        seed=args.seed,
        split_tag=f"contract_nli_v2_{args.dataset_config}_{args.response_mode}_baseline_max_{args.baseline_max_chars}",
    )

    cache_root = Path(args.cache_dir) / "contract_nli"
    config_path = Path(args.config)

    dev_rows: List[Dict[str, Any]] = []
    for ex in split["dev"]:
        bundle_path, _ = build_or_load_bundle_for_contract(
            text=ex["contract"],
            cache_root=cache_root,
            config_path=config_path,
            tenant_id=args.tenant,
            batch_prefix="contractnli-dev",
        )
        result = run_policyllm_enforcement(
            bundle_path=bundle_path,
            query=f"Does the contract state: {ex['hypothesis']}?",
            asserted_response=f"The contract states that {ex['hypothesis']}",
            llm_client=llm_client,
            response_mode=args.response_mode,
        )
        dev_rows.append({
            "id": ex["id"],
            "label": ex["label"],
            "score": result["score"],
            "has_violation": result["has_violation"],
        })

    thresholds = calibrate_ternary_thresholds(
        dev_rows,
        label_key="label",
        positive_label="entailment",
        negative_label="contradiction",
        neutral_label="not_mentioned",
        default_pos_threshold=args.default_pos_threshold,
        default_neg_threshold=args.default_neg_threshold,
    )

    test_predictions: List[Dict[str, Any]] = []
    y_true: List[str] = []
    y_policyllm: List[str] = []
    y_vanilla: List[str] = []
    y_system: List[str] = []

    for ex in split["test"]:
        bundle_path, cache_info = build_or_load_bundle_for_contract(
            text=ex["contract"],
            cache_root=cache_root,
            config_path=config_path,
            tenant_id=args.tenant,
            batch_prefix="contractnli-test",
        )
        policyllm_result = run_policyllm_enforcement(
            bundle_path=bundle_path,
            query=f"Does the contract state: {ex['hypothesis']}?",
            asserted_response=f"The contract states that {ex['hypothesis']}",
            llm_client=llm_client,
            response_mode=args.response_mode,
        )
        pred_policyllm = map_ternary_label(
            score=policyllm_result["score"],
            has_violation=policyllm_result["has_violation"],
            pos_threshold=thresholds["positive_threshold"],
            neg_threshold=thresholds["negative_threshold"],
            positive_label="entailment",
            negative_label="contradiction",
            neutral_label="not_mentioned",
        )

        contract_for_baseline = _maybe_truncate(ex["contract"], args.baseline_max_chars)
        pred_vanilla = _run_vanilla_baseline(llm_client, contract_for_baseline, ex["hypothesis"])
        pred_system = _run_system_prompt_baseline(llm_client, contract_for_baseline, ex["hypothesis"])

        y_true.append(ex["label"])
        y_policyllm.append(pred_policyllm)
        y_vanilla.append(pred_vanilla)
        y_system.append(pred_system)

        test_predictions.append(
            {
                "id": ex["id"],
                "gold_label": ex["label"],
                "score": policyllm_result["score"],
                "has_violation": policyllm_result["has_violation"],
                "pred_policyllm": pred_policyllm,
                "pred_vanilla": pred_vanilla,
                "pred_system_prompt": pred_system,
                "cache_hit": cache_info.get("cache_hit", False),
            }
        )

    labels = ["entailment", "contradiction", "not_mentioned"]
    metrics_policyllm = classification_metrics(y_true, y_policyllm, labels)
    metrics_vanilla = classification_metrics(y_true, y_vanilla, labels)
    metrics_system = classification_metrics(y_true, y_system, labels)

    output = {
        "benchmark": "ContractNLI",
        "dataset_source": str(args.dataset_path),
        "dataset_config": args.dataset_config,
        "split": split_name,
        "seed": args.seed,
        "dev_examples": args.dev_examples,
        "test_examples": args.max_examples,
        "sample_ids": {
            "dev": split["dev_ids"],
            "test": split["test_ids"],
        },
        "thresholds": thresholds,
        "model_settings": {
            "llm_provider": args.llm_provider,
            "backbone_model": args.model,
            "policyllm_generation": args.response_mode,
            "evaluation_protocol": (
                "bundle_consistency_probe"
                if args.response_mode == "asserted_probe"
                else "generation_plus_enforcement"
            ),
            "judge_model": args.model,
            "baseline_max_chars": args.baseline_max_chars,
        },
        "metrics": {
            "policyllm": metrics_policyllm,
            "vanilla": metrics_vanilla,
            "system_prompt": metrics_system,
        },
        "predictions": test_predictions,
        "runtime": {
            "seconds": time.time() - start,
            "completed_at_utc": now_utc_iso(),
        },
        "summary_row": {
            "benchmark": "ContractNLI",
            "method": "PolicyLLM",
            "accuracy": metrics_policyllm["accuracy"],
            "macro_f1": metrics_policyllm["macro_f1"],
        },
    }
    write_json(Path(args.output), output)
    return output


def build_or_load_bundle_for_contract(
    text: str,
    cache_root: Path,
    config_path: Path,
    tenant_id: str,
    batch_prefix: str,
):
    return build_or_load_bundle_for_contract_impl(
        text=text,
        cache_root=cache_root,
        config_path=config_path,
        tenant_id=tenant_id,
        batch_prefix=batch_prefix,
    )


def build_or_load_bundle_for_contract_impl(
    text: str,
    cache_root: Path,
    config_path: Path,
    tenant_id: str,
    batch_prefix: str,
):
    from eval.external_benchmarks.utils import get_or_create_bundle_for_text

    return get_or_create_bundle_for_text(
        text=text,
        cache_root=cache_root,
        config_path=config_path,
        tenant_id=tenant_id,
        batch_prefix=batch_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ContractNLI external benchmark adapter")
    parser.add_argument("--dataset-path", default=str(ROOT / "eval" / "external_benchmarks" / "contract_nli"))
    parser.add_argument("--dataset-config", default="contractnli_a")
    parser.add_argument("--split-ids-path", default=str(ROOT / "results" / "external" / "contract_nli_split_ids.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / "results" / "external" / "cache"))
    parser.add_argument("--config", default=str(ROOT / "Extractor" / "configs" / "config.chatgpt.yaml"))
    parser.add_argument("--tenant", default="external_contract_nli")
    parser.add_argument("--llm-provider", default="chatgpt")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--dev-examples", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-pos-threshold", type=float, default=0.85)
    parser.add_argument("--default-neg-threshold", type=float, default=0.70)
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
        help="Max chars for baseline contract text (0 means full text, no truncation).",
    )
    parser.add_argument("--output", default=str(ROOT / "results" / "external" / "contract_nli_results.json"))
    args = parser.parse_args()

    if args.dataset_path.endswith("contract_nli"):
        args.dataset_path = str(Path(args.dataset_path) / args.dataset_config)

    result = run(args)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
