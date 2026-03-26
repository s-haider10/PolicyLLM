"""CUAD extraction-only pilot adapter for PolicyLLM."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Extractor.src.config import load_config
from Extractor.src.regularize import router

from eval.baselines import KeywordRulesExtractor, RAGOnlyExtractor
from eval.external_benchmarks.utils import (
    ensure_dir,
    make_llm_client,
    now_utc_iso,
    sha256_text,
    write_json,
)
from eval.external_benchmarks.utils import get_or_create_bundle_for_text

COARSE_CLAUSE_KEYWORDS: Dict[str, List[str]] = {
    "termination": ["terminate", "termination", "expire", "end of term"],
    "liability": ["liability", "damages", "limitation of liability", "consequential"],
    "intellectual_property": ["intellectual property", "ip", "license", "ownership", "copyright", "patent"],
    "confidentiality": ["confidential", "confidentiality", "non-disclosure", "nda"],
    "non_compete": ["non compete", "non-compete", "solicit", "non-solicit", "restrictive covenant"],
    "indemnification": ["indemnify", "indemnification", "hold harmless"],
    "governing_law": ["governing law", "jurisdiction", "venue", "law of"],
    "assignment": ["assign", "assignment", "transfer"],
    "payment": ["payment", "fees", "invoice", "compensation", "price"],
    "audit_and_compliance": ["audit", "inspection", "compliance", "regulatory"],
}


def _load_cuad(local_path: Path, dataset_name: str):
    if local_path.exists():
        ds_dict = load_from_disk(str(local_path))
    else:
        ds_dict = load_dataset(dataset_name, trust_remote_code=True)
        ensure_dir(local_path.parent)
        ds_dict.save_to_disk(str(local_path))
    for split_name in ("test", "validation", "train"):
        if split_name in ds_dict:
            return split_name, ds_dict[split_name]
    return "train", ds_dict


def _match_categories(text: str) -> Set[str]:
    lower = text.lower()
    cats = set()
    for cat, keywords in COARSE_CLAUSE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            cats.add(cat)
    return cats


def _extract_contract_record(row: Dict[str, Any], idx: int) -> Tuple[str, str, str, Set[str]]:
    contract_id = str(row.get("title", row.get("contract_id", row.get("id", f"cuad-{idx}"))))
    context = str(row.get("context", row.get("document", row.get("text", ""))))
    question = str(row.get("question", row.get("clause_type", "")))
    # Gold coarse category inferred from clause question/label descriptor.
    gold_cats = _match_categories(question)
    return contract_id, context, question, gold_cats


def _group_contracts(ds: Any) -> Dict[str, Dict[str, Any]]:
    contracts: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(ds):
        cid, context, question, gold_cats = _extract_contract_record(row, idx)
        if cid not in contracts:
            contracts[cid] = {
                "contract_id": cid,
                "text": context,
                "gold_categories": set(),
                "questions": [],
            }
        contracts[cid]["gold_categories"].update(gold_cats)
        if question:
            contracts[cid]["questions"].append(question)
    # Remove contracts without text or without any mapped category.
    return {
        cid: v
        for cid, v in contracts.items()
        if isinstance(v.get("text"), str) and v["text"].strip() and v.get("gold_categories")
    }


def _select_contracts(contracts: Dict[str, Dict[str, Any]], num_contracts: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    items = list(contracts.values())
    if not items:
        return []

    # Deterministic tie-breaker order for stable greedy max-coverage selection.
    tie_order = list(range(len(items)))
    rng.shuffle(tie_order)
    for i, rank in enumerate(tie_order):
        items[i]["_tie_rank"] = rank

    selected: List[Dict[str, Any]] = []
    covered: Set[str] = set()
    remaining = list(items)

    while remaining and len(selected) < num_contracts:
        best = max(
            remaining,
            key=lambda x: (
                len(set(x["gold_categories"]) - covered),  # maximize new category coverage first
                len(x["gold_categories"]),                  # then prefer richer single contracts
                -int(x.get("_tie_rank", 0)),               # deterministic final tie-break
            ),
        )
        selected.append(best)
        covered.update(best["gold_categories"])
        remaining = [x for x in remaining if x["contract_id"] != best["contract_id"]]

    for entry in selected:
        entry.pop("_tie_rank", None)
    return selected


def _load_extracted_policies_from_cache(cache_root: Path, text: str) -> List[Dict[str, Any]]:
    doc_hash = sha256_text(text)
    doc_dir = cache_root / doc_hash
    jsonl_paths = [p for p in doc_dir.glob("*.jsonl") if "stage5" not in str(p)]
    if not jsonl_paths:
        return []
    policies: List[Dict[str, Any]] = []
    with open(jsonl_paths[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                policies.append(json.loads(line))
    return policies


def _policy_records_to_categories(policies: List[Dict[str, Any]]) -> Set[str]:
    cats: Set[str] = set()
    for p in policies:
        text_parts: List[str] = []
        if isinstance(p.get("description"), str):
            text_parts.append(p["description"])
        for c in p.get("conditions", []):
            if isinstance(c, dict):
                text_parts.extend(str(v) for v in c.values() if isinstance(v, (str, int, float, bool)))
        if isinstance(p.get("action_type"), str):
            text_parts.append(p["action_type"])
        joined = " ".join(text_parts)
        cats.update(_match_categories(joined))
    return cats


def _policyllm_policies_to_categories(policies: List[Dict[str, Any]]) -> Set[str]:
    cats: Set[str] = set()
    for p in policies:
        parts: List[str] = []
        for c in p.get("conditions", []):
            if isinstance(c, dict):
                parts.extend(str(v) for v in c.values() if isinstance(v, (str, int, float, bool)))
        for a in p.get("actions", []):
            if isinstance(a, dict):
                parts.extend(str(v) for v in a.values() if isinstance(v, (str, int, float, bool)))
        if isinstance(p.get("metadata", {}).get("source"), str):
            parts.append(p["metadata"]["source"])
        cats.update(_match_categories(" ".join(parts)))
    return cats


def _precision_recall(pred: Set[str], gold: Set[str]) -> Tuple[float, float]:
    if not pred:
        return 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return precision, recall


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


def run(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    split_name, ds = _load_cuad(Path(args.dataset_path), dataset_name=args.dataset_name)
    contracts = _group_contracts(ds)
    if args.selection_strategy == "max_coverage":
        selected = _select_contracts(contracts, num_contracts=args.num_contracts, seed=args.seed)
    else:
        rng = random.Random(args.seed)
        items = list(contracts.values())
        rng.shuffle(items)
        selected = items[: args.num_contracts]

    llm_client = make_llm_client(model_id=args.model, provider=args.llm_provider)
    rag_extractor = RAGOnlyExtractor(llm_client=llm_client, embedding_model=args.embedding_model)
    kw_extractor = KeywordRulesExtractor()

    cache_root = Path(args.cache_dir) / "cuad"
    config = load_config(args.config)

    per_contract: List[Dict[str, Any]] = []
    agg = {
        "policyllm": {"prec": [], "rec": []},
        "keyword_rules": {"prec": [], "rec": []},
        "rag": {"prec": [], "rec": []},
    }

    for c in selected:
        text = c["text"]
        gold = set(c["gold_categories"])
        contract_errors: List[str] = []

        # PolicyLLM extraction cache build.
        cache_info: Dict[str, Any] = {"cache_hit": False}
        policyllm_policies: List[Dict[str, Any]] = []
        if args.max_policyllm_chars > 0 and len(text) > args.max_policyllm_chars:
            chunks = _chunk_text(text, chunk_size=args.max_policyllm_chars, overlap=args.chunk_overlap_chars)
            chunk_hits: List[bool] = []
            for chunk_idx, chunk_text in enumerate(chunks):
                try:
                    bundle_path, chunk_cache = get_or_create_bundle_for_text(
                        text=chunk_text,
                        cache_root=cache_root,
                        config_path=Path(args.config),
                        tenant_id=args.tenant,
                        batch_prefix=f"cuad-chunk-{chunk_idx}",
                    )
                    _ = bundle_path
                    policyllm_policies.extend(_load_extracted_policies_from_cache(cache_root, chunk_text))
                    chunk_hits.append(bool(chunk_cache.get("cache_hit", False)))
                except Exception as e:  # noqa: BLE001
                    contract_errors.append(f"policyllm_chunk_{chunk_idx}_error: {e}")
            cache_info = {
                "cache_hit": all(chunk_hits) if chunk_hits else False,
                "chunked": True,
                "num_chunks": len(chunks),
            }
        else:
            try:
                bundle_path, cache_info = get_or_create_bundle_for_text(
                    text=text,
                    cache_root=cache_root,
                    config_path=Path(args.config),
                    tenant_id=args.tenant,
                    batch_prefix="cuad",
                )
                _ = bundle_path  # bundle produced as side-effect; extraction JSONL is read below.
                policyllm_policies = _load_extracted_policies_from_cache(cache_root, text)
            except Exception as e:  # noqa: BLE001
                contract_errors.append(f"policyllm_extraction_error: {e}")
        policyllm_pred = _policyllm_policies_to_categories(policyllm_policies)

        # Build temporary text file and regularize once for baseline extractors.
        doc_hash = sha256_text(text)
        text_path = cache_root / doc_hash / "cuad_contract.txt"
        ensure_dir(text_path.parent)
        text_path.write_text(text, encoding="utf-8")
        canonical = router.regularize(str(text_path), config)
        sections = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in canonical.sections]
        full_text = canonical.full_text

        try:
            kw_policies = kw_extractor.extract(sections, full_text)
        except Exception as e:  # noqa: BLE001
            contract_errors.append(f"keyword_baseline_error: {e}")
            kw_policies = []
        try:
            rag_policies = rag_extractor.extract(sections, full_text)
        except Exception as e:  # noqa: BLE001
            contract_errors.append(f"rag_baseline_error: {e}")
            rag_policies = []

        kw_pred = _policy_records_to_categories(kw_policies)
        rag_pred = _policy_records_to_categories(rag_policies)

        p_pol, r_pol = _precision_recall(policyllm_pred, gold)
        p_kw, r_kw = _precision_recall(kw_pred, gold)
        p_rag, r_rag = _precision_recall(rag_pred, gold)

        agg["policyllm"]["prec"].append(p_pol)
        agg["policyllm"]["rec"].append(r_pol)
        agg["keyword_rules"]["prec"].append(p_kw)
        agg["keyword_rules"]["rec"].append(r_kw)
        agg["rag"]["prec"].append(p_rag)
        agg["rag"]["rec"].append(r_rag)

        per_contract.append(
            {
                "contract_id": c["contract_id"],
                "gold_categories": sorted(gold),
                "pred_policyllm": sorted(policyllm_pred),
                "pred_keyword_rules": sorted(kw_pred),
                "pred_rag": sorted(rag_pred),
                "policyllm_precision": p_pol,
                "policyllm_recall": r_pol,
                "keyword_precision": p_kw,
                "keyword_recall": r_kw,
                "rag_precision": p_rag,
                "rag_recall": r_rag,
                "cache_hit": cache_info.get("cache_hit", False),
                "chunked_policyllm": bool(cache_info.get("chunked", False)),
                "num_policyllm_chunks": int(cache_info.get("num_chunks", 1)),
                "errors": contract_errors,
            }
        )

    def avg(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    result = {
        "benchmark": "CUAD",
        "split": split_name,
        "dataset_source": str(args.dataset_path),
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "num_contracts": args.num_contracts,
        "sample_ids": [c["contract_id"] for c in selected],
        "coarse_mapping": COARSE_CLAUSE_KEYWORDS,
        "coarse_mapping_version": "v1",
        "model_settings": {
            "llm_provider": args.llm_provider,
            "backbone_model": args.model,
            "embedding_model": args.embedding_model,
            "selection_strategy": args.selection_strategy,
            "max_policyllm_chars": args.max_policyllm_chars,
            "chunk_overlap_chars": args.chunk_overlap_chars,
        },
        "metrics": {
            "policyllm": {
                "clause_type_precision": avg(agg["policyllm"]["prec"]),
                "clause_type_recall": avg(agg["policyllm"]["rec"]),
            },
            "keyword_rules": {
                "clause_type_precision": avg(agg["keyword_rules"]["prec"]),
                "clause_type_recall": avg(agg["keyword_rules"]["rec"]),
            },
            "rag": {
                "clause_type_precision": avg(agg["rag"]["prec"]),
                "clause_type_recall": avg(agg["rag"]["rec"]),
            },
        },
        "per_contract": per_contract,
        "runtime": {
            "seconds": time.time() - start,
            "completed_at_utc": now_utc_iso(),
        },
        "summary_row": {
            "benchmark": "CUAD",
            "method": "PolicyLLM",
            "precision": avg(agg["policyllm"]["prec"]),
            "recall": avg(agg["policyllm"]["rec"]),
        },
    }
    write_json(Path(args.output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUAD extraction benchmark adapter")
    parser.add_argument("--dataset-name", default="theatticusproject/cuad-qa")
    parser.add_argument("--dataset-path", default=str(ROOT / "eval" / "external_benchmarks" / "cuad"))
    parser.add_argument("--cache-dir", default=str(ROOT / "results" / "external" / "cache"))
    parser.add_argument("--config", default=str(ROOT / "Extractor" / "configs" / "config.chatgpt.yaml"))
    parser.add_argument("--tenant", default="external_cuad")
    parser.add_argument("--llm-provider", default="chatgpt")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--num-contracts", type=int, default=10)
    parser.add_argument("--selection-strategy", choices=["random", "max_coverage"], default="random")
    parser.add_argument("--max-policyllm-chars", type=int, default=30000)
    parser.add_argument("--chunk-overlap-chars", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "results" / "external" / "cuad_results.json"))
    args = parser.parse_args()

    if args.dataset_path.endswith("cuad"):
        dataset_slug = args.dataset_name.replace("/", "__")
        args.dataset_path = str(Path(args.dataset_path) / dataset_slug)

    result = run(args)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
