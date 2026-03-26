"""Shared utilities for external benchmark adapters.

This module centralizes:
- deterministic split persistence (dev/test IDs)
- threshold calibration and score->label mapping
- classification metrics
- text->bundle caching for PolicyLLM extraction/validation
- common JSON/CSV artifact writing
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)


def _stratified_round_robin_sample(
    items: Sequence[Any],
    label_fn: Callable[[Any], str],
    n: int,
    rng: random.Random,
) -> List[Any]:
    grouped: Dict[str, List[Any]] = {}
    for item in items:
        grouped.setdefault(str(label_fn(item)), []).append(item)

    for label in sorted(grouped.keys()):
        rng.shuffle(grouped[label])

    labels = sorted(grouped.keys())
    selected: List[Any] = []
    while len(selected) < n:
        progressed = False
        for label in labels:
            bucket = grouped[label]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break

    if len(selected) < n:
        raise ValueError(f"Unable to sample {n} examples stratified; only {len(selected)} available")
    return selected


def build_or_load_split(
    split_path: Path,
    examples: Sequence[Any],
    id_fn: Callable[[Any], str],
    label_fn: Callable[[Any], str],
    dev_n: int,
    test_n: int,
    seed: int,
    split_tag: str | None = None,
) -> Dict[str, Any]:
    """Build deterministic dev/test split and persist IDs, or load existing split."""
    ensure_dir(split_path.parent)

    by_id = {id_fn(ex): ex for ex in examples}
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            payload = json.load(f)
        if split_tag and payload.get("split_tag") != split_tag:
            payload = {}
        else:
            dev_ids = payload.get("dev_ids", [])
            test_ids = payload.get("test_ids", [])
            dev = [by_id[i] for i in dev_ids if i in by_id]
            test = [by_id[i] for i in test_ids if i in by_id]
            if len(dev) >= dev_n and len(test) >= test_n:
                return {
                    "seed": payload.get("seed", seed),
                    "dev": dev[:dev_n],
                    "test": test[:test_n],
                    "dev_ids": dev_ids[:dev_n],
                    "test_ids": test_ids[:test_n],
                    "persisted": True,
                    "split_tag": payload.get("split_tag"),
                }

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    dev = _stratified_round_robin_sample(shuffled, label_fn=label_fn, n=dev_n, rng=rng)
    dev_ids = {id_fn(ex) for ex in dev}
    remaining = [ex for ex in shuffled if id_fn(ex) not in dev_ids]
    test = _stratified_round_robin_sample(remaining, label_fn=label_fn, n=test_n, rng=rng)

    payload = {
        "seed": seed,
        "dev_ids": [id_fn(ex) for ex in dev],
        "test_ids": [id_fn(ex) for ex in test],
        "created_at_utc": now_utc_iso(),
        "split_tag": split_tag,
    }
    write_json(split_path, payload)

    return {
        "seed": seed,
        "dev": dev,
        "test": test,
        "dev_ids": payload["dev_ids"],
        "test_ids": payload["test_ids"],
        "persisted": False,
        "split_tag": split_tag,
    }


def map_ternary_label(
    score: float,
    has_violation: bool,
    pos_threshold: float,
    neg_threshold: float,
    positive_label: str,
    negative_label: str,
    neutral_label: str,
) -> str:
    # Decoupled mapping: score is the primary signal.
    # Keep has_violation in the signature for backward-compatible call sites/artefacts.
    _ = has_violation
    if score >= pos_threshold:
        return positive_label
    if score < neg_threshold:
        return negative_label
    return neutral_label


def calibrate_ternary_thresholds(
    dev_rows: Sequence[Dict[str, Any]],
    label_key: str,
    positive_label: str,
    negative_label: str,
    neutral_label: str,
    score_key: str = "score",
    has_violation_key: str = "has_violation",
    default_pos_threshold: float = 0.85,
    default_neg_threshold: float = 0.70,
) -> Dict[str, Any]:
    y_true = [str(r[label_key]) for r in dev_rows]
    labels = [positive_label, negative_label, neutral_label]

    best = {
        "positive_threshold": default_pos_threshold,
        "negative_threshold": default_neg_threshold,
        "macro_f1": -1.0,
    }

    # Grid sweep with clear ordering constraint (neg < pos).
    pos_grid = [round(v, 2) for v in _frange(0.70, 0.96, 0.01)]
    neg_grid = [round(v, 2) for v in _frange(0.40, 0.86, 0.01)]

    for pos_t in pos_grid:
        for neg_t in neg_grid:
            if neg_t >= pos_t:
                continue
            y_pred = [
                map_ternary_label(
                    score=float(r[score_key]),
                    has_violation=bool(r[has_violation_key]),
                    pos_threshold=pos_t,
                    neg_threshold=neg_t,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    neutral_label=neutral_label,
                )
                for r in dev_rows
            ]
            macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
            if macro > best["macro_f1"]:
                best = {
                    "positive_threshold": pos_t,
                    "negative_threshold": neg_t,
                    "macro_f1": macro,
                }

    return best


def map_binary_label(score: float, threshold: float, positive_label: str, negative_label: str) -> str:
    return positive_label if score >= threshold else negative_label


def calibrate_binary_threshold(
    dev_rows: Sequence[Dict[str, Any]],
    label_key: str,
    positive_label: str,
    negative_label: str,
    score_key: str = "score",
    default_threshold: float = 0.85,
) -> Dict[str, Any]:
    y_true = [str(r[label_key]) for r in dev_rows]
    labels = [positive_label, negative_label]

    best = {
        "threshold": default_threshold,
        "macro_f1": -1.0,
    }
    for t in [round(v, 2) for v in _frange(0.40, 0.96, 0.01)]:
        y_pred = [map_binary_label(float(r[score_key]), t, positive_label, negative_label) for r in dev_rows]
        macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        if macro > best["macro_f1"]:
            best = {"threshold": t, "macro_f1": macro}
    return best


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    label_list = list(labels)
    accuracy = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0)) if y_true else 0.0
    per_class_scores = f1_score(y_true, y_pred, labels=label_list, average=None, zero_division=0) if y_true else []
    per_class = {label: float(score) for label, score in zip(label_list, per_class_scores)}
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
    }


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    v = start
    # include stop if exact; guard floating-point drift.
    while v <= stop + 1e-9:
        yield v
        v += step


def get_or_create_bundle_for_text(
    text: str,
    cache_root: Path,
    config_path: Path,
    tenant_id: str,
    batch_prefix: str,
) -> Tuple[Path, Dict[str, Any]]:
    """Compile (or reuse) a PolicyLLM bundle from raw text.

    Returns (bundle_path, cache_info).
    """
    doc_hash = sha256_text(text)
    doc_dir = cache_root / doc_hash
    ensure_dir(doc_dir)

    bundle_path = doc_dir / "compiled_policy_bundle.json"
    if bundle_path.exists():
        return bundle_path, {"doc_hash": doc_hash, "cache_hit": True}

    text_path = doc_dir / "source.txt"
    text_path.write_text(text, encoding="utf-8")

    from Extractor.src.config import load_config
    from Extractor.src import pipeline as extractor_pipeline
    from Validation.bundle_compiler import compile_from_policies, write_bundle

    config = load_config(str(config_path))
    try:
        extractor_pipeline.run_pipeline(
            input_path=str(text_path),
            output_dir=str(doc_dir),
            tenant_id=tenant_id,
            batch_id=f"{batch_prefix}-{doc_hash[:8]}",
            config=config,
            stage5_input=None,
        )
    except Exception as e:
        # Offline dry-run support: the extraction stub can fail schema validation
        # on some prompts. In that case, emit a small deterministic fallback bundle
        # so adapter integration paths can still be validated end-to-end.
        if getattr(config.llm, "provider", "") != "stub":
            raise

        fallback_policy = {
            "policy_id": f"POL-STUB-{doc_hash[:8]}",
            "conditions": [
                {
                    "type": "boolean_flag",
                    "parameter": "always_true",
                    "operator": "==",
                    "value": True,
                    "source_text": "stub fallback condition",
                }
            ],
            "actions": [
                {
                    "type": "required",
                    "action": "follow_contract_policy",
                    "requires": ["always_true"],
                    "source_text": text[:300],
                }
            ],
            "metadata": {
                "domain": "other",
                "priority": "informational",
                "source": "stub_fallback",
            },
        }
        fallback_jsonl = doc_dir / "fallback_policies.jsonl"
        with open(fallback_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(fallback_policy) + "\n")
        bundle = compile_from_policies([fallback_policy])
        write_bundle(bundle, str(bundle_path))
        return bundle_path, {
            "doc_hash": doc_hash,
            "cache_hit": False,
            "num_policies": 1,
            "stub_fallback": True,
            "pipeline_error": str(e),
        }

    jsonl_candidates = [p for p in doc_dir.glob("*.jsonl") if "stage5" not in str(p)]
    if not jsonl_candidates:
        raise RuntimeError(f"No extracted policy JSONL found under {doc_dir}")

    policies: List[Dict[str, Any]] = []
    with open(jsonl_candidates[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                policies.append(json.loads(line))

    bundle = compile_from_policies(policies)
    write_bundle(bundle, str(bundle_path))
    return bundle_path, {"doc_hash": doc_hash, "cache_hit": False, "num_policies": len(policies)}


def run_policyllm_enforcement(
    bundle_path: Path,
    query: str,
    asserted_response: str,
    llm_client: Any,
    response_mode: str = "asserted_probe",
) -> Dict[str, Any]:
    """Run enforcement deterministically on a fixed asserted response."""
    from Enforcement import load_bundle, enforce
    from Enforcement.orchestrator import EnforcementConfig

    try:
        bundle, index = load_bundle(str(bundle_path))
    except Exception:
        # Some extracted bundles can include `owners_to_notify: [null]`.
        # Sanitize obvious null-owner escalations once, then retry load.
        if not _sanitize_bundle_owners(bundle_path):
            raise
        bundle, index = load_bundle(str(bundle_path))
    config = EnforcementConfig(
        judge_enabled=True,
        smt_enabled=True,
        regex_enabled=True,
        max_retries=0,
        auto_correct_max_attempts=0,
    )

    if response_mode == "asserted_probe":
        generate_fn = lambda _prompt: asserted_response
    elif response_mode == "generated_response":
        generate_fn = lambda prompt: _generate_response_from_prompt(llm_client, prompt)
    else:
        raise ValueError(f"Unsupported response_mode '{response_mode}'")

    start = time.time()
    decision = enforce(
        query=query,
        bundle=bundle,
        bundle_index=index,
        llm_client=llm_client,
        judge_llm_client=llm_client,
        config=config,
        generate_fn=generate_fn,
    )
    return {
        "action": decision.action.value,
        "score": float(decision.score),
        "violations": list(decision.violations),
        "has_violation": bool(decision.violations),
        "latency_s": time.time() - start,
        "response_mode": response_mode,
    }


def _generate_response_from_prompt(llm_client: Any, prompt: Dict[str, str]) -> str:
    system_text = prompt.get("system", "") if isinstance(prompt, dict) else ""
    user_text = prompt.get("user", "") if isinstance(prompt, dict) else str(prompt)
    generation_prompt = (
        "Produce a natural assistant response to the user query under the provided guidance.\n"
        "Do not include analysis. Return only the final response text.\n\n"
        f"System guidance:\n{system_text}\n\n"
        f"User prompt:\n{user_text}\n\n"
        'Return JSON: {"response": "<final response text>"}'
    )
    result = llm_client.invoke_json(generation_prompt)
    if isinstance(result, dict):
        response = result.get("response")
        if isinstance(response, str) and response.strip():
            return response
    if isinstance(result, str):
        return result
    return str(result)


def _sanitize_bundle_owners(bundle_path: Path) -> bool:
    try:
        with open(bundle_path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False

    changed = False
    default_owner = "policy-owner@company.com"
    escalations = raw.get("escalations")
    if isinstance(escalations, list):
        for esc in escalations:
            if not isinstance(esc, dict):
                continue
            owners = esc.get("owners_to_notify")
            if isinstance(owners, list):
                cleaned = [o for o in owners if isinstance(o, str) and o.strip()]
                if not cleaned:
                    cleaned = [default_owner]
                if cleaned != owners:
                    esc["owners_to_notify"] = cleaned
                    changed = True
            elif owners is None:
                esc["owners_to_notify"] = [default_owner]
                changed = True

    if changed:
        with open(bundle_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
    return changed


def make_llm_client(model_id: str = "gpt-4o-mini", provider: str = "chatgpt") -> Any:
    from Extractor.src.llm.client import LLMClient

    if provider == "chatgpt" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it before running chatgpt-backed external benchmarks, "
            "or use --llm-provider stub for offline dry runs."
        )

    return LLMClient(
        provider=provider,
        model_id=model_id,
        temperature=0.0,
        max_tokens=2048,
    )
