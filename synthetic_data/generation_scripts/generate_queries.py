#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from ollama_client import OllamaClient


def load_constitution(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["policies"]


def build_query(policy: dict, category: str, rng: random.Random, client: Optional[OllamaClient]) -> dict:
    domain = policy["domain"]
    pid = policy["policy_id"]

    if category == "valid_path":
        deterministic = f"Customer asks about {domain}; conditions match policy {pid}. What is the compliant action?"
        expected = "PASS"
        instruction = "Generate a realistic user query that should clearly comply with the given policy."
    elif category == "violation":
        deterministic = f"Agent proposes violating {pid} by ignoring required constraints for {domain}. Is that allowed?"
        expected = "ESCALATE"
        instruction = "Generate a realistic user query that should clearly violate policy constraints."
    elif category == "uncovered":
        deterministic = f"Question about unrelated scenario outside known {domain} scope and without policy conditions."
        expected = rng.choice(["REGENERATE", "ESCALATE"])
        instruction = "Generate a query that is plausibly relevant but not directly covered by policy conditions."
    else:  # edge_case
        deterministic = f"Boundary case for {pid}: one condition is near threshold. How should system respond?"
        expected = rng.choice(["PASS", "AUTO_CORRECT", "REGENERATE"])
        instruction = "Generate a boundary-condition query that is ambiguous around policy thresholds."

    if client is None:
        text = deterministic
    else:
        prompt = (
            f"{instruction}\n"
            "Return only one user query sentence or short paragraph, no bullet list.\n\n"
            f"Policy ID: {pid}\n"
            f"Domain: {domain}\n"
            f"Policy Conditions: {json.dumps(policy.get('conditions', []))}\n"
            f"Policy Actions: {json.dumps(policy.get('actions', []))}\n"
        )
        text = client.generate(prompt)

    return {
        "query_id": f"Q-{rng.randint(100000, 999999)}",
        "category": category,
        "policy_id": pid,
        "query": text,
        "expected_action": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic query set with expected compliance outcomes.")
    parser.add_argument("--constitution", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="mistral:latest")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-llm", action="store_true", help="Disable Ollama calls and use deterministic templates")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policies = load_constitution(args.constitution)
    llm_client = None if args.no_llm else OllamaClient(model=args.model, host=args.ollama_host, temperature=args.temperature)

    categories = [
        ("valid_path", 0.4),
        ("violation", 0.3),
        ("uncovered", 0.2),
        ("edge_case", 0.1),
    ]

    weighted = [c for c, _ in categories]
    weights = [w for _, w in categories]

    queries = []
    for _ in range(args.num_queries):
        category = rng.choices(weighted, weights=weights, k=1)[0]
        policy = rng.choice(policies)
        queries.append(build_query(policy, category, rng, llm_client))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "num_queries": args.num_queries,
        "model": None if args.no_llm else args.model,
        "ollama_host": None if args.no_llm else args.ollama_host,
        "queries": queries,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
