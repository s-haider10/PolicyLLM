#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from bedrock_client import BedrockClient


def load_constitution(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["policies"]


def build_query(policy: dict, category: str, rng: random.Random, client: Optional[BedrockClient]) -> dict:
    domain = policy["domain"]
    pid = policy["policy_id"]

    if category == "valid_path":
        deterministic = f"Customer asks about {domain}; conditions match policy {pid}. What is the compliant action?"
        expected = "PASS"
        instruction = (
            f"Generate a realistic user query (1-3 sentences) that clearly complies with the policy for {domain}. "
            f"The query must satisfy these conditions: {json.dumps(policy.get('conditions', []))}. "
            "Output ONLY the user query, nothing else."
        )
    elif category == "violation":
        deterministic = f"Agent proposes violating {pid} by ignoring required constraints for {domain}. Is that allowed?"
        expected = "ESCALATE"
        instruction = (
            f"Generate a realistic user query (1-3 sentences) that clearly violates the policy for {domain}. "
            f"The query should violate at least one of these conditions: {json.dumps(policy.get('conditions', []))}. "
            "Output ONLY the user query, nothing else."
        )
    elif category == "uncovered":
        deterministic = f"Question about unrelated scenario outside known {domain} scope and without policy conditions."
        expected = rng.choice(["REGENERA TE", "ESCALATE"])
        instruction = (
            f"Generate a user query (1-3 sentences) related to {domain} but not directly covered by the policy conditions. "
            f"The policy conditions are: {json.dumps(policy.get('conditions', []))}. Make the query ambiguous. "
            "Output ONLY the user query, nothing else."
        )
    else:  # edge_case
        deterministic = f"Boundary case for {pid}: one condition is near threshold. How should system respond?"
        expected = rng.choice(["PASS", "AUTO_CORRECT", "REGENERATE"])
        instruction = (
            f"Generate a boundary-condition query (1-3 sentences) for {domain} where conditions are right at the threshold. "
            f"The conditions are: {json.dumps(policy.get('conditions', []))}. Make it genuinely unclear. "
            "Output ONLY the user query, nothing else."
        )

    if client is None:
        text = deterministic
    else:
        text = client.generate(instruction)

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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policies = load_constitution(args.constitution)
    
    # Initialize Claude via AWS Bedrock
    print("Using Claude via AWS Bedrock API")
    llm_client = BedrockClient(model_name="claude-opus-4-5-20251101")

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
        "model": "claude-opus-4-5-20251101",
        "provider": "aws-bedrock",
        "temperature": args.temperature,
        "queries": queries,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
