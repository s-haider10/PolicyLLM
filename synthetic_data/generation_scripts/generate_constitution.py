#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

DOMAINS = ["refund", "privacy", "security", "shipping", "returns"]
PRIORITIES = ["regulatory", "company", "department", "situational"]
ROLES = ["customer_support", "refund_team", "manager", "privacy_officer", "security_officer"]


def build_policy(index: int, rng: random.Random) -> dict:
    domain = DOMAINS[index % len(DOMAINS)]
    priority = PRIORITIES[index % len(PRIORITIES)]
    role = ROLES[index % len(ROLES)]

    if domain in {"refund", "returns"}:
        days = 7 + (index % 4) * 7
        policy = {
            "policy_id": f"POL-{domain.upper()}-{index:03d}",
            "title": f"{domain.title()} window policy {index}",
            "domain": domain,
            "priority": priority,
            "owner": role,
            "scope": {"product_category": ["electronics", "appliances", "accessories"][index % 3]},
            "conditions": [
                {"field": "has_receipt", "op": "==", "value": True},
                {"field": "days_since_purchase", "op": "<=", "value": days},
            ],
            "actions": [{"type": "required", "value": "offer_refund"}],
            "exceptions": [{"field": "physical_damage", "op": "==", "value": True, "effect": "deny_refund"}],
            "metadata": {"effective_date": "2026-01-01", "version": "1.0"},
        }
    elif domain == "privacy":
        policy = {
            "policy_id": f"POL-PRIVACY-{index:03d}",
            "title": f"PII handling policy {index}",
            "domain": domain,
            "priority": priority,
            "owner": role,
            "scope": {"channel": ["chat", "email", "phone"][index % 3]},
            "conditions": [{"field": "contains_pii", "op": "==", "value": True}],
            "actions": [{"type": "prohibited", "value": "disclose_pii"}],
            "exceptions": [],
            "metadata": {"effective_date": "2026-01-01", "version": "1.0"},
        }
    elif domain == "security":
        threshold = 200 + (index % 5) * 100
        policy = {
            "policy_id": f"POL-SEC-{index:03d}",
            "title": f"Identity verification threshold {index}",
            "domain": domain,
            "priority": priority,
            "owner": role,
            "scope": {"operation": "refund_approval"},
            "conditions": [{"field": "refund_amount", "op": ">=", "value": threshold}],
            "actions": [{"type": "required", "value": "verify_identity"}],
            "exceptions": [],
            "metadata": {"effective_date": "2026-01-01", "version": "1.0"},
        }
    else:  # shipping
        policy = {
            "policy_id": f"POL-SHIP-{index:03d}",
            "title": f"Shipping SLA policy {index}",
            "domain": domain,
            "priority": priority,
            "owner": role,
            "scope": {"region": ["domestic", "international"][index % 2]},
            "conditions": [{"field": "item_in_stock", "op": "==", "value": True}],
            "actions": [{"type": "required", "value": "ship_within_48h"}],
            "exceptions": [{"field": "severe_weather_alert", "op": "==", "value": True, "effect": "allow_delay"}],
            "metadata": {"effective_date": "2026-01-01", "version": "1.0"},
        }

    if rng.random() < 0.2:
        policy["metadata"]["tags"] = ["customer-facing", "critical"][0:1]
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic ground-truth constitution.")
    parser.add_argument("--num-policies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policies = [build_policy(i + 1, rng) for i in range(args.num_policies)]

    args.out.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "1.0",
        "seed": args.seed,
        "num_policies": args.num_policies,
        "policies": policies,
    }

    out_file = args.out / "ground_truth_constitution.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
