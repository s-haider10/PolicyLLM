#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from bedrock_client import BedrockClient


def load_constitution(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["policies"]


def explicit_render(policy: dict) -> str:
    conds = " and ".join([f"{c['field']} {c['op']} {c['value']}" for c in policy["conditions"]])
    action = ", ".join(a["value"] for a in policy["actions"])
    exceptions = policy.get("exceptions", [])
    exc = ""
    if exceptions:
        e = exceptions[0]
        exc = f" Exception: if {e['field']} {e['op']} {e['value']}, then {e['effect']}."
    return (
        f"Policy {policy['policy_id']} ({policy['priority']} priority). "
        f"If {conds}, agents must {action}.{exc}"
    )


def llm_render_policy(
    policy: dict,
    style: str,
    client: Optional[BedrockClient],
    rng: random.Random,
    include_conflict: bool = False,
) -> str:
    base_text = explicit_render(policy) if style == "explicit" else implicit_render(policy, rng)
    if include_conflict:
        base_text = f"{base_text}\n{conflicting_statement(policy)}"

    if client is None:
        return base_text

    # Clear, stage-specific instructions for the LLM
    if style == "explicit":
        instruction = (
            "Rewrite the following policy as a clear, direct statement in one paragraph. "
            "State all conditions and actions explicitly using formal policy language. "
            "Keep all specific values and constraints exactly as given. "
            "Output ONLY the rewritten policy text, nothing else.\n\n"
            f"POLICY: {base_text}\n\n"
            "REWRITTEN POLICY:"
        )
    elif style == "implicit":
        instruction = (
            "Rewrite the following policy as an implicit guideline using natural, conversational language. "
            "Express it indirectly through historical practices using phrases like 'typically' or 'usually'. "
            "Keep the same constraints but make them less obvious. "
            "Output ONLY the rewritten policy text, nothing else.\n\n"
            f"POLICY: {base_text}\n\n"
            "REWRITTEN POLICY:"
        )
    elif style == "hybrid":
        instruction = (
            "Rewrite the following policy mixing explicit and implicit styles. "
            "Combine formal statements with informal examples. "
            "Keep all constraints present but vary the tone. "
            "Output ONLY the rewritten policy text, nothing else.\n\n"
            f"POLICY: {base_text}\n\n"
            "REWRITTEN POLICY:"
        )
    else:
        instruction = (
            f"Rewrite the following policy in {style} style as one natural paragraph. "
            "Keep all factual constraints unchanged. "
            "Output ONLY the rewritten policy text, nothing else.\n\n"
            f"POLICY: {base_text}\n\n"
            "REWRITTEN POLICY:"
        )
    
    return client.generate(instruction)


def implicit_render(policy: dict, rng: random.Random) -> str:
    templates = [
        "In most cases, teams treat {domain} requests as valid only when {hint}; this usually leads to {action}.",
        "Historically, when handling {domain}, staff first check whether {hint}, and outcomes tend to be {action}.",
        "As a rule of thumb in {domain}, unless special circumstances apply, the path often ends with {action} after confirming {hint}.",
    ]
    hint = " and ".join([f"{c['field']} {c['op']} {c['value']}" for c in policy["conditions"]])
    action = ", ".join(a["value"] for a in policy["actions"])
    return rng.choice(templates).format(domain=policy["domain"], hint=hint, action=action)


def conflicting_statement(policy: dict) -> str:
    if not policy["conditions"]:
        return "Low-priority local note: follow team discretion."
    c0 = policy["conditions"][0]
    if c0["op"] == "<=":
        alt = {**c0, "op": ">", "value": c0["value"]}
    elif c0["op"] == "==":
        alt = {**c0, "op": "==", "value": (not c0["value"]) if isinstance(c0["value"], bool) else c0["value"]}
    else:
        alt = {**c0, "op": "<=", "value": c0["value"]}

    return (
        f"Low-priority department exception: if {alt['field']} {alt['op']} {alt['value']}, "
        f"override standard behavior for {policy['policy_id']}."
    )


def write_doc(path: Path, title: str, body_lines: List[str]) -> None:
    text = "# " + title + "\n\n" + "\n\n".join(body_lines) + "\n"
    path.write_text(text, encoding="utf-8")


def generate_stage_docs(
    stage: int,
    policies: List[dict],
    num_documents: int,
    policies_per_doc: int,
    rng: random.Random,
    out_dir: Path,
    conflict_rate: float,
    llm_client: Optional[BedrockClient],
) -> List[Dict]:
    docs_meta = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_idx in range(1, num_documents + 1):
        selected = rng.sample(policies, k=min(policies_per_doc, len(policies)))
        body = []
        conflicts = []

        for p in selected:
            if stage == 1:
                body.append(llm_render_policy(p, "explicit", llm_client, rng))
            elif stage == 2:
                body.append(llm_render_policy(p, "explicit", llm_client, rng))
                if rng.random() < conflict_rate:
                    conflict = llm_render_policy(p, "explicit", llm_client, rng, include_conflict=True)
                    body.append(conflict)
                    conflicts.append({"policy_id": p["policy_id"], "conflict_text": conflict})
            elif stage == 3:
                body.append(llm_render_policy(p, "implicit", llm_client, rng))
            else:
                raise ValueError("stage must be 1,2,3 in this function")

        file_path = out_dir / f"doc_{doc_idx:03d}.md"
        write_doc(file_path, f"Stage {stage} synthetic document {doc_idx}", body)

        docs_meta.append(
            {
                "doc_id": f"doc_{doc_idx:03d}",
                "path": str(file_path),
                "stage": stage,
                "policy_ids": [p["policy_id"] for p in selected],
                "conflicts": conflicts,
            }
        )

    return docs_meta


def generate_stage4_docs(
    policies: List[dict],
    num_documents: int,
    policies_per_doc: int,
    rng: random.Random,
    out_dir: Path,
    distribution: List[float],
    llm_client: Optional[BedrockClient],
) -> List[Dict]:
    labels = ["explicit", "conflict", "implicit", "hybrid"]
    docs_meta = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_idx in range(1, num_documents + 1):
        profile = rng.choices(labels, weights=distribution, k=1)[0]
        selected = rng.sample(policies, k=min(policies_per_doc, len(policies)))
        body = []
        conflicts = []

        for p in selected:
            if profile == "explicit":
                body.append(llm_render_policy(p, "explicit", llm_client, rng))
            elif profile == "conflict":
                body.append(llm_render_policy(p, "explicit", llm_client, rng))
                if rng.random() < 0.5:
                    conflict = llm_render_policy(p, "explicit", llm_client, rng, include_conflict=True)
                    body.append(conflict)
                    conflicts.append({"policy_id": p["policy_id"], "conflict_text": conflict})
            elif profile == "implicit":
                body.append(llm_render_policy(p, "implicit", llm_client, rng))
            else:  # hybrid
                body.append(
                    llm_render_policy(p, "explicit", llm_client, rng)
                    if rng.random() < 0.5
                    else llm_render_policy(p, "implicit", llm_client, rng)
                )
                if rng.random() < 0.25:
                    conflict = llm_render_policy(p, "hybrid", llm_client, rng, include_conflict=True)
                    body.append(conflict)
                    conflicts.append({"policy_id": p["policy_id"], "conflict_text": conflict})

        file_path = out_dir / f"doc_{doc_idx:03d}.md"
        write_doc(file_path, f"Stage 4 ({profile}) synthetic document {doc_idx}", body)

        docs_meta.append(
            {
                "doc_id": f"doc_{doc_idx:03d}",
                "path": str(file_path),
                "stage": 4,
                "profile": profile,
                "policy_ids": [p["policy_id"] for p in selected],
                "conflicts": conflicts,
            }
        )

    return docs_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic documents for stage experiments.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--constitution", type=Path, required=True)
    parser.add_argument("--num-documents", type=int, default=20)
    parser.add_argument("--policies-per-doc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conflict-rate", type=float, default=0.35)
    parser.add_argument("--distribution", type=str, default="0.5,0.15,0.25,0.1")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policies = load_constitution(args.constitution)
    
    # Initialize Claude via AWS Bedrock
    print("Using Claude via AWS Bedrock API")
    llm_client = BedrockClient(model_name="claude-opus-4-5-20251101")

    if args.stage in {1, 2, 3}:
        docs_meta = generate_stage_docs(
            stage=args.stage,
            policies=policies,
            num_documents=args.num_documents,
            policies_per_doc=args.policies_per_doc,
            rng=rng,
            out_dir=args.out,
            conflict_rate=args.conflict_rate,
            llm_client=llm_client,
        )
    else:
        distribution = [float(x.strip()) for x in args.distribution.split(",")]
        if len(distribution) != 4:
            raise ValueError("--distribution must contain exactly 4 comma-separated values")
        if abs(sum(distribution) - 1.0) > 1e-6:
            raise ValueError("--distribution must sum to 1.0")
        docs_meta = generate_stage4_docs(
            policies=policies,
            num_documents=args.num_documents,
            policies_per_doc=args.policies_per_doc,
            rng=rng,
            out_dir=args.out,
            distribution=distribution,
            llm_client=llm_client,
        )

    manifest = {
        "stage": args.stage,
        "seed": args.seed,
        "num_documents": args.num_documents,
        "policies_per_doc": args.policies_per_doc,
        "model": "claude-opus-4-5-20251101",
        "provider": "aws-bedrock",
        "temperature": args.temperature,
        "documents": docs_meta,
    }
    manifest_path = args.out.parent / "document_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Generated {len(docs_meta)} documents in {args.out}")


if __name__ == "__main__":
    main()
