"""PolicyLLM — End-to-end pipeline: Extract -> Validate -> Enforce.

Usage:
    # Full pipeline: document -> extraction -> bundle -> enforcement
    python main.py run input.pdf --query "I want a refund" --provider ollama --model mistral:latest

    # Individual stages
    python main.py extract input.pdf --out out/ --config configs/config.example.yaml
    python main.py validate out/policies.jsonl --out compiled_policy_bundle.json
    python main.py enforce --bundle compiled_policy_bundle.json --query "I want a refund"
"""
import argparse
import json
import os
import sys
import logging

logger = logging.getLogger("PolicyLLM")


def cmd_extract(args):
    """Stage 1: Extract policies from documents."""
    sys.path.insert(0, ".")
    from Extractor.src.config import load_config
    from Extractor.src import pipeline

    config = load_config(args.config)
    pipeline.run_pipeline(
        input_path=args.input,
        output_dir=args.out,
        tenant_id=args.tenant,
        batch_id=args.batch,
        config=config,
        stage5_input=None,
    )
    logger.info("Extraction complete. Output in %s/", args.out)


def cmd_validate(args):
    """Stage 2: Compile extracted policies into an enforcement bundle."""
    from Validation.bundle_compiler import compile_from_policies, write_bundle

    policies = []
    with open(args.input, encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            policies = json.loads(content)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    policies.append(json.loads(line))

    if not policies:
        logger.error("No policies found in %s", args.input)
        sys.exit(1)

    bundle = compile_from_policies(policies)
    write_bundle(bundle, args.out)
    meta = bundle["bundle_metadata"]
    logger.info(
        "Bundle written to %s (%d rules, %d constraints, %d paths)",
        args.out, meta["policy_count"], meta["constraint_count"], meta["path_count"],
    )


def cmd_enforce(args):
    """Stage 3: Enforce policies against a query."""
    from Enforcement import load_bundle, enforce, ComplianceAction
    from Enforcement.orchestrator import EnforcementConfig
    from Enforcement.audit import AuditLogger

    sys.path.insert(0, ".")
    from Extractor.src.llm.client import LLMClient

    bundle, index = load_bundle(args.bundle)

    llm = LLMClient(
        provider=args.provider,
        model_id=args.model,
        temperature=0.0,
        max_tokens=2048,
    )

    judge_llm = llm
    if args.judge_model:
        judge_llm = LLMClient(
            provider=args.provider,
            model_id=args.judge_model,
            temperature=0.0,
            max_tokens=1024,
        )

    config = EnforcementConfig(
        judge_enabled=not args.no_judge,
        smt_enabled=not args.no_smt,
    )

    audit = AuditLogger(args.audit_log) if args.audit_log else None

    generate_fn = None
    if args.response:
        generate_fn = lambda _prompt: args.response

    decision = enforce(
        query=args.query,
        bundle=bundle,
        bundle_index=index,
        llm_client=llm,
        judge_llm_client=judge_llm,
        config=config,
        generate_fn=generate_fn,
        audit_logger=audit,
    )

    print(json.dumps(decision.model_dump(), indent=2, default=str))


def cmd_run(args):
    """Full pipeline: extract -> validate -> enforce."""
    # Stage 1: Extract
    extract_out = args.out or "out"
    logger.info("=== Stage 1: Extraction ===")

    sys.path.insert(0, ".")
    from Extractor.src.config import load_config
    from Extractor.src import pipeline

    config = load_config(args.config)
    pipeline.run_pipeline(
        input_path=args.input,
        output_dir=extract_out,
        tenant_id=args.tenant,
        batch_id=args.batch,
        config=config,
        stage5_input=None,
    )

    # Find the policies JSONL output
    jsonl_files = [f for f in os.listdir(extract_out) if f.endswith(".jsonl")]
    if not jsonl_files:
        logger.error("No .jsonl output found in %s/", extract_out)
        sys.exit(1)
    policies_path = os.path.join(extract_out, jsonl_files[0])
    logger.info("Extracted policies: %s", policies_path)

    # Stage 2: Validate & Compile
    logger.info("=== Stage 2: Validation & Bundle Compilation ===")
    from Validation.bundle_compiler import compile_from_policies, write_bundle

    policies = []
    with open(policies_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                policies.append(json.loads(line))

    if not policies:
        logger.error("No policies extracted.")
        sys.exit(1)

    bundle_data = compile_from_policies(policies)
    bundle_path = args.bundle_out or os.path.join(extract_out, "compiled_policy_bundle.json")
    write_bundle(bundle_data, bundle_path)
    meta = bundle_data["bundle_metadata"]
    logger.info(
        "Bundle: %d rules, %d constraints, %d paths",
        meta["policy_count"], meta["constraint_count"], meta["path_count"],
    )

    # Stage 3: Enforce
    logger.info("=== Stage 3: Enforcement ===")
    from Enforcement import load_bundle, enforce
    from Enforcement.orchestrator import EnforcementConfig
    from Enforcement.audit import AuditLogger
    from Extractor.src.llm.client import LLMClient

    bundle, index = load_bundle(bundle_path)

    llm = LLMClient(
        provider=args.provider,
        model_id=args.model,
        temperature=0.0,
        max_tokens=2048,
    )

    enforce_config = EnforcementConfig(
        judge_enabled=not args.no_judge,
        smt_enabled=not args.no_smt,
    )

    audit = AuditLogger(args.audit_log) if args.audit_log else None

    decision = enforce(
        query=args.query,
        bundle=bundle,
        bundle_index=index,
        llm_client=llm,
        config=enforce_config,
        audit_logger=audit,
    )

    print(json.dumps(decision.model_dump(), indent=2, default=str))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="PolicyLLM",
        description="End-to-end policy extraction, validation, and enforcement pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract policies from documents")
    p_extract.add_argument("input", help="Path to input document or directory")
    p_extract.add_argument("--out", default="out", help="Output directory")
    p_extract.add_argument("--config", default="Extractor/configs/config.example.yaml", help="Extractor YAML config")
    p_extract.add_argument("--tenant", default="tenant_default", help="Tenant identifier")
    p_extract.add_argument("--batch", default="batch_default", help="Batch identifier")
    p_extract.set_defaults(func=cmd_extract)

    # --- validate ---
    p_validate = subparsers.add_parser("validate", help="Compile policies into enforcement bundle")
    p_validate.add_argument("input", help="Path to policies JSONL or JSON file")
    p_validate.add_argument("--out", default="compiled_policy_bundle.json", help="Output bundle path")
    p_validate.set_defaults(func=cmd_validate)

    # --- enforce ---
    p_enforce = subparsers.add_parser("enforce", help="Enforce policies against a query")
    p_enforce.add_argument("--bundle", required=True, help="Path to compiled_policy_bundle.json")
    p_enforce.add_argument("--query", required=True, help="User query to enforce")
    p_enforce.add_argument("--provider", default="stub", help="LLM provider (stub|ollama|bedrock_claude|chatgpt|anthropic)")
    p_enforce.add_argument("--model", default="mistral:latest", help="LLM model ID")
    p_enforce.add_argument("--judge-model", default=None, help="Judge LLM model ID (defaults to --model)")
    p_enforce.add_argument("--response", default=None, help="Pre-generated response to verify (skip generation)")
    p_enforce.add_argument("--audit-log", default="audit/enforcement.jsonl", help="Audit log path")
    p_enforce.add_argument("--no-judge", action="store_true", help="Disable judge LLM check")
    p_enforce.add_argument("--no-smt", action="store_true", help="Disable SMT check")
    p_enforce.set_defaults(func=cmd_enforce)

    # --- run (full pipeline) ---
    p_run = subparsers.add_parser("run", help="Full pipeline: extract -> validate -> enforce")
    p_run.add_argument("input", help="Path to input document")
    p_run.add_argument("--query", required=True, help="User query to enforce against extracted policies")
    p_run.add_argument("--out", default="out", help="Output directory for extraction artifacts")
    p_run.add_argument("--bundle-out", default=None, help="Output path for compiled bundle (default: <out>/compiled_policy_bundle.json)")
    p_run.add_argument("--config", default="Extractor/configs/config.example.yaml", help="Extractor YAML config")
    p_run.add_argument("--tenant", default="tenant_default", help="Tenant identifier")
    p_run.add_argument("--batch", default="batch_default", help="Batch identifier")
    p_run.add_argument("--provider", default="ollama", help="LLM provider")
    p_run.add_argument("--model", default="mistral:latest", help="LLM model ID")
    p_run.add_argument("--audit-log", default="audit/enforcement.jsonl", help="Audit log path")
    p_run.add_argument("--no-judge", action="store_true", help="Disable judge LLM check")
    p_run.add_argument("--no-smt", action="store_true", help="Disable SMT check")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
