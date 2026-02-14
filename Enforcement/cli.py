"""CLI entry point for the Enforcement module."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Run policy enforcement pipeline")
    parser.add_argument("--bundle", required=True, help="Path to compiled_policy_bundle.json")
    parser.add_argument("--query", required=True, help="User query to enforce")
    parser.add_argument("--provider", default="stub", help="LLM provider (stub|ollama|bedrock_claude|chatgpt|anthropic)")
    parser.add_argument("--model", default="mistral:latest", help="LLM model ID")
    parser.add_argument("--judge-model", default=None, help="Judge LLM model ID (defaults to --model)")
    parser.add_argument("--response", default=None, help="Pre-generated response to verify (skip generation)")
    parser.add_argument("--audit-log", default="audit/enforcement.jsonl", help="Audit log path")
    parser.add_argument("--no-judge", action="store_true", help="Disable judge LLM check")
    parser.add_argument("--no-smt", action="store_true", help="Disable SMT check")
    args = parser.parse_args()

    from .bundle_loader import load_bundle
    from .orchestrator import EnforcementConfig, enforce
    from .audit import AuditLogger

    # Load bundle
    bundle, index = load_bundle(args.bundle)

    # Initialize LLM client
    sys.path.insert(0, ".")
    from Extractor.src.llm.client import LLMClient

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

    audit = AuditLogger(args.audit_log)

    # If a pre-generated response is provided, skip generation
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


if __name__ == "__main__":
    main()
