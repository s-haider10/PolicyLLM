"""CLI entry point for the evaluation framework."""
import argparse
import sys

from Extractor.src.llm.client import LLMClient
from Enforcement.orchestrator import EnforcementConfig

from .scenarios import load_suite, filter_scenarios
from .runner import run_suite
from .reporter import print_summary, write_json_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="evals",
        description="Run PolicyLLM enforcement evaluations.",
    )
    parser.add_argument("--suite", required=True, help="Path to eval suite JSON file")
    parser.add_argument("--provider", default="stub", help="LLM provider (stub|ollama|chatgpt|bedrock_claude|anthropic)")
    parser.add_argument("--model", default="mistral:latest", help="LLM model ID")
    parser.add_argument("--judge-model", default=None, help="Judge LLM model ID (defaults to --model)")
    parser.add_argument("--tags", nargs="*", default=None, help="Filter scenarios by tags")
    parser.add_argument("--ids", nargs="*", default=None, help="Filter scenarios by IDs")
    parser.add_argument("--output", default=None, help="Path to write JSON report")
    parser.add_argument("--no-judge", action="store_true", help="Disable judge LLM check")
    parser.add_argument("--no-smt", action="store_true", help="Disable SMT check")

    args = parser.parse_args(argv)

    # Load and filter suite
    suite = load_suite(args.suite)
    if args.tags or args.ids:
        suite = filter_scenarios(
            suite,
            tags=set(args.tags) if args.tags else None,
            ids=set(args.ids) if args.ids else None,
        )

    if not suite.scenarios:
        print("No scenarios matched the given filters.")
        sys.exit(1)

    # Build LLM clients
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

    # Run
    result = run_suite(
        suite=suite,
        llm_client=llm,
        judge_llm_client=judge_llm,
        config=config,
        provider=args.provider,
        model=args.model,
    )

    # Report
    print_summary(result)
    if args.output:
        write_json_report(result, args.output)
        print(f"\nJSON report written to {args.output}")

    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
