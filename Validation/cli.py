"""CLI entry point: policies JSONL -> compiled_policy_bundle.json."""
import argparse
import json
import sys
from typing import Any, Dict, List


def _load_policies(input_path: str) -> List[Dict[str, Any]]:
    """Load policies from JSONL or JSON file."""
    policies = []
    with open(input_path, encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            policies = json.loads(content)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    policies.append(json.loads(line))
    return policies


def main():
    parser = argparse.ArgumentParser(description="Compile policies into enforcement bundle")
    parser.add_argument("input", help="Path to policies JSONL or JSON file")
    parser.add_argument("--out", default="compiled_policy_bundle.json", help="Output bundle path")
    args = parser.parse_args()

    policies = _load_policies(args.input)
    if not policies:
        print("No policies found in input file.", file=sys.stderr)
        sys.exit(1)

    from .bundle_compiler import compile_from_policies, write_bundle

    bundle = compile_from_policies(policies)
    write_bundle(bundle, args.out)
    print(f"Bundle written to {args.out} ({bundle['bundle_metadata']['policy_count']} rules, "
          f"{bundle['bundle_metadata']['constraint_count']} constraints, "
          f"{bundle['bundle_metadata']['path_count']} paths)")


if __name__ == "__main__":
    main()
