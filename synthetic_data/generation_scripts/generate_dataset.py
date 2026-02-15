#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

# Load environment variables from .env if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass


def run(cmd):
    # Ensure PYTHONPATH includes current directory for local modules
    # and pass through environment variables (including AWS credentials)
    env = os.environ.copy()
    scripts_dir = Path(__file__).resolve().parent
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{scripts_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(scripts_dir)
    subprocess.run(cmd, check=True, env=env)


def write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for stages 1-4 and sensitivity mixes.")
    parser.add_argument("--root", type=Path, default=Path("synthetic_data"))
    parser.add_argument("--num-policies", type=int, default=20)
    parser.add_argument("--num-documents", type=int, default=20)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    py = sys.executable

    stage_dirs = {
        "stage1_explicit": args.root / "stage1_explicit",
        "stage2_conflicts": args.root / "stage2_conflicts",
        "stage3_implicit": args.root / "stage3_implicit",
        "stage4_mixed": args.root / "stage4_mixed",
    }

    # Shared constitution
    run([
        py,
        str(scripts / "generate_constitution.py"),
        "--num-policies",
        str(args.num_policies),
        "--seed",
        str(args.seed),
        "--out",
        str(stage_dirs["stage1_explicit"]),
    ])

    constitution_src = stage_dirs["stage1_explicit"] / "ground_truth_constitution.json"

    # Copy constitution to all stages
    for name, d in stage_dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        if name != "stage1_explicit":
            (d / "ground_truth_constitution.json").write_text(constitution_src.read_text(encoding="utf-8"), encoding="utf-8")

    # Stage 1-3 docs (using Claude via Bedrock - no flags needed)
    run([py, str(scripts / "generate_documents.py"), "--stage", "1", "--constitution", str(constitution_src), "--num-documents", str(args.num_documents), "--seed", str(args.seed), "--out", str(stage_dirs["stage1_explicit"] / "documents")])
    run([py, str(scripts / "generate_documents.py"), "--stage", "2", "--constitution", str(constitution_src), "--num-documents", str(args.num_documents), "--seed", str(args.seed + 1), "--out", str(stage_dirs["stage2_conflicts"] / "documents")])
    run([py, str(scripts / "generate_documents.py"), "--stage", "3", "--constitution", str(constitution_src), "--num-documents", str(args.num_documents), "--seed", str(args.seed + 2), "--out", str(stage_dirs["stage3_implicit"] / "documents")])

    # Stage 4 baseline
    baseline_dist = "0.5,0.15,0.25,0.1"
    run([py, str(scripts / "generate_documents.py"), "--stage", "4", "--constitution", str(constitution_src), "--num-documents", str(args.num_documents), "--seed", str(args.seed + 3), "--distribution", baseline_dist, "--out", str(stage_dirs["stage4_mixed"] / "documents")])

    # Stage 4 sensitivity mini-runs
    sensitivity = {
        "baseline": "0.5,0.15,0.25,0.1",
        "explicit_heavy": "0.65,0.1,0.15,0.1",
        "implicit_heavy": "0.35,0.15,0.4,0.1",
        "conflict_stress": "0.4,0.3,0.2,0.1",
    }
    sens_root = stage_dirs["stage4_mixed"] / "sensitivity"
    for name, dist in sensitivity.items():
        for seed_offset in range(3):
            run_dir = sens_root / name / f"seed_{args.seed + 10 + seed_offset}"
            run([
                py,
                str(scripts / "generate_documents.py"),
                "--stage",
                "4",
                "--constitution",
                str(constitution_src),
                "--num-documents",
                str(args.num_documents),
                "--seed",
                str(args.seed + 10 + seed_offset),
                "--distribution",
                dist,
                "--out",
                str(run_dir / "documents"),
            ])

    # Queries for stage 4
    run([
        py,
        str(scripts / "generate_queries.py"),
        "--constitution",
        str(constitution_src),
        "--num-queries",
        str(args.num_queries),
        "--seed",
        str(args.seed + 100),
        "--out",
        str(stage_dirs["stage4_mixed"] / "test_queries.json"),
    ])

    write_config(
        stage_dirs["stage4_mixed"] / "sensitivity" / "sensitivity_config.json",
        {
            "seed_base": args.seed,
            "num_documents_per_run": args.num_documents,
            "num_seeds": 3,
            "model": "claude-3-sonnet-20240229",
            "provider": "aws-bedrock",
            "distributions": sensitivity,
        },
    )

    print("Synthetic dataset generation complete.")


if __name__ == "__main__":
    main()
