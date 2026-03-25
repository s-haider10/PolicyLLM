"""Standalone entrypoint for Z3 conflict detection in a subprocess.

Used when in-process Z3 fails (e.g. macOS architecture mismatch: x86_64 process
vs arm64 lib). The parent runs this with the native arch (e.g. arch -arm64)
so the correct libz3 is loaded.

Reads one JSON object from stdin: {"policy_ir": {...}, "decision_graph": {...}}
Writes one JSON object to stdout: the conflict_report from detect_conflicts().
"""
import json
import sys


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        sys.stderr.write(f"conflict_detector_runner: stdin JSON error: {e}\n")
        sys.exit(1)
    policy_ir = payload.get("policy_ir")
    decision_graph = payload.get("decision_graph")
    if not policy_ir or not decision_graph:
        sys.stderr.write("conflict_detector_runner: need policy_ir and decision_graph\n")
        sys.exit(1)
    try:
        from Validation.conflict_detector import detect_conflicts
        report = detect_conflicts(decision_graph, policy_ir)
    except Exception as e:
        sys.stderr.write(f"conflict_detector_runner: detect_conflicts error: {e}\n")
        sys.exit(1)
    json.dump(report, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
