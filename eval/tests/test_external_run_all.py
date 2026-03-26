from __future__ import annotations

import argparse

from eval.external_benchmarks import run_all


def test_run_all_aggregates_partial_failures(tmp_path, monkeypatch):
    def fake_run_step(name, cmd, cwd):
        if name == "contract_nli":
            return {"name": name, "status": "failed", "command": cmd, "return_code": 1}
        return {"name": name, "status": "ok", "command": cmd}

    def fake_load_json(path):
        p = str(path)
        if "legalbench_unfair_tos_results.json" in p:
            return {"metrics": {"policyllm": {"accuracy": 0.8, "macro_f1": 0.7}}}
        if "legalbench_privacy_policy_entailment_results.json" in p:
            return {"metrics": {"policyllm": {"accuracy": 0.6, "macro_f1": 0.55}}}
        if "cuad_results.json" in p:
            return {"metrics": {"policyllm": {"clause_type_precision": 0.4, "clause_type_recall": 0.3}}}
        return None

    monkeypatch.setattr(run_all, "_run_step", fake_run_step)
    monkeypatch.setattr(run_all, "_load_json_if_exists", fake_load_json)

    args = argparse.Namespace(
        seed=42,
        dev_examples=2,
        max_examples=3,
        num_contracts=3,
        llm_provider="stub",
        model="stub",
        config="Extractor/configs/config.stub.yaml",
        embedding_model="all-MiniLM-L6-v2",
        output_dir=str(tmp_path),
    )

    summary = run_all.run(args)

    assert len(summary["statuses"]) == 4
    assert any(s["status"] == "failed" for s in summary["statuses"])
    assert len(summary["rows"]) == 3  # ContractNLI missing, others present.
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()
