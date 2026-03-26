from __future__ import annotations

from pathlib import Path

from eval.external_benchmarks import utils


def test_build_or_load_split_deterministic_and_disjoint(tmp_path: Path):
    examples = []
    for i in range(60):
        label = ["entailment", "contradiction", "not_mentioned"][i % 3]
        examples.append({"id": f"id-{i}", "label": label})

    split_path = tmp_path / "split.json"
    first = utils.build_or_load_split(
        split_path=split_path,
        examples=examples,
        id_fn=lambda x: x["id"],
        label_fn=lambda x: x["label"],
        dev_n=10,
        test_n=30,
        seed=42,
    )
    second = utils.build_or_load_split(
        split_path=split_path,
        examples=examples,
        id_fn=lambda x: x["id"],
        label_fn=lambda x: x["label"],
        dev_n=10,
        test_n=30,
        seed=42,
    )

    assert first["dev_ids"] == second["dev_ids"]
    assert first["test_ids"] == second["test_ids"]
    assert set(first["dev_ids"]).isdisjoint(set(first["test_ids"]))


def test_calibrate_binary_threshold_is_deterministic():
    dev_rows = [
        {"label": "entailment", "score": 0.95},
        {"label": "entailment", "score": 0.91},
        {"label": "entailment", "score": 0.88},
        {"label": "non_entailment", "score": 0.30},
        {"label": "non_entailment", "score": 0.41},
        {"label": "non_entailment", "score": 0.55},
    ]

    first = utils.calibrate_binary_threshold(
        dev_rows=dev_rows,
        label_key="label",
        positive_label="entailment",
        negative_label="non_entailment",
        score_key="score",
        default_threshold=0.85,
    )
    second = utils.calibrate_binary_threshold(
        dev_rows=dev_rows,
        label_key="label",
        positive_label="entailment",
        negative_label="non_entailment",
        score_key="score",
        default_threshold=0.85,
    )

    assert first == second


def test_label_mapping_boundaries():
    assert (
        utils.map_ternary_label(
            score=0.90,
            has_violation=False,
            pos_threshold=0.85,
            neg_threshold=0.70,
            positive_label="entailment",
            negative_label="contradiction",
            neutral_label="not_mentioned",
        )
        == "entailment"
    )
    assert (
        utils.map_ternary_label(
            score=0.60,
            has_violation=True,
            pos_threshold=0.85,
            neg_threshold=0.70,
            positive_label="entailment",
            negative_label="contradiction",
            neutral_label="not_mentioned",
        )
        == "contradiction"
    )
    assert (
        utils.map_ternary_label(
            score=0.76,
            has_violation=True,
            pos_threshold=0.85,
            neg_threshold=0.70,
            positive_label="entailment",
            negative_label="contradiction",
            neutral_label="not_mentioned",
        )
        == "not_mentioned"
    )

    assert utils.map_binary_label(0.90, 0.85, "yes", "no") == "yes"
    assert utils.map_binary_label(0.84, 0.85, "yes", "no") == "no"


def test_ternary_mapping_decoupled_from_violation_gate():
    # Previously, high score + violation could be forced into neutral.
    # Score-first mapping should still produce entailment.
    assert (
        utils.map_ternary_label(
            score=0.92,
            has_violation=True,
            pos_threshold=0.85,
            neg_threshold=0.70,
            positive_label="entailment",
            negative_label="contradiction",
            neutral_label="not_mentioned",
        )
        == "entailment"
    )

    # Previously, low score + no violation could be forced into neutral.
    # Score-first mapping should still produce contradiction.
    assert (
        utils.map_ternary_label(
            score=0.61,
            has_violation=False,
            pos_threshold=0.85,
            neg_threshold=0.70,
            positive_label="entailment",
            negative_label="contradiction",
            neutral_label="not_mentioned",
        )
        == "contradiction"
    )
