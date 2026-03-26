from __future__ import annotations

import numpy as np
import pytest

from eval import baselines


def test_infer_embedding_backend_type():
    assert baselines.infer_embedding_backend_type("all-MiniLM-L6-v2") == "sentence-transformers"
    assert baselines.infer_embedding_backend_type("BAAI/bge-large-en-v1.5") == "sentence-transformers"
    assert baselines.infer_embedding_backend_type("text-embedding-3-small") == "openai"


def test_openai_embedder_fails_fast_without_api_key(monkeypatch: pytest.MonkeyPatch):
    model = "text-embedding-3-small"
    baselines._RAG_EMBEDDERS.pop(model, None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        baselines._resolve_rag_embedder(model)


def test_cosine_similarity_invariant_to_scaling():
    query = np.array([3.0, 4.0], dtype=float)
    matrix = np.array([[3.0, 4.0], [6.0, 8.0], [4.0, 3.0]], dtype=float)

    sims = baselines._cosine_similarities(query, matrix)
    assert sims[0] == pytest.approx(1.0, abs=1e-8)
    assert sims[1] == pytest.approx(1.0, abs=1e-8)
    assert sims[2] < 1.0

    sims_scaled = baselines._cosine_similarities(query * 10.0, matrix * 7.0)
    assert sims_scaled[0] == pytest.approx(sims[0], abs=1e-8)
    assert sims_scaled[1] == pytest.approx(sims[1], abs=1e-8)
    assert sims_scaled[2] == pytest.approx(sims[2], abs=1e-8)
