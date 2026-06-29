"""Hybrid retriever fallback and fusion tests."""

from __future__ import annotations

from autotopo.rag.retriever import HybridRetriever, LocalRetriever


class FakeEmbeddingModel:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "semantic" in lowered or "dense_target" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def _write_kb(root):
    kb = root / "knowledge_base" / "solver_api_kb"
    kb.mkdir(parents=True)
    (kb / "api.md").write_text(
        "# API\n\n"
        "PythonSimpMMAEngine uses MMA and rmin.\n\n"
        "dense_target uses a special parameter projection rule.\n",
        encoding="utf-8",
    )


def test_dense_disabled_falls_back_to_lexical(tmp_path):
    _write_kb(tmp_path)
    retriever = LocalRetriever(tmp_path / "knowledge_base")

    evidence = retriever.retrieve("MMA rmin", limit=2)

    assert evidence
    assert all(item.dense_score == 0.0 for item in evidence)
    assert all("dense_fallback_reason" in item.metadata for item in evidence)


def test_fake_dense_retriever_participates_in_fusion(tmp_path):
    _write_kb(tmp_path)
    retriever = HybridRetriever(tmp_path / "knowledge_base", embedding_model=FakeEmbeddingModel())

    evidence = retriever.retrieve("semantic", limit=2)

    assert evidence
    assert evidence[0].dense_score > 0
    assert "dense_target" in evidence[0].content


def test_score_breakdown_exists(tmp_path):
    _write_kb(tmp_path)
    retriever = HybridRetriever(tmp_path / "knowledge_base", embedding_model=FakeEmbeddingModel())

    evidence = retriever.retrieve("semantic MMA", limit=2)

    assert evidence
    assert all(hasattr(item, "lexical_score") for item in evidence)
    assert all(hasattr(item, "dense_score") for item in evidence)
    assert all(hasattr(item, "final_score") for item in evidence)
