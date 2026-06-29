"""Local lexical, dense, and hybrid retrievers for AutoTopo RAG."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

from autotopo.rag.kb_loader import KnowledgeChunk, load_knowledge_chunks
from autotopo.schemas import QueryContext, RetrievedEvidence


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
_PHRASE_TERMS = {
    "singular stiffness matrix",
    "rigid body motion",
    "shape mismatch",
    "missing dependency",
    "invalid boundary condition",
    "load on fixed dof",
    "volume constraint",
    "density collapse",
    "mma oscillation",
}
_EXACT_TERMS = {
    "simp",
    "mma",
    "oc",
    "slsqp",
    "mbb",
    "cantilever",
    "l_shape",
    "python_simp_mma",
    "dolfin_adjoint",
    "fenics",
    "matlab",
    "penal",
    "rmin",
    "volfrac",
    "volume_fraction",
    "checkerboard",
    "grayness",
    "singular_stiffness_matrix",
    "no_support",
    "no_load",
    "load_on_fixed_dof",
    "rigid_body_motion",
}


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str] | str):  # pragma: no cover - structural protocol
        ...


def tokenize(text: str) -> list[str]:
    """中英文混合关键词切分。"""

    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _chunk_text(chunk: KnowledgeChunk) -> str:
    return " ".join([
        chunk.source,
        chunk.kind,
        chunk.chunk_type,
        chunk.title,
        chunk.heading,
        chunk.content,
    ])


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _as_vectors(raw) -> list[list[float]]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not raw:
        return []
    if isinstance(raw[0], (int, float)):
        return [[float(value) for value in raw]]
    return [[float(value) for value in row] for row in raw]


def _to_evidence(
    chunk: KnowledgeChunk,
    *,
    lexical_score: float = 0.0,
    dense_score: float = 0.0,
    final_score: float = 0.0,
    metadata: dict | None = None,
) -> RetrievedEvidence:
    merged_metadata = {**chunk.metadata, **(metadata or {}), "chunk_type": chunk.chunk_type}
    final = round(final_score, 6)
    return RetrievedEvidence(
        evidence_id=chunk.chunk_id,
        parent_id=chunk.parent_id,
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        kind=chunk.kind,
        heading=chunk.heading,
        content=chunk.content,
        score=final,
        lexical_score=round(lexical_score, 6),
        dense_score=round(dense_score, 6),
        final_score=final,
        metadata=merged_metadata,
    )


class BaseRetriever:
    """Shared retriever interface."""

    def retrieve(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 5,
        context: QueryContext | None = None,
    ) -> list[RetrievedEvidence]:
        raise NotImplementedError


class LexicalRetriever(BaseRetriever):
    """Deterministic TF-IDF/BM25-like keyword retriever."""

    def __init__(
        self,
        root: str | Path = "knowledge_base",
        *,
        chunks: list[KnowledgeChunk] | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_knowledge_chunks(root)
        self._chunk_terms = [Counter(tokenize(_chunk_text(chunk))) for chunk in self.chunks]
        self._df: Counter[str] = Counter()
        for terms in self._chunk_terms:
            self._df.update(terms.keys())

    def _exact_boost(self, query: str, query_terms: Counter[str], chunk: KnowledgeChunk, context: QueryContext | None) -> float:
        text = _chunk_text(chunk).lower()
        boost = 0.0
        for term, qtf in query_terms.items():
            if term in _EXACT_TERMS and re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text):
                boost += 0.12 * qtf
            elif ("_" in term or term.endswith("error") or term.endswith("exception")) and term in text:
                boost += 0.08 * qtf
        lowered_query = query.lower()
        for phrase in _PHRASE_TERMS:
            if phrase in lowered_query and phrase in text:
                boost += 0.25
        if context is not None:
            if context.benchmark_type and context.benchmark_type.lower() in text:
                boost += 0.2
            if context.optimizer and context.optimizer.lower() in text:
                boost += 0.15
            if context.solver_backend and context.solver_backend.lower() in text:
                boost += 0.15
            for failure_mode in context.failure_modes:
                if failure_mode.lower() in text:
                    boost += 0.16
        return boost

    def retrieve(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 5,
        context: QueryContext | None = None,
    ) -> list[RetrievedEvidence]:
        """按关键词相关性返回证据。"""

        allowed = set(kinds or [])
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.chunks:
            return []

        total_docs = max(1, len(self.chunks))
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk, chunk_terms in zip(self.chunks, self._chunk_terms):
            if allowed and chunk.kind not in allowed:
                continue
            score = 0.0
            chunk_len = max(1, sum(chunk_terms.values()))
            for term, qtf in query_terms.items():
                tf = chunk_terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log((1 + total_docs) / (1 + self._df[term])) + 1.0
                score += qtf * idf * (tf / chunk_len) ** 0.5
            score += self._exact_boost(query, query_terms, chunk, context)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_evidence(chunk, lexical_score=score, final_score=score)
            for score, chunk in scored[:limit]
        ]


class DenseRetriever(BaseRetriever):
    """Optional in-memory dense retriever.

    It is disabled by default to avoid mandatory model downloads in tests and
    local deterministic runs.
    """

    def __init__(
        self,
        root: str | Path = "knowledge_base",
        *,
        chunks: list[KnowledgeChunk] | None = None,
        embedding_model: EmbeddingModel | None = None,
        model_name: str | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_knowledge_chunks(root)
        self.embedding_model: EmbeddingModel | None = None
        self.embeddings: list[list[float]] = []
        self.enabled = False
        self.fallback_reason = ""

        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif model_name:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self.embedding_model = SentenceTransformer(model_name)
            except Exception as exc:  # pragma: no cover - optional dependency path
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                return
        else:
            self.fallback_reason = "dense retriever disabled: no embedding model configured"
            return

        try:
            self.embeddings = self._encode([_chunk_text(chunk) for chunk in self.chunks])
            self.enabled = bool(self.embeddings)
        except Exception as exc:
            self.enabled = False
            self.embeddings = []
            self.fallback_reason = f"{type(exc).__name__}: {exc}"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_model is None:
            return []
        return _as_vectors(self.embedding_model.encode(texts))

    def retrieve(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 5,
        context: QueryContext | None = None,
    ) -> list[RetrievedEvidence]:
        del context
        if not self.enabled or self.embedding_model is None:
            return []
        allowed = set(kinds or [])
        try:
            query_vectors = self._encode([query])
        except Exception as exc:
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            return []
        if not query_vectors:
            return []

        query_vector = query_vectors[0]
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            if allowed and chunk.kind not in allowed:
                continue
            score = _cosine(query_vector, embedding)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_evidence(chunk, dense_score=score, final_score=score)
            for score, chunk in scored[:limit]
        ]


class HybridRetriever(BaseRetriever):
    """Fuse lexical and optional dense retrieval with reciprocal rank fusion."""

    def __init__(
        self,
        root: str | Path = "knowledge_base",
        *,
        chunks: list[KnowledgeChunk] | None = None,
        lexical: LexicalRetriever | None = None,
        dense: DenseRetriever | None = None,
        embedding_model: EmbeddingModel | None = None,
        dense_model_name: str | None = None,
        lexical_weight: float = 0.7,
        dense_weight: float = 0.3,
    ) -> None:
        shared_chunks = chunks if chunks is not None else load_knowledge_chunks(root)
        self.lexical = lexical or LexicalRetriever(root, chunks=shared_chunks)
        self.dense = dense or DenseRetriever(
            root,
            chunks=shared_chunks,
            embedding_model=embedding_model,
            model_name=dense_model_name,
        )
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight

    def retrieve(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 5,
        context: QueryContext | None = None,
    ) -> list[RetrievedEvidence]:
        candidate_limit = max(limit * 4, limit)
        lexical_results = self.lexical.retrieve(query, kinds=kinds, limit=candidate_limit, context=context)
        dense_results = self.dense.retrieve(query, kinds=kinds, limit=candidate_limit, context=context)

        fused: dict[str, dict] = {}

        def add(results: list[RetrievedEvidence], weight: float, score_key: str) -> None:
            for rank, item in enumerate(results, start=1):
                entry = fused.setdefault(
                    item.chunk_id,
                    {
                        "item": item,
                        "rrf": 0.0,
                        "lexical_score": 0.0,
                        "dense_score": 0.0,
                    },
                )
                entry["rrf"] += weight / (60.0 + rank)
                entry[score_key] = max(entry[score_key], getattr(item, score_key))

        add(lexical_results, self.lexical_weight, "lexical_score")
        add(dense_results, self.dense_weight, "dense_score")

        results: list[RetrievedEvidence] = []
        dense_reason = self.dense.fallback_reason if not self.dense.enabled else ""
        for entry in fused.values():
            item: RetrievedEvidence = entry["item"]
            metadata = dict(item.metadata)
            if dense_reason:
                metadata["dense_fallback_reason"] = dense_reason
            final_score = entry["rrf"]
            results.append(
                item.model_copy(
                    update={
                        "score": round(final_score, 6),
                        "lexical_score": round(entry["lexical_score"], 6),
                        "dense_score": round(entry["dense_score"], 6),
                        "final_score": round(final_score, 6),
                        "metadata": metadata,
                    }
                )
            )
        results.sort(key=lambda item: item.final_score, reverse=True)
        return results[:limit]


class LocalRetriever(HybridRetriever):
    """Backward-compatible local retriever entry point."""

    def __init__(
        self,
        root: str | Path = "knowledge_base",
        *,
        embedding_model: EmbeddingModel | None = None,
        dense_model_name: str | None = None,
        lexical_weight: float = 0.7,
        dense_weight: float = 0.3,
    ) -> None:
        super().__init__(
            root,
            embedding_model=embedding_model,
            dense_model_name=dense_model_name,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
        )
