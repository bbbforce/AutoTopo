"""Domain-aware reranking for AutoTopo RAG evidence."""

from __future__ import annotations

import re

from autotopo.schemas import BenchmarkType, QueryContext, RetrievedEvidence


_KIND_PRIORITIES: dict[str, dict[str, float]] = {
    "code_generation": {
        "case_template": 0.38,
        "solver_api": 0.28,
        "physics_rule": 0.16,
    },
    "execution_repair": {
        "failure": 0.42,
        "solver_api": 0.22,
        "physics_rule": 0.12,
    },
    "critic_repair": {
        "failure": 0.32,
        "physics_rule": 0.32,
    },
    "validation": {
        "physics_rule": 0.36,
        "failure": 0.28,
    },
}


def _text(item: RetrievedEvidence) -> str:
    return " ".join([item.source, item.kind, item.heading, item.content]).lower()


def _has_exact(term: str, text: str) -> bool:
    lowered = term.lower().strip()
    if not lowered:
        return False
    if re.search(r"[^a-z0-9_]", lowered):
        return lowered in text
    return re.search(rf"(?<![a-z0-9_]){re.escape(lowered)}(?![a-z0-9_])", text) is not None


class DomainReranker:
    """Rerank evidence using benchmark, backend, optimizer, and failure context."""

    def rerank(
        self,
        evidence: list[RetrievedEvidence],
        context: QueryContext,
        *,
        limit: int | None = None,
    ) -> list[RetrievedEvidence]:
        reranked: list[RetrievedEvidence] = []
        preferred = _KIND_PRIORITIES.get(context.task_type, {})
        for item in evidence:
            text = _text(item)
            features: dict[str, float] = {}

            if item.kind in preferred:
                features["kind_match"] = preferred[item.kind]
            elif preferred:
                features["noise_penalty"] = -0.12

            if context.benchmark_type and _has_exact(context.benchmark_type, text):
                features["benchmark_match"] = 0.24

            backend = (context.solver_backend or "").lower()
            if backend and _has_exact(backend, text):
                features["backend_match"] = 0.2

            optimizer = (context.optimizer or "").lower()
            if optimizer and _has_exact(optimizer, text):
                features["optimizer_match"] = 0.22
            if optimizer == "mma" and _has_exact("oc", text) and not _has_exact("mma", text):
                features["optimizer_mismatch_penalty"] = -0.35

            exact_terms = list(context.structured_terms)
            exact_terms.extend(token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)?", context.error_text or ""))
            exact_hits = sorted({term for term in exact_terms if _has_exact(term, text)})
            if exact_hits:
                features["exact_api_match"] = min(0.28, 0.05 * len(exact_hits))

            failure_hits = sorted({mode for mode in context.failure_modes if _has_exact(mode, text)})
            if failure_hits:
                features["failure_mode_match"] = min(0.34, 0.18 * len(failure_hits))

            rerank_score = round(sum(features.values()), 6)
            base_score = item.final_score if item.final_score else item.score
            final_score = round(base_score + rerank_score, 6)
            reranked.append(
                item.model_copy(
                    update={
                        "score": final_score,
                        "rerank_score": rerank_score,
                        "final_score": final_score,
                        "rerank_features": features,
                    }
                )
            )

        reranked.sort(key=lambda item: item.final_score, reverse=True)
        if limit is not None:
            return reranked[:limit]
        return reranked


def rerank_for_case(
    evidence: list[RetrievedEvidence],
    benchmark_type: BenchmarkType,
) -> list[RetrievedEvidence]:
    """Compatibility wrapper for the first deterministic RAG baseline."""

    return DomainReranker().rerank(
        evidence,
        QueryContext(
            task_type="code_generation",
            benchmark_type=benchmark_type.value,
            solver_backend="python_simp_mma",
            optimizer="MMA",
        ),
    )
