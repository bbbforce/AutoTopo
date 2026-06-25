"""Domain reranker feature tests."""

from __future__ import annotations

from autotopo.rag.reranker import DomainReranker
from autotopo.schemas import QueryContext, RetrievedEvidence


def _evidence(evidence_id: str, *, kind: str, source: str, content: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk_id=evidence_id,
        parent_id=evidence_id,
        kind=kind,
        source=source,
        heading=evidence_id,
        content=content,
        score=0.01,
        final_score=0.01,
    )


def test_mbb_query_prioritizes_mbb_template():
    evidence = [
        _evidence("cant", kind="case_template", source="case_template_kb/cantilever.md", content="Benchmark type: cantilever"),
        _evidence("mbb", kind="case_template", source="case_template_kb/mbb.md", content="Benchmark type: mbb"),
    ]

    ranked = DomainReranker().rerank(
        evidence,
        QueryContext(task_type="code_generation", benchmark_type="mbb", optimizer="MMA"),
    )

    assert ranked[0].evidence_id == "mbb"


def test_mma_query_does_not_prioritize_oc_template():
    evidence = [
        _evidence("oc", kind="case_template", source="case_template_kb/oc.md", content="OC update template"),
        _evidence("mma", kind="case_template", source="case_template_kb/mma.md", content="MMA update template"),
    ]

    ranked = DomainReranker().rerank(
        evidence,
        QueryContext(task_type="code_generation", benchmark_type="mbb", optimizer="MMA"),
    )

    assert ranked[0].evidence_id == "mma"


def test_execution_failure_prefers_failure_kb():
    evidence = [
        _evidence("template", kind="case_template", source="case_template_kb/mbb.md", content="MBB template"),
        _evidence("failure", kind="failure", source="failure_kb/execution.md", content="singular_stiffness_matrix repair"),
    ]

    ranked = DomainReranker().rerank(
        evidence,
        QueryContext(task_type="execution_repair", error_text="singular stiffness matrix"),
    )

    assert ranked[0].evidence_id == "failure"


def test_critic_repair_prefers_physics_or_failure_kb():
    evidence = [
        _evidence("api", kind="solver_api", source="solver_api_kb/python.md", content="API reference"),
        _evidence("physics", kind="physics_rule", source="physics_rule_kb/default.md", content="checkerboard repair increases rmin"),
        _evidence("failure", kind="failure", source="failure_kb/topology.md", content="checkerboard failure"),
    ]

    ranked = DomainReranker().rerank(
        evidence,
        QueryContext(task_type="critic_repair", failure_modes=["checkerboard"]),
    )

    assert ranked[0].kind in {"physics_rule", "failure"}
