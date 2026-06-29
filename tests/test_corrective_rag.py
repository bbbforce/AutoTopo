"""本地 corrective RAG 测试。"""

from __future__ import annotations

from autotopo.rag.corrective_rag import retrieve_for_codegen, retrieve_for_execution_failure, retrieve_for_quality_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.schemas import BenchmarkMethod, EvaluatorReport, ExecutionReport, FailureMode


def test_codegen_retrieval_returns_evidence():
    spec = default_case_spec("mbb", quick=True)
    evidence = retrieve_for_codegen(spec, LocalRetriever())

    assert evidence
    assert all(item.evidence_id and item.source and item.content for item in evidence)


def test_failure_retrieval_uses_failure_kb():
    report = ExecutionReport(
        case_id="case",
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        success=False,
        output_dir="/tmp/out",
        error_type="RuntimeError",
        exception="singular stiffness matrix",
    )

    evidence = retrieve_for_execution_failure(report, LocalRetriever())

    assert evidence
    assert any(item.kind == "failure" for item in evidence)
    assert any("no_support" in item.content or "rigid_body_motion" in item.content for item in evidence)


def test_quality_retrieval_uses_failure_or_physics_kb():
    report = EvaluatorReport(
        case_id="case",
        success=False,
        has_quality_failure=True,
        failure_modes=[FailureMode.GRAYNESS_TOO_HIGH, FailureMode.CHECKERBOARD],
        messages=["grayness checkerboard"],
    )

    evidence = retrieve_for_quality_failure(report, LocalRetriever())

    assert evidence
    assert {item.kind for item in evidence} <= {"failure", "physics_rule"}


def test_checkerboard_retrieval_mentions_increasing_rmin():
    report = EvaluatorReport(
        case_id="case",
        success=False,
        has_quality_failure=True,
        failure_modes=[FailureMode.CHECKERBOARD],
        messages=["checkerboard pattern"],
    )

    evidence = retrieve_for_quality_failure(report, LocalRetriever())

    assert any("rmin" in item.content or "filter radius" in item.content.lower() for item in evidence)


def test_grayness_retrieval_mentions_penal_continuation():
    report = EvaluatorReport(
        case_id="case",
        success=False,
        has_quality_failure=True,
        failure_modes=[FailureMode.GRAYNESS_TOO_HIGH],
        messages=["grayness too high"],
    )

    evidence = retrieve_for_quality_failure(report, LocalRetriever())

    assert any("penal continuation" in item.content.lower() or "projection" in item.content.lower() for item in evidence)
