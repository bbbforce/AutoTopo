"""CaseSpec-aware RAG and corrective failure RAG."""

from __future__ import annotations

from autotopo.diagnostics.failure_modes import diagnose_execution_report
from autotopo.diagnostics.repair_rules import build_repair_plan
from autotopo.rag.query_builder import (
    build_codegen_query,
    build_critic_failure_query,
    build_execution_failure_query,
    build_validation_query,
    render_query,
)
from autotopo.rag.reranker import DomainReranker
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import (
    CaseSpec,
    ExecutionReport,
    EvaluatorReport,
    FailureDiagnosis,
    RepairPlan,
    RetrievedEvidence,
    ValidationReport,
)


class ExecutionCorrectiveRAG:
    """Retrieve evidence, diagnosis, and bounded plans for execution failures."""

    def __init__(
        self,
        retriever: LocalRetriever | None = None,
        reranker: DomainReranker | None = None,
    ) -> None:
        self.retriever = retriever or LocalRetriever()
        self.reranker = reranker or DomainReranker()

    def retrieve(
        self,
        report: ExecutionReport,
        case_spec: CaseSpec | None = None,
        *,
        limit: int = 6,
    ) -> list[RetrievedEvidence]:
        context = build_execution_failure_query(report, case_spec)
        evidence = self.retriever.retrieve(
            render_query(context),
            kinds=["failure", "solver_api", "physics_rule"],
            limit=max(limit * 2, limit),
            context=context,
        )
        return self.reranker.rerank(evidence, context, limit=limit)

    def diagnose_and_plan(
        self,
        report: ExecutionReport,
        case_spec: CaseSpec,
        *,
        repair_iteration: int,
        max_repair_rounds: int,
    ) -> tuple[list[RetrievedEvidence], FailureDiagnosis, RepairPlan]:
        evidence = self.retrieve(report, case_spec)
        evidence_ids = [item.evidence_id for item in evidence]
        diagnosis = diagnose_execution_report(report).model_copy(update={"evidence_ids": evidence_ids})
        repair_plan = build_repair_plan(
            case_spec,
            diagnosis.failure_modes,
            repair_iteration=repair_iteration,
            max_repair_rounds=max_repair_rounds,
            evidence_ids=evidence_ids,
        )
        return evidence, diagnosis, repair_plan


class CriticCorrectiveRAG:
    """Retrieve evidence and bounded repair plans for topology quality failures."""

    def __init__(
        self,
        retriever: LocalRetriever | None = None,
        reranker: DomainReranker | None = None,
    ) -> None:
        self.retriever = retriever or LocalRetriever()
        self.reranker = reranker or DomainReranker()

    def retrieve(
        self,
        report: EvaluatorReport,
        case_spec: CaseSpec | None = None,
        *,
        limit: int = 6,
    ) -> list[RetrievedEvidence]:
        context = build_critic_failure_query(report, case_spec)
        evidence = self.retriever.retrieve(
            render_query(context),
            kinds=["failure", "physics_rule"],
            limit=max(limit * 2, limit),
            context=context,
        )
        return self.reranker.rerank(evidence, context, limit=limit)

    def plan(
        self,
        report: EvaluatorReport,
        case_spec: CaseSpec,
        *,
        repair_iteration: int,
        max_repair_rounds: int,
    ) -> tuple[list[RetrievedEvidence], RepairPlan]:
        evidence = self.retrieve(report, case_spec)
        repair_plan = build_repair_plan(
            case_spec,
            report.failure_modes,
            repair_iteration=repair_iteration,
            max_repair_rounds=max_repair_rounds,
            evidence_ids=[item.evidence_id for item in evidence],
        )
        return evidence, repair_plan


def retrieve_for_codegen(case_spec: CaseSpec, retriever: LocalRetriever | None = None) -> list[RetrievedEvidence]:
    """生成代码前检索模板、求解器 API 和物理规则。"""

    retriever = retriever or LocalRetriever()
    context = build_codegen_query(case_spec)
    evidence = retriever.retrieve(
        render_query(context),
        kinds=["case_template", "solver_api", "physics_rule"],
        limit=16,
        context=context,
    )
    return DomainReranker().rerank(evidence, context, limit=8)


def retrieve_for_execution_failure(
    report: ExecutionReport,
    retriever: LocalRetriever | None = None,
    case_spec: CaseSpec | None = None,
) -> list[RetrievedEvidence]:
    """运行失败后检索 failure_kb、solver API 和修复规则。"""

    return ExecutionCorrectiveRAG(retriever).retrieve(report, case_spec)


def retrieve_for_quality_failure(
    evaluator_report: EvaluatorReport,
    retriever: LocalRetriever | None = None,
    case_spec: CaseSpec | None = None,
) -> list[RetrievedEvidence]:
    """结果质量失败后检索拓扑质量与物理规则。"""

    return CriticCorrectiveRAG(retriever).retrieve(evaluator_report, case_spec)


def retrieve_for_validation_failure(
    validation_report: ValidationReport,
    case_spec: CaseSpec,
    retriever: LocalRetriever | None = None,
) -> list[RetrievedEvidence]:
    """验证失败后检索物理规则解释，求解仍保持 fail-closed。"""

    retriever = retriever or LocalRetriever()
    context = build_validation_query(validation_report, case_spec)
    evidence = retriever.retrieve(
        render_query(context),
        kinds=["physics_rule", "failure"],
        limit=12,
        context=context,
    )
    return DomainReranker().rerank(evidence, context, limit=6)
