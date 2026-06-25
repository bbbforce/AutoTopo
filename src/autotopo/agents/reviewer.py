"""Reviewer agent：运行失败后的诊断与修复计划。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.agents.llm_utils import AgentTrace, try_invoke_structured
from autotopo.diagnostics.failure_modes import diagnose_execution_report
from autotopo.diagnostics.repair_rules import build_repair_plan
from autotopo.rag.corrective_rag import retrieve_for_execution_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import CaseSpec, ExecutionReport, FailureDiagnosis, FailureMode, RepairPlan, RetrievedEvidence


REVIEWER_SYSTEM_PROMPT = """\
You are the Reviewer agent for AutoTopo's minimal research workflow.

Diagnose a failed PythonSimpMMAEngine execution using the provided report,
deterministic baseline diagnosis, and retrieved evidence. Return only a
FailureDiagnosis JSON object. Use failure_modes only from the allowed enum list.
Do not propose arbitrary code execution. Repair will be bounded locally.
"""


def _normalize_llm_diagnosis(
    diagnosis: FailureDiagnosis,
    *,
    case_spec: CaseSpec,
    deterministic: FailureDiagnosis,
    evidence_ids: list[str],
) -> FailureDiagnosis:
    if not diagnosis.has_failure or not diagnosis.failure_modes:
        return deterministic
    return diagnosis.model_copy(
        update={
            "case_id": case_spec.case_id,
            "has_failure": True,
            "likely_causes": diagnosis.likely_causes or deterministic.likely_causes,
            "repair_suggestions": diagnosis.repair_suggestions or deterministic.repair_suggestions,
            "evidence_ids": evidence_ids,
        }
    )


def review_execution_failure(
    case_spec: CaseSpec,
    report: ExecutionReport,
    *,
    repair_iteration: int,
    max_repair_rounds: int,
    retriever: LocalRetriever | None = None,
    use_llm: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> tuple[FailureDiagnosis, RepairPlan, list[RetrievedEvidence]]:
    """诊断执行失败并生成有界修复计划。"""

    evidence = retrieve_for_execution_failure(report, retriever, case_spec)
    evidence_ids = [item.evidence_id for item in evidence]
    deterministic = diagnose_execution_report(report).model_copy(update={"evidence_ids": evidence_ids})
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Review this failed topology optimization execution.\n"
                f"allowed_failure_modes: {[mode.value for mode in FailureMode]}\n"
                f"case_spec: {case_spec.model_dump(mode='json')}\n"
                f"execution_report: {report.model_dump(mode='json')}\n"
                f"deterministic_diagnosis: {deterministic.model_dump(mode='json')}\n"
                f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                "Return a FailureDiagnosis. Keep auto_repair_allowed conservative."
            )
        ),
    ]
    llm_diagnosis = try_invoke_structured(
        agent="reviewer",
        messages=messages,
        output_model=FailureDiagnosis,
        provider=llm_provider,
        llm=llm,
        use_llm=use_llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    diagnosis = (
        _normalize_llm_diagnosis(
            llm_diagnosis,
            case_spec=case_spec,
            deterministic=deterministic,
            evidence_ids=evidence_ids,
        )
        if isinstance(llm_diagnosis, FailureDiagnosis)
        else deterministic
    )
    repair_plan = build_repair_plan(
        case_spec,
        diagnosis.failure_modes,
        repair_iteration=repair_iteration,
        max_repair_rounds=max_repair_rounds,
        evidence_ids=diagnosis.evidence_ids,
    )
    return diagnosis, repair_plan, evidence
