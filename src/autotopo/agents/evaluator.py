"""Evaluator agent：执行成功后的结果质量评估。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.agents.llm_utils import (
    AgentTrace,
    decision_allows_override,
    decision_trace_payload,
    enrich_latest_trace,
    try_invoke_structured,
)
from autotopo.diagnostics.repair_rules import build_repair_plan
from autotopo.diagnostics.topology_metrics import checkerboard_score, connectivity_score, grayness_index, volume_error
from autotopo.rag.corrective_rag import retrieve_for_quality_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import AgentDecision, CaseSpec, EvaluatorReport, ExecutionReport, FailureMode, RepairPlan, RetrievedEvidence


EVALUATOR_SYSTEM_PROMPT = """\
你是 AutoTopo research_graph 的 Evaluator agent。

你会看到本地拓扑质量指标、ExecutionReport、CaseSpec 和 RAG 证据。你的职责是判断
本地质量告警是否可能是保守阈值导致的误判。只有证据充分时，才返回 pass/allow/approve。
"""


OVERRIDABLE_QUALITY_MODES = {
    FailureMode.NON_CONVERGENCE,
    FailureMode.VOLUME_CONSTRAINT_VIOLATION,
    FailureMode.GRAYNESS_TOO_HIGH,
    FailureMode.CHECKERBOARD,
    FailureMode.DISCONNECTED_ISLANDS,
    FailureMode.TOO_THIN_MEMBERS,
}


def _passive_mask(case_spec: CaseSpec) -> np.ndarray | None:
    raw = case_spec.problem.get("domain", {}).get("passive_void_mask")
    if raw is None:
        return None
    return np.asarray(raw, dtype=bool)


def evaluate_execution(
    case_spec: CaseSpec,
    execution_report: ExecutionReport,
    *,
    repair_iteration: int = 0,
    max_repair_rounds: int = 3,
    retriever: LocalRetriever | None = None,
    use_llm: bool = False,
    allow_llm_override: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> tuple[EvaluatorReport, RepairPlan | None, list[RetrievedEvidence]]:
    """计算拓扑指标；高自治模式可由 LLM 覆盖质量告警。"""

    evidence: list[RetrievedEvidence] = []
    failure_modes: list[FailureMode] = []
    messages: list[str] = []
    density_path = execution_report.files.get("density", "")
    density = np.load(density_path) if density_path and Path(density_path).exists() else None
    passive = _passive_mask(case_spec)

    if density is None:
        failure_modes.append(FailureMode.PYTHON_EXCEPTION)
        messages.append("缺少 density.npy，无法评估。")
        report = EvaluatorReport(case_id=case_spec.case_id, success=False, has_quality_failure=True, failure_modes=failure_modes, messages=messages)
    else:
        compliance = execution_report.compliance
        vol_err = volume_error(density, case_spec.volume_fraction, passive)
        gray = grayness_index(density, passive)
        checker = checkerboard_score(density, passive)
        conn = connectivity_score(density, passive)

        objective_nan_or_inf = compliance is None or not math.isfinite(float(compliance))
        active_density = density[~passive] if passive is not None else density
        density_collapse = bool(active_density.size and (float(np.mean(active_density)) < 0.02 or float(np.mean(active_density)) > 0.98))

        if objective_nan_or_inf:
            failure_modes.append(FailureMode.COMPLIANCE_NAN_OR_INF)
            messages.append("柔度不是有限数。")
        if density_collapse:
            failure_modes.append(FailureMode.DENSITY_COLLAPSE)
            messages.append("设计变量接近全空或全实，疑似密度坍缩。")
        if vol_err > 0.08:
            failure_modes.append(FailureMode.VOLUME_CONSTRAINT_VIOLATION)
            messages.append(f"体积分数误差过大: {vol_err:.3f}")
        if gray > 0.93:
            failure_modes.append(FailureMode.GRAYNESS_TOO_HIGH)
            messages.append(f"灰度指标过高: {gray:.3f}")
        if checker > 0.45:
            failure_modes.append(FailureMode.CHECKERBOARD)
            messages.append(f"棋盘格指标过高: {checker:.3f}")
        if conn < 0.25:
            failure_modes.append(FailureMode.DISCONNECTED_ISLANDS)
            messages.append(f"实体连通性过低: {conn:.3f}")
        if not execution_report.converged:
            failure_modes.append(FailureMode.NON_CONVERGENCE)
            messages.append("未在当前迭代上限内达到 tol，作为收敛警告记录。")

        quality_failures = [mode for mode in failure_modes if mode != FailureMode.NON_CONVERGENCE]
        report = EvaluatorReport(
            case_id=case_spec.case_id,
            success=not quality_failures,
            has_quality_failure=bool(quality_failures),
            local_has_quality_failure=bool(quality_failures),
            failure_modes=failure_modes,
            compliance=compliance,
            volume_error=vol_err,
            grayness_index=gray,
            checkerboard_score=checker,
            connectivity_score=conn,
            objective_nan_or_inf=objective_nan_or_inf,
            density_collapse=density_collapse,
            converged=execution_report.converged,
            messages=messages or ["拓扑质量检查通过。"],
        )

    repair_plan = None
    if report.has_quality_failure:
        evidence = retrieve_for_quality_failure(report, retriever, case_spec)
        evidence_ids = [item.evidence_id for item in evidence]
        report = report.model_copy(update={"evidence_ids": evidence_ids})
        if use_llm and allow_llm_override:
            messages_for_llm = [
                SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "请判断本地拓扑质量告警是否可以高置信放行。\n"
                        f"可放行 failure_modes: {[mode.value for mode in OVERRIDABLE_QUALITY_MODES]}\n"
                        f"case_spec: {case_spec.model_dump(mode='json')}\n"
                        f"execution_report: {execution_report.model_dump(mode='json')}\n"
                        f"local_evaluator_report: {report.model_dump(mode='json')}\n"
                        f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                        "若放行，请把被覆盖的 failure_modes 写入 overridden_failure_modes。"
                    )
                ),
            ]
            decision = try_invoke_structured(
                agent="evaluator",
                messages=messages_for_llm,
                output_model=AgentDecision,
                provider=llm_provider,
                llm=llm,
                use_llm=use_llm,
                llm_overrides=llm_overrides,
                trace=trace,
            )
            if isinstance(decision, AgentDecision):
                decision = decision.model_copy(
                    update={
                        "case_id": case_spec.case_id,
                        "target_agent": "evaluator",
                        "evidence_ids": [item for item in decision.evidence_ids if item in set(evidence_ids)] or evidence_ids,
                    }
                )
                quality_modes = [mode for mode in report.failure_modes if mode != FailureMode.NON_CONVERGENCE]
                allowed, overridden, _remaining = decision_allows_override(
                    decision,
                    failure_modes=quality_modes,
                    allowed_modes=OVERRIDABLE_QUALITY_MODES,
                )
                enrich_latest_trace(
                    trace,
                    agent="evaluator",
                    **decision_trace_payload(decision, overridden=overridden, evidence_ids=decision.evidence_ids),
                )
                if allowed:
                    messages = list(report.messages)
                    messages.append("LLM 高置信放行本地拓扑质量告警。")
                    report = report.model_copy(
                        update={
                            "success": True,
                            "has_quality_failure": False,
                            "messages": messages,
                            "llm_decision": decision,
                            "overridden_failure_modes": overridden,
                        }
                    )
                    return report, None, evidence
                report = report.model_copy(update={"llm_decision": decision, "overridden_failure_modes": overridden})
        repair_plan = build_repair_plan(
            case_spec,
            [mode for mode in report.failure_modes if mode != FailureMode.NON_CONVERGENCE],
            repair_iteration=repair_iteration,
            max_repair_rounds=max_repair_rounds,
            evidence_ids=evidence_ids,
        )
        report = report.model_copy(update={"repair_plan": repair_plan})
    return report, repair_plan, evidence
