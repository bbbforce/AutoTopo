"""Evaluator agent：执行成功后的结果质量评估。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from autotopo.diagnostics.repair_rules import build_repair_plan
from autotopo.diagnostics.topology_metrics import checkerboard_score, connectivity_score, grayness_index, volume_error
from autotopo.rag.corrective_rag import retrieve_for_quality_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import CaseSpec, EvaluatorReport, ExecutionReport, FailureMode, RepairPlan, RetrievedEvidence


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
) -> tuple[EvaluatorReport, RepairPlan | None, list[RetrievedEvidence]]:
    """计算拓扑指标并在失败时给出修复计划。"""

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
        repair_plan = build_repair_plan(
            case_spec,
            [mode for mode in report.failure_modes if mode != FailureMode.NON_CONVERGENCE],
            repair_iteration=repair_iteration,
            max_repair_rounds=max_repair_rounds,
            evidence_ids=[item.evidence_id for item in evidence],
        )
        report = report.model_copy(update={"repair_plan": repair_plan})
    return report, repair_plan, evidence
