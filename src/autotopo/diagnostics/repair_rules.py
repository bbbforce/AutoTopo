"""有界自动修复规则。"""

from __future__ import annotations

from autotopo.schemas import CaseSpec, FailureMode, RepairPlan


def build_repair_plan(
    case_spec: CaseSpec,
    failure_modes: list[FailureMode],
    *,
    repair_iteration: int,
    max_repair_rounds: int = 3,
    evidence_ids: list[str] | None = None,
) -> RepairPlan:
    """根据失败模式生成保守参数修复。"""

    if repair_iteration >= max_repair_rounds:
        return RepairPlan(
            case_id=case_spec.case_id,
            should_repair=False,
            repair_iteration=repair_iteration,
            max_repair_rounds=max_repair_rounds,
            repair_type="fail_closed",
            rationale="达到最大修复轮数。",
            reason="达到最大修复轮数。",
            failure_modes=failure_modes,
            evidence_ids=evidence_ids or [],
            auto_apply_allowed=False,
            risk_level="high",
        )

    updates: dict[str, float | int] = {}
    reasons: list[str] = []
    mode_set = set(failure_modes)

    if FailureMode.INVALID_FILTER_RADIUS in mode_set or FailureMode.CHECKERBOARD in mode_set:
        updates["rmin"] = round(min(max(case_spec.rmin * 1.35, 1.5), 4.0), 4)
        reasons.append("增大过滤半径以抑制棋盘格或无效过滤设置。")
    if FailureMode.GRAYNESS_TOO_HIGH in mode_set or FailureMode.INVALID_PENAL in mode_set:
        updates["penal"] = round(min(max(case_spec.penal * 1.25, 3.0), 6.0), 4)
        updates["max_iter"] = int(min(max(case_spec.max_iter + 5, case_spec.max_iter * 2), 80))
        reasons.append("增大 SIMP 罚因子以降低灰度单元。")
    if FailureMode.NON_CONVERGENCE in mode_set or FailureMode.MMA_OSCILLATION in mode_set:
        updates["max_iter"] = int(min(max(case_spec.max_iter + 5, case_spec.max_iter * 2), 80))
        reasons.append("增加迭代上限以改善收敛。")
    if FailureMode.VOLUME_CONSTRAINT_VIOLATION in mode_set:
        updates["rmin"] = round(max(updates.get("rmin", case_spec.rmin), case_spec.rmin), 4)
        reasons.append("保持体积分数约束不变，仅重新投影设计变量。")
    if FailureMode.DISCONNECTED_ISLANDS in mode_set or FailureMode.INVALID_LOAD_PATH in mode_set:
        updates["rmin"] = round(min(max(updates.get("rmin", case_spec.rmin * 1.25), 1.5), 4.0), 4)
        reasons.append("增大过滤半径以鼓励连续载荷路径。")

    reason = " ".join(reasons) if reasons else "没有可自动修复的安全参数。"
    old_values = {key: getattr(case_spec, key) for key in updates}
    return RepairPlan(
        case_id=case_spec.case_id,
        should_repair=bool(updates),
        repair_iteration=repair_iteration,
        max_repair_rounds=max_repair_rounds,
        repair_type="parameter_patch" if updates else "fail_closed",
        target=", ".join(updates.keys()),
        old_value=old_values or None,
        new_value=updates or None,
        rationale=reason,
        parameter_updates=updates,
        reason=reason,
        failure_modes=failure_modes,
        evidence_ids=evidence_ids or [],
        auto_repair_allowed=bool(updates),
        auto_apply_allowed=bool(updates),
        risk_level="low" if updates else "medium",
    )


def apply_repair_plan(case_spec: CaseSpec, repair_plan: RepairPlan) -> CaseSpec:
    """只应用白名单参数更新。"""

    allowed = {"penal", "rmin", "max_iter", "tol"}
    updates = {
        key: value
        for key, value in repair_plan.parameter_updates.items()
        if key in allowed
    }
    if not updates:
        return case_spec
    return case_spec.model_copy(update=updates)
