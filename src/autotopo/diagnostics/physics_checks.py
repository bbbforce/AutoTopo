"""CaseSpec 物理合理性检查。"""

from __future__ import annotations

from autotopo.schemas import BenchmarkType, CaseSpec, FailureMode, Severity, ValidationReport


def _is_nonzero_direction(direction: object) -> bool:
    try:
        return any(abs(float(value)) > 0 for value in direction)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _load_hits_fixed_dof(load: dict, bc: dict) -> bool:
    if load.get("location", "") != bc.get("location", ""):
        return False
    direction = load.get("direction", [0, 0])
    try:
        fx = float(direction[0])
        fy = float(direction[1])
    except (TypeError, ValueError, IndexError):
        return False
    bc_type = bc.get("type", "")
    if bc_type == "fixed":
        return abs(fx) > 0 or abs(fy) > 0
    if bc_type in {"fixed_x", "symmetry"}:
        return abs(fx) > 0
    if bc_type in {"fixed_y", "roller"}:
        return abs(fy) > 0
    return False


def validate_case_spec(case_spec: CaseSpec) -> ValidationReport:
    """fail-closed 验证，物理不完整时不进入求解。"""

    modes: list[FailureMode] = []
    messages: list[str] = []
    problem = case_spec.problem or {}
    bcs = problem.get("boundary_conditions", [])
    loads = problem.get("loads", [])

    if not bcs:
        modes.append(FailureMode.NO_SUPPORT)
        messages.append("缺少支撑边界条件。")
    if not loads:
        modes.append(FailureMode.NO_LOAD)
        messages.append("缺少载荷。")
    if not (0.1 <= case_spec.volume_fraction <= 0.8):
        modes.append(FailureMode.INVALID_VOLUME_FRACTION)
        messages.append("体积分数应位于工程合理范围 [0.1, 0.8]。")
    if case_spec.rmin <= 0:
        modes.append(FailureMode.INVALID_FILTER_RADIUS)
        messages.append("过滤半径必须为正。")
    if not (1.0 <= case_spec.penal <= 6.0):
        modes.append(FailureMode.INVALID_PENAL)
        messages.append("SIMP 罚因子应位于工程合理范围 [1.0, 6.0]。")
    if not isinstance(case_spec.nelx, int) or not isinstance(case_spec.nely, int) or case_spec.nelx <= 0 or case_spec.nely <= 0:
        modes.append(FailureMode.INVALID_MESH_RESOLUTION)
        messages.append("nelx/nely 必须为正整数。")

    for load in loads:
        load_loc = load.get("location", "")
        if not _is_nonzero_direction(load.get("direction", [0, 0])):
            modes.append(FailureMode.NO_LOAD)
            messages.append("载荷方向为零向量。")
        for bc in bcs:
            if _load_hits_fixed_dof(load, bc):
                modes.append(FailureMode.LOAD_ON_FIXED_DOF)
                messages.append(f"载荷位置 {load_loc} 与固定约束重合。")

    has_x = any(bc.get("type") in {"fixed", "fixed_x", "symmetry", "roller"} for bc in bcs)
    has_y = any(bc.get("type") in {"fixed", "fixed_y", "roller"} for bc in bcs)
    if bcs and (not has_x or not has_y):
        modes.append(FailureMode.RIGID_BODY_MOTION)
        messages.append("支撑不足以约束刚体运动。")

    domain = problem.get("domain", {})
    if case_spec.benchmark_type == BenchmarkType.L_SHAPE:
        has_void = bool(domain.get("passive_void_mask") or domain.get("non_design_regions"))
        if not has_void:
            modes.append(FailureMode.INVALID_LOAD_PATH)
            messages.append("L 型梁需要 passive void mask 或明确的 L 型几何定义。")

    if case_spec.benchmark_type == BenchmarkType.MBB and bcs:
        has_symmetry_support = any(
            bc.get("location") == "left_edge" and bc.get("type") in {"fixed_x", "fixed", "symmetry"}
            for bc in bcs
        )
        has_vertical_support = any(
            bc.get("type") in {"fixed_y", "fixed", "roller"} for bc in bcs
        )
        if not has_symmetry_support or not has_vertical_support:
            modes.append(FailureMode.RIGID_BODY_MOTION)
            messages.append("MBB 梁需要合理的对称/竖向支撑设置。")

    if case_spec.benchmark_type == BenchmarkType.CANTILEVER and bcs:
        has_end_fixed = any(
            bc.get("type") == "fixed" and ("left" in bc.get("location", "") or "right" in bc.get("location", ""))
            for bc in bcs
        )
        if not has_end_fixed:
            modes.append(FailureMode.RIGID_BODY_MOTION)
            messages.append("悬臂梁至少需要一端固定。")

    severity = Severity.SEVERE if modes else Severity.MINOR
    return ValidationReport(
        case_id=case_spec.case_id,
        is_valid=not modes,
        failure_modes=list(dict.fromkeys(modes)),
        severity=severity,
        messages=messages or ["验证通过。"],
        normalized_problem=problem,
    )
