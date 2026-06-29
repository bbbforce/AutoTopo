"""失败模式定义与运行失败诊断。"""

from __future__ import annotations

from dataclasses import dataclass

from autotopo.schemas import ExecutionReport, FailureDiagnosis, FailureMode, Severity


@dataclass(frozen=True)
class FailureModeInfo:
    """失败模式元信息。"""

    mode: FailureMode
    severity: Severity
    detection_rule: str
    likely_causes: tuple[str, ...]
    repair_suggestions: tuple[str, ...]
    auto_repair_allowed: bool


FAILURE_MODE_REGISTRY: dict[FailureMode, FailureModeInfo] = {
    FailureMode.PYTHON_EXCEPTION: FailureModeInfo(
        FailureMode.PYTHON_EXCEPTION,
        Severity.MODERATE,
        "Executor 捕获 Python 异常。",
        ("代码路径或输入参数触发异常",),
        ("读取 traceback 并按具体异常修复",),
        True,
    ),
    FailureMode.MISSING_DEPENDENCY: FailureModeInfo(
        FailureMode.MISSING_DEPENDENCY,
        Severity.SEVERE,
        "异常文本包含 ImportError 或 ModuleNotFoundError。",
        ("运行环境缺少依赖",),
        ("保持 optional import lazy，或在环境中安装缺失依赖",),
        False,
    ),
    FailureMode.SHAPE_MISMATCH: FailureModeInfo(
        FailureMode.SHAPE_MISMATCH,
        Severity.MODERATE,
        "异常文本包含 shape、dimension 或 broadcast。",
        ("网格尺寸与密度/矩阵尺寸不一致",),
        ("重建网格相关矩阵并校验 passive mask 尺寸",),
        True,
    ),
    FailureMode.SINGULAR_STIFFNESS_MATRIX: FailureModeInfo(
        FailureMode.SINGULAR_STIFFNESS_MATRIX,
        Severity.SEVERE,
        "异常文本包含 singular 或 matrix factorization failure。",
        ("支撑不足", "密度坍塌", "载荷路径无效"),
        ("检查边界条件和载荷，增加最小刚度保护",),
        True,
    ),
    FailureMode.INVALID_BOUNDARY_CONDITION: FailureModeInfo(
        FailureMode.INVALID_BOUNDARY_CONDITION,
        Severity.SEVERE,
        "边界条件位置无法映射到 DOF。",
        ("位置字符串不在支持范围内",),
        ("改用 left_edge、bottom_right 等模板位置",),
        True,
    ),
}


def diagnose_execution_report(report: ExecutionReport) -> FailureDiagnosis:
    """从 ExecutionReport 生成失败诊断。"""

    if report.success:
        return FailureDiagnosis(case_id=report.case_id, has_failure=False)

    text = " ".join([
        report.error_type or "",
        report.exception or "",
        report.traceback or "",
    ]).lower()

    modes: list[FailureMode] = []
    if "modulenotfounderror" in text or "importerror" in text:
        modes.append(FailureMode.MISSING_DEPENDENCY)
    if "shape" in text or "dimension" in text or "broadcast" in text:
        modes.append(FailureMode.SHAPE_MISMATCH)
    if "singular" in text or "factor" in text:
        modes.append(FailureMode.SINGULAR_STIFFNESS_MATRIX)
    if "boundary" in text or "dof" in text:
        modes.append(FailureMode.INVALID_BOUNDARY_CONDITION)
    if not modes:
        modes.append(FailureMode.PYTHON_EXCEPTION)

    infos = [FAILURE_MODE_REGISTRY.get(mode) for mode in modes]
    severity = Severity.SEVERE if any(info and info.severity == Severity.SEVERE for info in infos) else Severity.MODERATE
    causes: list[str] = []
    suggestions: list[str] = []
    auto_repair_allowed = True
    for info in infos:
        if info is None:
            continue
        causes.extend(info.likely_causes)
        suggestions.extend(info.repair_suggestions)
        auto_repair_allowed = auto_repair_allowed and info.auto_repair_allowed

    return FailureDiagnosis(
        case_id=report.case_id,
        has_failure=True,
        failure_modes=modes,
        severity=severity,
        likely_causes=list(dict.fromkeys(causes)),
        repair_suggestions=list(dict.fromkeys(suggestions)),
        auto_repair_allowed=auto_repair_allowed,
    )

