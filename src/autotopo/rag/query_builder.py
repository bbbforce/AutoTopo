"""CaseSpec-aware retrieval query builders."""

from __future__ import annotations

from pathlib import Path

from autotopo.schemas import CaseSpec, EvaluatorReport, ExecutionReport, QueryContext, ValidationReport


def _tail_lines(text: str | None, limit: int = 20) -> str:
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def _maybe_read_summary(path: str, max_chars: int = 1200) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return path
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return path


def _case_terms(case_spec: CaseSpec) -> list[str]:
    problem = case_spec.problem or {}
    loads = problem.get("loads", [])
    bcs = problem.get("boundary_conditions", [])
    terms = [
        case_spec.case_id,
        case_spec.benchmark_type.value,
        case_spec.optimizer,
        "python_simp_mma",
        "SIMP",
        "MMA",
        "compliance",
        "volume constraint",
        f"nelx {case_spec.nelx}",
        f"nely {case_spec.nely}",
        f"volume_fraction {case_spec.volume_fraction}",
        f"penal {case_spec.penal}",
        f"rmin {case_spec.rmin}",
    ]
    for load in loads:
        terms.extend([
            str(load.get("location", "")),
            str(load.get("type", "")),
            "load",
        ])
    for bc in bcs:
        terms.extend([
            str(bc.get("location", "")),
            str(bc.get("type", "")),
            "support",
        ])
    return [term for term in terms if term]


def render_query(context: QueryContext) -> str:
    """Render a QueryContext to the plain text consumed by retrievers."""

    pieces = [
        context.task_type,
        context.case_id,
        context.benchmark_type or "",
        context.solver_backend,
        context.optimizer,
        context.natural_language,
        context.error_text,
        " ".join(context.structured_terms),
        " ".join(context.failure_modes),
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def build_codegen_query(case_spec: CaseSpec) -> QueryContext:
    return QueryContext(
        task_type="code_generation",
        benchmark_type=case_spec.benchmark_type.value,
        solver_backend="python_simp_mma",
        optimizer=case_spec.optimizer,
        case_id=case_spec.case_id,
        natural_language=case_spec.natural_language,
        structured_terms=_case_terms(case_spec),
    )


def build_execution_failure_query(
    execution_report: ExecutionReport,
    case_spec: CaseSpec | None = None,
) -> QueryContext:
    structured_terms: list[str] = []
    if case_spec is not None:
        structured_terms.extend(_case_terms(case_spec))
    structured_terms.extend([
        execution_report.error_type or "",
        execution_report.exception or "",
        execution_report.optimizer,
        execution_report.optimizer_fallback or "",
        "execution failure",
        "repair rule",
    ])
    for path in execution_report.files.values():
        structured_terms.append(Path(path).name)
    error_text = "\n".join([
        execution_report.error_type or "",
        execution_report.exception or "",
        _tail_lines(execution_report.traceback, 20),
        _maybe_read_summary(execution_report.stderr_path),
    ])
    lowered_error = error_text.lower()
    if "singular stiffness" in lowered_error or "singular matrix" in lowered_error:
        structured_terms.extend(["singular_stiffness_matrix", "no_support", "rigid_body_motion"])
    if "boundary" in lowered_error:
        structured_terms.append("invalid_boundary_condition")
    if "shape" in lowered_error and "mismatch" in lowered_error:
        structured_terms.append("shape_mismatch")
    return QueryContext(
        task_type="execution_repair",
        benchmark_type=case_spec.benchmark_type.value if case_spec is not None else None,
        solver_backend="python_simp_mma",
        optimizer=case_spec.optimizer if case_spec is not None else execution_report.optimizer,
        case_id=execution_report.case_id,
        structured_terms=[term for term in structured_terms if term],
        error_text=error_text,
    )


def build_critic_failure_query(
    critic_report: EvaluatorReport,
    case_spec: CaseSpec | None = None,
) -> QueryContext:
    failure_modes = [mode.value for mode in critic_report.failure_modes]
    metrics = [
        f"grayness_index {critic_report.grayness_index}",
        f"checkerboard_score {critic_report.checkerboard_score}",
        f"connectivity_score {critic_report.connectivity_score}",
        f"volume_error {critic_report.volume_error}",
        f"convergence_success {critic_report.converged}",
    ]
    if case_spec is not None:
        metrics.extend([
            f"penal {case_spec.penal}",
            f"rmin {case_spec.rmin}",
            f"max_iter {case_spec.max_iter}",
            f"volume_fraction {case_spec.volume_fraction}",
        ])
    return QueryContext(
        task_type="critic_repair",
        benchmark_type=case_spec.benchmark_type.value if case_spec is not None else None,
        solver_backend="python_simp_mma",
        optimizer=case_spec.optimizer if case_spec is not None else "MMA",
        case_id=critic_report.case_id,
        natural_language=" ".join(critic_report.messages),
        structured_terms=metrics + ["topology quality", "parameter repair", "penal continuation", "filter radius"],
        failure_modes=failure_modes,
    )


def build_validation_query(
    validation_report: ValidationReport,
    case_spec: CaseSpec,
) -> QueryContext:
    failure_modes = [mode.value for mode in validation_report.failure_modes]
    structured_terms = _case_terms(case_spec) + [
        "physics validation",
        "fail closed",
        "support",
        "load",
        "volume_fraction",
        "filter radius",
        "penal",
    ]
    return QueryContext(
        task_type="validation",
        benchmark_type=case_spec.benchmark_type.value,
        solver_backend="python_simp_mma",
        optimizer=case_spec.optimizer,
        case_id=case_spec.case_id,
        natural_language=" ".join(validation_report.messages),
        structured_terms=structured_terms,
        failure_modes=failure_modes,
    )
