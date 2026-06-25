"""Executor agent：执行后端并始终返回 ExecutionReport。"""

from __future__ import annotations

import contextlib
import io
import json
import traceback as tb
from pathlib import Path

from autotopo.engines.python_simp_mma_engine import PythonSimpMMAEngine
from autotopo.engines.structured_benchmarks import case_to_problem
from autotopo.schemas import CaseSpec, CodePlan, ExecutionReport


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def execute(case_spec: CaseSpec, code_plan: CodePlan, output_dir: str | Path) -> ExecutionReport:
    """执行计划，捕获 stdout/stderr/traceback。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stdout_path = out / "run_stdout.log"
    stderr_path = out / "run_stderr.log"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            engine = PythonSimpMMAEngine()
            problem = case_to_problem(case_spec)
            engine.setup(problem)
            result = engine.optimize(
                max_iter=case_spec.max_iter,
                tol=case_spec.tol,
                penal=case_spec.penal,
                rmin=case_spec.rmin,
                volfrac=case_spec.volume_fraction,
            )
            files = engine.save_outputs(out, result)

        stdout_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
        stderr_path.write_text(stderr_buffer.getvalue(), encoding="utf-8")
        compliance = result.compliance_history[-1] if result.compliance_history else None
        volume = result.volume_history[-1] if result.volume_history else None
        report = ExecutionReport(
            case_id=case_spec.case_id,
            method=code_plan.method,
            success=True,
            output_dir=str(out),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            optimizer="MMA",
            optimizer_fallback=None,
            iterations=result.iterations,
            converged=result.converged,
            compliance=compliance,
            volume_fraction=volume,
            files=files,
            metrics={"mesh_info": result.mesh_info},
        )
    except Exception as exc:  # noqa: BLE001 - Executor 必须吞掉异常并返回报告
        stdout_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
        stderr_path.write_text(stderr_buffer.getvalue(), encoding="utf-8")
        report = ExecutionReport(
            case_id=case_spec.case_id,
            method=code_plan.method,
            success=False,
            output_dir=str(out),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error_type=type(exc).__name__,
            exception=str(exc),
            traceback=tb.format_exc(),
            optimizer="MMA",
        )

    _write_json(out / "execution_report.json", report.model_dump(mode="json"))
    return report

