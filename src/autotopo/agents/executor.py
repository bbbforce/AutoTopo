"""Executor agent：执行后端并始终返回 ExecutionReport。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import traceback as tb
from pathlib import Path

from autotopo.engines.python_simp_mma_engine import PythonSimpMMAEngine
from autotopo.engines.structured_benchmarks import case_to_problem
from autotopo.schemas import CaseSpec, CodePlan, ExecutionReport


BANNED_IMPORT_ROOTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "multiprocessing",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "threading",
    "urllib",
}
BANNED_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}
BANNED_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "spawn"),
    ("os", "execv"),
    ("os", "execve"),
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _generated_failure(
    *,
    case_spec: CaseSpec,
    code_plan: CodePlan,
    output_dir: Path,
    error_type: str,
    exception: str,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> ExecutionReport:
    report = ExecutionReport(
        case_id=case_spec.case_id,
        method=code_plan.method,
        success=False,
        output_dir=str(output_dir),
        stdout_path=str(stdout_path or ""),
        stderr_path=str(stderr_path or ""),
        error_type=error_type,
        exception=exception,
        optimizer="generated_script",
        files={
            "generated_code": code_plan.generated_code_path,
            "generated_code_manifest": code_plan.generated_code_manifest_path,
        },
        metrics={"sandbox": {"execution_mode": "generated_script"}},
    )
    _write_json(output_dir / "execution_report.json", report.model_dump(mode="json"))
    return report


def _ensure_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _static_check_generated_code(code_path: Path, output_dir: Path) -> list[str]:
    errors: list[str] = []
    if not code_path.exists():
        return [f"生成脚本不存在: {code_path}"]
    if not _ensure_inside(code_path, output_dir):
        return ["生成脚本必须位于输出目录内。"]
    try:
        tree = ast.parse(code_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"生成脚本语法错误: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    errors.append(f"禁止导入模块: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in BANNED_IMPORT_ROOTS:
                errors.append(f"禁止导入模块: {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
                errors.append(f"禁止调用: {func.id}")
            elif isinstance(func, ast.Attribute):
                owner = func.value.id if isinstance(func.value, ast.Name) else ""
                if (owner, func.attr) in BANNED_ATTR_CALLS:
                    errors.append(f"禁止调用: {owner}.{func.attr}")
    return errors


def _run_generated_script(
    case_spec: CaseSpec,
    code_plan: CodePlan,
    output_dir: Path,
    *,
    timeout_s: int,
) -> ExecutionReport:
    code_path = Path(code_plan.generated_code_path)
    stdout_path = output_dir / "generated_stdout.log"
    stderr_path = output_dir / "generated_stderr.log"
    errors = _static_check_generated_code(code_path, output_dir)
    if errors:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("\n".join(errors), encoding="utf-8")
        return _generated_failure(
            case_spec=case_spec,
            code_plan=code_plan,
            output_dir=output_dir,
            error_type="GeneratedCodeRejected",
            exception="; ".join(errors),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONNOUSERSITE": "1",
    }
    cmd = [
        sys.executable,
        str(code_path.resolve()),
        "--case-spec",
        str((output_dir / "case_spec.json").resolve()),
        "--code-plan",
        str((output_dir / "code_plan.json").resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return _generated_failure(
            case_spec=case_spec,
            code_plan=code_plan,
            output_dir=output_dir,
            error_type="GeneratedCodeTimeout",
            exception=f"生成脚本执行超过 {timeout_s} 秒。",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return _generated_failure(
            case_spec=case_spec,
            code_plan=code_plan,
            output_dir=output_dir,
            error_type="GeneratedCodeProcessError",
            exception=f"生成脚本退出码: {proc.returncode}",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    report_path = output_dir / "execution_report.json"
    if not report_path.exists():
        return _generated_failure(
            case_spec=case_spec,
            code_plan=code_plan,
            output_dir=output_dir,
            error_type="GeneratedCodeContractError",
            exception="生成脚本未写 execution_report.json。",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    try:
        report = ExecutionReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 生成脚本输出必须转为失败报告
        return _generated_failure(
            case_spec=case_spec,
            code_plan=code_plan,
            output_dir=output_dir,
            error_type="GeneratedCodeContractError",
            exception=f"execution_report.json 校验失败: {type(exc).__name__}: {exc}",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    files = dict(report.files)
    files.setdefault("generated_code", code_plan.generated_code_path)
    files.setdefault("generated_code_manifest", code_plan.generated_code_manifest_path)
    metrics = dict(report.metrics)
    metrics["sandbox"] = {
        "execution_mode": "generated_script",
        "returncode": proc.returncode,
        "timeout_s": timeout_s,
    }
    report = report.model_copy(
        update={
            "case_id": case_spec.case_id,
            "method": code_plan.method,
            "output_dir": str(output_dir),
            "stdout_path": report.stdout_path or str(stdout_path),
            "stderr_path": report.stderr_path or str(stderr_path),
            "optimizer": report.optimizer or "generated_script",
            "files": files,
            "metrics": metrics,
        }
    )
    _write_json(output_dir / "execution_report.json", report.model_dump(mode="json"))
    return report


def execute(
    case_spec: CaseSpec,
    code_plan: CodePlan,
    output_dir: str | Path,
    *,
    generated_code_timeout_s: int = 60,
) -> ExecutionReport:
    """执行计划，捕获 stdout/stderr/traceback。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if code_plan.allow_generated_code and code_plan.execution_mode == "generated_script":
        return _run_generated_script(
            case_spec,
            code_plan,
            out,
            timeout_s=generated_code_timeout_s,
        )

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
