"""Optional LLM paths for the minimal research agents."""

from __future__ import annotations

import json

import numpy as np
import pytest

from autotopo.agents.coder import GeneratedSolverDraft, select_or_generate_code
from autotopo.agents.evaluator import evaluate_execution
from autotopo.agents.executor import execute
from autotopo.agents.planner import plan_code
from autotopo.agents.reviewer import review_execution_failure
from autotopo.agents.scientist import CaseSpecDraft, build_case_spec
from autotopo.agents.validator import validate
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.schemas import (
    AgentAuthority,
    AgentDecision,
    BenchmarkMethod,
    BenchmarkType,
    CodePlan,
    ExecutionReport,
    FailureDiagnosis,
    FailureMode,
)


class FakeLLM:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.result


def test_scientist_uses_llm_draft_then_rebuilds_template_problem():
    trace = []
    llm = FakeLLM(
        CaseSpecDraft(
            case_id="llm_mbb",
            benchmark_type=BenchmarkType.MBB,
            volume_fraction=0.45,
        )
    )

    spec = build_case_spec(
        "请做一个标准 MBB 梁",
        quick=True,
        use_llm=True,
        llm=llm,
        trace=trace,
    )

    assert spec.case_id == "llm_mbb"
    assert spec.benchmark_type == BenchmarkType.MBB
    assert spec.nelx == 12
    assert spec.volume_fraction == pytest.approx(0.45)
    assert spec.problem["loads"][0]["location"] == "top_left"
    assert trace == [
        {
            "agent": "scientist",
            "enabled": True,
            "used_llm": True,
            "fallback_reason": "",
        }
    ]


def test_scientist_explicit_text_parameters_override_llm_defaults():
    trace = []
    llm = FakeLLM(
        CaseSpecDraft(
            case_id="llm_mbb",
            benchmark_type=BenchmarkType.MBB,
            nelx=90,
            nely=30,
            penal=3.0,
            rmin=1.5,
        )
    )

    spec = build_case_spec(
        "标准半对称 MBB 梁拓扑优化问题。设计域尺寸为 150x50，"
        "目标是最小化柔度，体积分数约束为 0.5，惩罚因子p=1，过滤半径r=10",
        quick=False,
        use_llm=True,
        llm=llm,
        trace=trace,
    )

    assert spec.case_id == "llm_mbb"
    assert spec.nelx == 150
    assert spec.nely == 50
    assert spec.volume_fraction == pytest.approx(0.5)
    assert spec.penal == pytest.approx(1.0)
    assert spec.rmin == pytest.approx(10.0)
    assert spec.problem["domain"]["nelx"] == 150
    assert spec.problem["parameters"]["rmin"] == pytest.approx(10.0)
    assert trace[0]["used_llm"] is True


def test_scientist_falls_back_when_llm_fails():
    trace = []

    spec = build_case_spec(
        "做一个 L 型梁",
        quick=True,
        use_llm=True,
        llm=FakeLLM(error=RuntimeError("no api key")),
        trace=trace,
    )

    assert spec.benchmark_type == BenchmarkType.L_SHAPE
    assert trace[0]["agent"] == "scientist"
    assert trace[0]["used_llm"] is False
    assert "RuntimeError" in trace[0]["fallback_reason"]


def test_planner_uses_llm_steps_but_keeps_safe_execution_contract():
    case = default_case_spec("cantilever", quick=True)
    trace = []
    unsafe_llm_plan = CodePlan(
        case_id=case.case_id,
        method=BenchmarkMethod.BASELINE_DIRECT,
        engine="some_other_engine",
        template_id="custom",
        optimizer="OC",
        allow_generated_code=True,
        steps=["LLM: choose canonical SIMP/MMA template"],
        parameters={"nelx": 999},
    )

    plan = plan_code(
        case,
        BenchmarkMethod.BASELINE_DIRECT,
        use_llm=True,
        llm=FakeLLM(unsafe_llm_plan),
        trace=trace,
    )

    assert plan.engine == "python_simp_mma"
    assert plan.template_id == "cantilever"
    assert plan.optimizer == "MMA"
    assert plan.allow_generated_code is False
    assert plan.parameters["nelx"] == case.nelx
    assert plan.steps == ["LLM: choose canonical SIMP/MMA template"]
    assert trace[0]["used_llm"] is True


def test_reviewer_uses_llm_diagnosis_with_deterministic_bounded_repair(tmp_path):
    case = default_case_spec("cantilever", quick=True)
    report = ExecutionReport(
        case_id=case.case_id,
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        success=False,
        output_dir=str(tmp_path),
        exception="singular matrix",
        traceback="matrix factorization failed",
    )
    llm_diagnosis = FailureDiagnosis(
        case_id="wrong_case",
        has_failure=True,
        failure_modes=[FailureMode.SINGULAR_STIFFNESS_MATRIX],
        likely_causes=["LLM saw an unsupported load path"],
        repair_suggestions=["Use bounded local repair only"],
        auto_repair_allowed=True,
    )
    trace = []

    diagnosis, repair_plan, _ = review_execution_failure(
        case,
        report,
        repair_iteration=0,
        max_repair_rounds=2,
        use_llm=True,
        llm=FakeLLM(llm_diagnosis),
        trace=trace,
    )

    assert diagnosis.case_id == case.case_id
    assert diagnosis.failure_modes == [FailureMode.SINGULAR_STIFFNESS_MATRIX]
    assert diagnosis.likely_causes == ["LLM saw an unsupported load path"]
    assert repair_plan.case_id == case.case_id
    assert repair_plan.parameter_updates == {}
    assert trace[0]["agent"] == "reviewer"
    assert trace[0]["used_llm"] is True


def test_validator_llm_primary_can_override_physics_failure():
    case = default_case_spec("cantilever", quick=True)
    problem = dict(case.problem)
    problem["boundary_conditions"] = []
    broken = case.model_copy(update={"problem": problem})
    decision = AgentDecision(
        decision="pass",
        confidence=0.92,
        overridden_failure_modes=[FailureMode.NO_SUPPORT],
        reasons=["RAG 证据显示该生成 benchmark 可交给后续环节处理。"],
    )
    trace = []

    report = validate(
        broken,
        use_llm=True,
        allow_llm_override=True,
        llm=FakeLLM(decision),
        trace=trace,
    )

    assert report.local_is_valid is False
    assert report.is_valid is True
    assert report.overridden_failure_modes == [FailureMode.NO_SUPPORT]
    assert report.llm_decision is not None
    assert trace[0]["agent"] == "validator"
    assert trace[0]["confidence"] == pytest.approx(0.92)


def test_validator_low_confidence_llm_keeps_fail_closed():
    case = default_case_spec("cantilever", quick=True)
    problem = dict(case.problem)
    problem["boundary_conditions"] = []
    broken = case.model_copy(update={"problem": problem})
    decision = AgentDecision(
        decision="pass",
        confidence=0.4,
        overridden_failure_modes=[FailureMode.NO_SUPPORT],
    )

    report = validate(
        broken,
        use_llm=True,
        allow_llm_override=True,
        llm=FakeLLM(decision),
    )

    assert report.is_valid is False
    assert FailureMode.NO_SUPPORT in report.failure_modes


def test_coder_generated_script_executes_through_json_contract(tmp_path):
    case = default_case_spec("cantilever", quick=True)
    base_plan = CodePlan(
        case_id=case.case_id,
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        template_id=case.benchmark_type.value,
    )
    script = """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--case-spec")
parser.add_argument("--code-plan")
parser.add_argument("--output-dir")
args = parser.parse_args()
out = Path(args.output_dir)
report = {
    "case_id": "will_be_normalized",
    "method": "ours_corrective_rag",
    "success": True,
    "output_dir": str(out),
    "stdout_path": str(out / "generated_stdout.log"),
    "stderr_path": str(out / "generated_stderr.log"),
    "optimizer": "generated_script",
    "iterations": 1,
    "converged": True,
    "compliance": 1.0,
    "volume_fraction": 0.5,
    "files": {},
    "metrics": {"source": "fake"},
}
(out / "execution_report.json").write_text(json.dumps(report), encoding="utf-8")
"""
    trace = []
    generated_plan = select_or_generate_code(
        base_plan,
        case_spec=case,
        output_dir=tmp_path,
        agent_authority=AgentAuthority.LLM_PRIMARY,
        allow_generated_code=True,
        use_llm=True,
        llm=FakeLLM(
            GeneratedSolverDraft(
                decision=AgentDecision(decision="generate", confidence=0.9),
                code=script,
                steps=["生成 JSON 脚本接口求解器"],
            )
        ),
        trace=trace,
    )
    (tmp_path / "case_spec.json").write_text(json.dumps(case.model_dump(mode="json")), encoding="utf-8")
    (tmp_path / "code_plan.json").write_text(json.dumps(generated_plan.model_dump(mode="json")), encoding="utf-8")

    report = execute(case, generated_plan, tmp_path, generated_code_timeout_s=5)

    assert generated_plan.execution_mode == "generated_script"
    assert (tmp_path / "generated_solver.py").exists()
    assert (tmp_path / "generated_code_manifest.json").exists()
    assert report.success is True
    assert report.case_id == case.case_id
    assert report.metrics["sandbox"]["execution_mode"] == "generated_script"
    assert trace[0]["agent"] == "coder"
    assert trace[0]["generated_code_path"].endswith("generated_solver.py")


def test_generated_script_sandbox_rejects_banned_import(tmp_path):
    case = default_case_spec("cantilever", quick=True)
    code_path = tmp_path / "generated_solver.py"
    code_path.write_text("import subprocess\n", encoding="utf-8")
    plan = CodePlan(
        case_id=case.case_id,
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        template_id=case.benchmark_type.value,
        allow_generated_code=True,
        execution_mode="generated_script",
        generated_code_path=str(code_path),
    )

    report = execute(case, plan, tmp_path, generated_code_timeout_s=5)

    assert report.success is False
    assert report.error_type == "GeneratedCodeRejected"
    assert "subprocess" in (tmp_path / "generated_stderr.log").read_text(encoding="utf-8")


def test_evaluator_llm_primary_can_override_quality_failure(tmp_path):
    case = default_case_spec("cantilever", quick=True)
    density_path = tmp_path / "density.npy"
    np.save(density_path, np.full((4, 4), 0.5))
    execution = ExecutionReport(
        case_id=case.case_id,
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        success=True,
        output_dir=str(tmp_path),
        compliance=1.0,
        converged=True,
        files={"density": str(density_path)},
    )
    decision = AgentDecision(
        decision="pass",
        confidence=0.88,
        overridden_failure_modes=[
            FailureMode.VOLUME_CONSTRAINT_VIOLATION,
            FailureMode.GRAYNESS_TOO_HIGH,
        ],
        reasons=["该探索性生成运行允许保留灰度设计。"],
    )
    trace = []

    report, repair_plan, _ = evaluate_execution(
        case,
        execution,
        use_llm=True,
        allow_llm_override=True,
        llm=FakeLLM(decision),
        trace=trace,
    )

    assert report.local_has_quality_failure is True
    assert report.success is True
    assert report.has_quality_failure is False
    assert report.grayness_index == pytest.approx(1.0)
    assert report.overridden_failure_modes == [
        FailureMode.VOLUME_CONSTRAINT_VIOLATION,
        FailureMode.GRAYNESS_TOO_HIGH,
    ]
    assert repair_plan is None
    assert trace[0]["agent"] == "evaluator"
    assert trace[0]["confidence"] == pytest.approx(0.88)
