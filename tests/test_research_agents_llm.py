"""Optional LLM paths for the minimal research agents."""

from __future__ import annotations

import pytest

from autotopo.agents.planner import plan_code
from autotopo.agents.reviewer import review_execution_failure
from autotopo.agents.scientist import CaseSpecDraft, build_case_spec
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.schemas import (
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
