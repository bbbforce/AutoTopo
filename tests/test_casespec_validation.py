"""CaseSpec 与 Validator 测试。"""

from __future__ import annotations

import pytest

from autotopo.agents.scientist import build_case_spec
from autotopo.agents.validator import validate
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.schemas import BenchmarkType, CaseSpec, FailureMode


def test_scientist_structured_params_take_priority():
    spec = build_case_spec(
        "做一个悬臂梁",
        structured_params={"benchmark_type": "mbb", "case_id": "forced_mbb", "volume_fraction": 0.35},
        quick=True,
    )

    assert spec.benchmark_type == BenchmarkType.MBB
    assert spec.case_id == "forced_mbb"
    assert spec.volume_fraction == 0.35


def test_scientist_extracts_explicit_chinese_parameters_from_text():
    spec = build_case_spec(
        "标准半对称 MBB 梁拓扑优化问题。设计域尺寸为 150x50，"
        "目标是最小化柔度，体积分数约束为 0.5，惩罚因子p=1，过滤半径r=10",
        quick=False,
    )

    assert spec.benchmark_type == BenchmarkType.MBB
    assert spec.nelx == 150
    assert spec.nely == 50
    assert spec.volume_fraction == pytest.approx(0.5)
    assert spec.penal == pytest.approx(1.0)
    assert spec.rmin == pytest.approx(10.0)
    assert spec.problem["domain"]["nelx"] == 150
    assert spec.problem["parameters"]["penal"] == pytest.approx(1.0)
    assert spec.problem["parameters"]["rmin"] == pytest.approx(10.0)


def test_validator_accepts_valid_case():
    spec = default_case_spec("cantilever", quick=True)

    report = validate(spec)

    assert report.is_valid is True
    assert report.failure_modes == []


def test_validator_fail_closed_without_support():
    spec = default_case_spec("cantilever", quick=True)
    problem = dict(spec.problem)
    problem["boundary_conditions"] = []
    broken = spec.model_copy(update={"problem": problem})

    report = validate(broken)

    assert report.is_valid is False
    assert FailureMode.NO_SUPPORT in report.failure_modes


def test_casespec_rejects_invalid_volume_fraction():
    with pytest.raises(ValueError):
        CaseSpec(case_id="bad", benchmark_type="cantilever", volume_fraction=1.5)
