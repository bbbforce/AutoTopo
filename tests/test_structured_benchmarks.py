"""结构化 benchmark 模板测试。"""

from __future__ import annotations

from autotopo.engines.structured_benchmarks import case_to_problem, default_case_spec, l_shape_passive_void_mask, minimal_benchmark_cases
from autotopo.schemas import BenchmarkType


def test_minimal_case_matrix_has_six_cases():
    cases = minimal_benchmark_cases(quick=True)

    assert len(cases) == 6
    assert {case.benchmark_type for case in cases} == {
        BenchmarkType.MBB,
        BenchmarkType.CANTILEVER,
        BenchmarkType.L_SHAPE,
    }
    assert {case.variant for case in cases} == {"clear", "fuzzy"}


def test_l_shape_mask_removes_upper_right_quadrant():
    mask = l_shape_passive_void_mask(10, 10)

    assert len(mask) == 10
    assert len(mask[0]) == 10
    assert mask[0][5] is True
    assert mask[9][9] is False


def test_case_to_problem_contains_expected_mbb_supports():
    spec = default_case_spec("mbb", quick=True)
    problem = case_to_problem(spec)

    assert problem["domain"]["nelx"] == spec.nelx
    assert {"type": "fixed_x", "location": "left_edge"} in problem["boundary_conditions"]
    assert {"type": "fixed_y", "location": "bottom_right"} in problem["boundary_conditions"]
    assert problem["parameters"]["optimizer"] == "MMA"

