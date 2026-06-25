"""失败诊断、拓扑指标和修复规则测试。"""

from __future__ import annotations

import numpy as np

from autotopo.diagnostics.failure_modes import diagnose_execution_report
from autotopo.diagnostics.repair_rules import apply_repair_plan, build_repair_plan
from autotopo.diagnostics.topology_metrics import checkerboard_score, connectivity_score, grayness_index, volume_error
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.schemas import BenchmarkMethod, ExecutionReport, FailureMode


def test_diagnose_missing_dependency():
    report = ExecutionReport(
        case_id="case",
        method=BenchmarkMethod.OURS_CORRECTIVE_RAG,
        success=False,
        output_dir="/tmp/out",
        error_type="ModuleNotFoundError",
        exception="No module named scipy",
    )

    diagnosis = diagnose_execution_report(report)

    assert diagnosis.has_failure is True
    assert FailureMode.MISSING_DEPENDENCY in diagnosis.failure_modes
    assert diagnosis.auto_repair_allowed is False


def test_topology_metrics_detect_gray_and_checkerboard():
    gray = np.full((4, 4), 0.5)
    checker = np.indices((4, 4)).sum(axis=0) % 2

    assert grayness_index(gray) == 1.0
    assert checkerboard_score(checker.astype(float)) > 0.9
    assert connectivity_score(checker.astype(float)) < 1.0
    assert volume_error(np.full((4, 4), 0.4), 0.4) == 0.0


def test_repair_plan_updates_only_safe_parameters():
    spec = default_case_spec("cantilever", quick=True)

    plan = build_repair_plan(
        spec,
        [FailureMode.GRAYNESS_TOO_HIGH, FailureMode.CHECKERBOARD],
        repair_iteration=0,
        max_repair_rounds=3,
    )
    repaired = apply_repair_plan(spec, plan)

    assert plan.should_repair is True
    assert repaired.penal >= spec.penal
    assert repaired.rmin >= spec.rmin
    assert repaired.volume_fraction == spec.volume_fraction

