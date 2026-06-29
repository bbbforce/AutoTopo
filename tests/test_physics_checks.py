"""Expanded deterministic physics validation tests."""

from __future__ import annotations

from autotopo.agents.validator import validate
from autotopo.engines.structured_benchmarks import default_case_spec
from autotopo.schemas import FailureMode


def test_no_support_fail_closed():
    spec = default_case_spec("cantilever", quick=True)
    problem = dict(spec.problem)
    problem["boundary_conditions"] = []

    report = validate(spec.model_copy(update={"problem": problem}))

    assert report.is_valid is False
    assert FailureMode.NO_SUPPORT in report.failure_modes


def test_no_load_fail_closed():
    spec = default_case_spec("cantilever", quick=True)
    problem = dict(spec.problem)
    problem["loads"] = []

    report = validate(spec.model_copy(update={"problem": problem}))

    assert report.is_valid is False
    assert FailureMode.NO_LOAD in report.failure_modes


def test_invalid_volfrac_fail_closed():
    spec = default_case_spec("cantilever", quick=True).model_copy(update={"volume_fraction": 0.9})

    report = validate(spec)

    assert report.is_valid is False
    assert FailureMode.INVALID_VOLUME_FRACTION in report.failure_modes


def test_load_on_fixed_dof_detected():
    spec = default_case_spec("cantilever", quick=True)
    problem = dict(spec.problem)
    problem["loads"] = [
        {"type": "point_force", "location": "left_edge", "magnitude": 1.0, "direction": [1, 0]}
    ]

    report = validate(spec.model_copy(update={"problem": problem}))

    assert report.is_valid is False
    assert FailureMode.LOAD_ON_FIXED_DOF in report.failure_modes
