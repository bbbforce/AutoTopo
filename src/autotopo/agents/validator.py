"""Validator agent：物理和参数 fail-closed 检查。"""

from __future__ import annotations

from autotopo.diagnostics.physics_checks import validate_case_spec
from autotopo.schemas import CaseSpec, ValidationReport


def validate(case_spec: CaseSpec) -> ValidationReport:
    """验证 CaseSpec，不通过时 workflow 不进入 Executor。"""

    return validate_case_spec(case_spec)

