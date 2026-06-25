"""Coder agent：第一轮只选择已有模板和后端，不自由生成求解器代码。"""

from __future__ import annotations

from autotopo.schemas import CodePlan


def select_or_generate_code(code_plan: CodePlan) -> CodePlan:
    """返回可执行计划；当前 deterministic 版本不生成新代码。"""

    return code_plan.model_copy(
        update={
            "allow_generated_code": False,
            "steps": code_plan.steps + ["确认不生成自由形式求解器代码"],
        }
    )

