"""问题路由节点。

分析结构化问题定义，判断所需的约束和目标函数
是否已在标准库中。简单问题走标准路径，复杂问题
路由到理论推导 + 代码生成路径。
"""

from __future__ import annotations

from typing import Any

from autotopo.library.objectives import KNOWN_OBJECTIVES
from autotopo.library.constraints import KNOWN_CONSTRAINTS
from autotopo.state import AutoTopoState


def route_problem(state: AutoTopoState) -> dict[str, Any]:
    """路由节点：判断问题复杂度并决定执行路径。"""
    problem = state["problem_definition"]

    # 检查约束是否都在标准库中
    required_constraints = {c["type"] for c in problem.get("constraints", [])}
    unknown = required_constraints - KNOWN_CONSTRAINTS

    # 检查目标函数
    objective = problem.get("objective", "minimize_compliance")
    objective_known = objective in KNOWN_OBJECTIVES

    if not unknown and objective_known:
        return {
            "route": "standard_path",
            "unknown_constraints": [],
        }
    else:
        return {
            "route": "complex_path",
            "unknown_constraints": list(unknown),
        }


def route_decision(state: AutoTopoState) -> str:
    """条件边函数：返回路由标签。"""
    return state["route"]
