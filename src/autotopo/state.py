"""LangGraph 全局状态定义。

所有节点通过读写此状态进行通信。
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class AutoTopoState(TypedDict, total=False):
    """工作流全局状态"""

    # ── 用户输入 ──
    user_input: str                          # 用户文本描述
    image_paths: list[str]                   # 设计域/非设计域示意图路径

    # ── 解析结果 ──
    problem_definition: dict[str, Any]       # OptimizationProblem.model_dump()
    problem_yaml: str                        # YAML 序列化

    # ── 路由 ──
    route: str                               # "standard_path" | "complex_path"
    unknown_constraints: list[str]           # 标准库中未覆盖的约束类型

    # ── 理论推导 & 代码生成 ──
    theory_result: str                       # 力学推导结果 (LaTeX + 伪代码)
    generated_code: str                      # 生成的 Python 代码

    # ── 仿真 ──
    current_params: dict[str, Any]           # 当前优化参数
    solve_result: dict[str, Any]             # 求解结果
    density_field: Any                       # 密度场数组
    result_image_path: str                   # 结果图路径

    # ── 评估 & 反馈 ──
    evaluation: dict[str, Any]               # EvaluationResult.model_dump()
    iteration: int                           # 当前反馈迭代次数
    max_retries: int                         # 最大重试次数
    history: list[dict[str, Any]]            # 迭代历史记录

    # ── 输出 ──
    output_path: str                         # 最终结果保存路径
    error: Optional[str]                     # 错误信息
