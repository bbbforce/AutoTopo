"""LangGraph 主工作流图定义。

编排所有节点，定义条件分支和循环反馈边。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langgraph.graph import END, StateGraph

from autotopo.nodes.codegen_agent import code_generation
from autotopo.nodes.evaluator import apply_fixes, evaluate_result, should_retry
from autotopo.nodes.input_parser import parse_input
from autotopo.nodes.router import route_decision, route_problem
from autotopo.nodes.simulator import run_simulation
from autotopo.nodes.theory_agent import theory_derivation
from autotopo.state import AutoTopoState


def _save_output(state: AutoTopoState) -> dict[str, Any]:
    """最终输出节点：保存结果到指定位置。"""
    output_dir = Path(state.get("output_path", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存问题定义
    problem_path = output_dir / "problem_definition.yaml"
    problem_path.write_text(
        yaml.dump(state.get("problem_definition", {}), allow_unicode=True),
        encoding="utf-8",
    )

    # 保存评估历史
    if state.get("history"):
        from autotopo.utils.io import save_json
        save_json(state["history"], str(output_dir / "evaluation_history.json"))

    # 最终结果图路径
    final_img = state.get("result_image_path", "")

    return {"output_path": str(output_dir), "result_image_path": final_img}


def build_graph() -> StateGraph:
    """构建并返回 AutoTopo 工作流图。"""

    workflow = StateGraph(AutoTopoState)

    # ── 注册节点 ──
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("route_problem", route_problem)
    workflow.add_node("theory_derivation", theory_derivation)
    workflow.add_node("code_generation", code_generation)
    workflow.add_node("run_simulation", run_simulation)
    workflow.add_node("evaluate_result", evaluate_result)
    workflow.add_node("apply_fixes", apply_fixes)
    workflow.add_node("save_output", _save_output)

    # ── 定义边 ──

    # 入口 → 解析
    workflow.set_entry_point("parse_input")

    # 解析 → 路由
    workflow.add_edge("parse_input", "route_problem")

    # 路由 → 条件分支
    workflow.add_conditional_edges(
        "route_problem",
        route_decision,
        {
            "standard_path": "run_simulation",   # 简单问题直接仿真
            "complex_path": "theory_derivation",  # 复杂问题先推导
        },
    )

    # 理论推导 → 代码生成 → 仿真
    workflow.add_edge("theory_derivation", "code_generation")
    workflow.add_edge("code_generation", "run_simulation")

    # 仿真 → 视觉评估
    workflow.add_edge("run_simulation", "evaluate_result")

    # 评估 → 条件分支（反馈闭环）
    workflow.add_conditional_edges(
        "evaluate_result",
        should_retry,
        {
            "retry": "apply_fixes",  # 存在缺陷 → 修正
            "accept": "save_output",  # 质量合格 → 保存
        },
    )

    # 修正 → 重新仿真（循环）
    workflow.add_edge("apply_fixes", "run_simulation")

    # 保存 → 结束
    workflow.add_edge("save_output", END)

    return workflow


def compile_graph():
    """编译工作流图，返回可执行的 app。"""
    return build_graph().compile()
