"""LangGraph 主工作流图定义。

编排所有节点，定义条件分支和循环反馈边。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml
from langgraph.graph import END, StateGraph

from autotopo.nodes.codegen_agent import code_generation
from autotopo.nodes.evaluator import (
    apply_fixes,
    evaluate_result,
    prepare_final_refine,
    should_retry,
)
from autotopo.nodes.input_parser import parse_input
from autotopo.nodes.router import route_decision, route_problem
from autotopo.nodes.simulator import run_simulation
from autotopo.nodes.theory_agent import theory_derivation
from autotopo.state import AutoTopoState


def _instrument_node(
    name: str,
    node: Callable[[AutoTopoState], dict[str, Any]],
    tracer: Any | None,
    *,
    agent: str | None = None,
) -> Callable[[AutoTopoState], dict[str, Any]]:
    """为 LangGraph 节点增加可选运行时事件记录。"""

    if tracer is None:
        return node

    def wrapped(state: AutoTopoState) -> dict[str, Any]:
        token = tracer.start_stage(
            name,
            agent=agent,
            summary=f"{name} 开始",
            payload={
                "iteration": state.get("iteration", 0),
                "solve_stage": state.get("solve_stage", ""),
                "route": state.get("route", ""),
            },
        )
        try:
            result = node(state)
        except Exception as exc:
            tracer.fail_stage(token, exc, payload={"state_keys": sorted(state.keys())})
            raise

        tracer.complete_stage(
            token,
            summary=f"{name} 完成",
            payload=result,
        )
        return result

    return wrapped


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

    # 导出收敛历史图
    solve_result = state.get("solve_result", {})
    compliance_history = solve_result.get("compliance_history", [])
    volume_history = solve_result.get("volume_history", [])
    if compliance_history and volume_history:
        from autotopo.utils.visualization import plot_convergence_history
        plot_convergence_history(
            compliance_history,
            volume_history,
            str(output_dir / "convergence_history.png"),
        )

    # 生成 Markdown 汇总报告
    _generate_report(state, output_dir)

    # 最终结果图路径
    final_img = state.get("result_image_path", "")

    return {"output_path": str(output_dir), "result_image_path": final_img}


def _generate_report(state: AutoTopoState, output_dir: Path) -> None:
    """生成 Markdown 格式汇总报告。"""
    problem = state.get("problem_definition", {})
    params = state.get("current_params", {})
    history = state.get("history", [])
    evaluation = state.get("evaluation", {})
    solve_result = state.get("solve_result", {})
    timings = solve_result.get("timings", {})

    lines = [
        "# AutoTopo 优化报告 (FEniCS + dolfin-adjoint)\n",
        f"## 问题描述\n",
        f"{problem.get('description', 'N/A')}\n",
        f"## 设计域\n",
        f"| 参数 | 值 |",
        f"|------|-----|",
        f"| 尺寸 | {problem.get('domain', {}).get('width', '?')} × {problem.get('domain', {}).get('height', '?')} |",
        f"| 网格分辨率 | {problem.get('domain', {}).get('mesh_resolution', '?')} |",
        f"| 目标函数 | {problem.get('objective', 'minimize_compliance')} |",
        f"| 体积分数 | {params.get('volfrac', '?')} |",
        f"| 罚因子 | {params.get('penal', '?')} |",
        f"| Helmholtz 过滤半径 | {params.get('rmin', '?')} |",
        f"| 优化器 | {params.get('optimizer', 'SLSQP')} |",
        "",
        f"## 求解结果\n",
        f"- 求解阶段: {solve_result.get('solve_stage', state.get('solve_stage', '?'))}",
        f"- 迭代次数: {solve_result.get('iterations', '?')}",
        f"- 收敛: {'是' if solve_result.get('converged') else '否'}",
        f"- 早停: {'是' if solve_result.get('early_stopped') else '否'}",
    ]

    compliance = solve_result.get("compliance_history", [])
    if compliance:
        lines.append(f"- 最终柔度: {compliance[-1]:.4f}")

    if timings:
        lines.extend([
            "",
            f"## 性能统计\n",
            f"| 阶段 | 耗时(s) |",
            f"|------|---------|",
        ])
        for key in ["mesh", "setup", "optimization", "export", "total"]:
            if key in timings:
                lines.append(f"| {key} | {timings[key]:.3f} |")

    if state.get("result_image_path"):
        lines.extend([
            "",
            f"## 结果图\n",
            f"![topology result]({Path(state['result_image_path']).name})",
        ])

    convergence_img = output_dir / "convergence_history.png"
    if convergence_img.exists():
        lines.extend([
            "",
            f"## 收敛历史\n",
            f"![convergence](convergence_history.png)",
        ])

    if history:
        lines.extend([
            "",
            f"## 评估迭代记录\n",
            f"| 迭代 | penal | rmin | 缺陷 | 严重度 |",
            f"|------|-------|------|------|--------|",
        ])
        for h in history:
            p = h.get("params", {})
            e = h.get("evaluation", {})
            defects = ", ".join(e.get("defect_types", []))
            lines.append(
                f"| {h.get('iteration', '?')} "
                f"| {p.get('penal', '?')} "
                f"| {p.get('rmin', '?')} "
                f"| {defects or '无'} "
                f"| {e.get('severity', '-')} |"
            )

    if evaluation:
        lines.extend([
            "",
            f"## 最终评估\n",
            f"- 存在缺陷: {'是' if evaluation.get('has_defects') else '否'}",
            f"- 评估结论: {evaluation.get('reasoning', 'N/A')}",
        ])

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_graph(tracer: Any | None = None) -> StateGraph:
    """构建并返回 AutoTopo 工作流图。"""

    workflow = StateGraph(AutoTopoState)

    # ── 注册节点 ──
    workflow.add_node("parse_input", _instrument_node("parse_input", parse_input, tracer, agent="Parser"))
    workflow.add_node("route_problem", _instrument_node("route_problem", route_problem, tracer, agent="Router"))
    workflow.add_node(
        "theory_derivation",
        _instrument_node("theory_derivation", theory_derivation, tracer, agent="Theory Agent"),
    )
    workflow.add_node(
        "code_generation",
        _instrument_node("code_generation", code_generation, tracer, agent="Codegen Agent"),
    )
    workflow.add_node(
        "run_simulation",
        _instrument_node("run_simulation", run_simulation, tracer, agent="Simulator"),
    )
    workflow.add_node(
        "evaluate_result",
        _instrument_node("evaluate_result", evaluate_result, tracer, agent="Evaluator"),
    )
    workflow.add_node("apply_fixes", _instrument_node("apply_fixes", apply_fixes, tracer, agent="Fixer"))
    workflow.add_node(
        "prepare_final_refine",
        _instrument_node("prepare_final_refine", prepare_final_refine, tracer, agent="Refiner"),
    )
    workflow.add_node("save_output", _instrument_node("save_output", _save_output, tracer, agent="Reporter"))

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
            "final": "prepare_final_refine",  # 预览通过 → 最终精修
            "accept": "save_output",  # 质量合格 → 保存
        },
    )

    # 修正 → 重新仿真（循环）
    workflow.add_edge("apply_fixes", "run_simulation")

    # 最终精修 → 重新仿真一次
    workflow.add_edge("prepare_final_refine", "run_simulation")

    # 保存 → 结束
    workflow.add_edge("save_output", END)

    return workflow


def compile_graph(tracer: Any | None = None):
    """编译工作流图，返回可执行的 app。"""
    return build_graph(tracer=tracer).compile()
