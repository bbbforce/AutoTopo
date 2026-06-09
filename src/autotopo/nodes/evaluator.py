"""视觉评估与反馈修正节点。

利用视觉 LLM 检查拓扑优化结果图，
识别灰度单元、棋盘格、孤岛等缺陷，
并自主生成修正方案。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.llm_factory import get_llm
from autotopo.schemas import EvaluationResult
from autotopo.state import AutoTopoState


EVALUATOR_SYSTEM_PROMPT = """\
你是一个拓扑优化结果质量评估专家。你将收到两张图片：
1. **拓扑优化密度分布结果图**（第一张）：深色部分为实体，白色部分为孔洞
2. **收敛历史曲线图**（第二张）：包含柔度（Compliance）和体积分数（Volume Fraction）随迭代步的变化

本项目使用 FEniCS + dolfin-adjoint 引擎，采用 SIMP 方法 + Helmholtz PDE 过滤 + SLSQP 优化器。

请综合分析两张图片，识别以下常见缺陷并以 JSON 格式给出修正建议：

## 缺陷类型

1. **灰度单元 (gray_elements)**：密度值处于 0 和 1 之间的中间区域，
   表现为灰色区域。修正方案：增大 SIMP 惩罚因子 (penal)，或减小 Helmholtz 过滤半径 (rmin)。

2. **棋盘格 (checkerboard)**：相邻单元密度值交替为 0 和 1，
   呈现棋盘状图案。修正方案：增大 Helmholtz 过滤半径 (rmin)。

3. **孤岛 (island)**：出现与主结构断开的小块材料区域。
   修正方案：增大过滤半径或调整惩罚因子。不得修改体积分数约束。

4. **断裂 (disconnection)**：结构在载荷传递路径上出现断裂。
   通常是边界条件或载荷设置有误。

## 收敛曲线分析要点
- 柔度曲线是否已平稳收敛？若末段仍在剧烈波动，说明需增加迭代次数或调整参数
- 体积分数是否稳定在目标值附近？若偏差较大，说明优化约束可能有问题

## 输出要求
- 综合分析密度分布图和收敛曲线
- 判断是否存在上述任何缺陷
- 如果有缺陷，评估严重程度 (minor/moderate/severe)
- 给出具体的参数调整建议（参数名、当前值、建议值、理由）
- 可调整参数仅限：penal (SIMP罚因子), rmin (Helmholtz过滤半径比例)
- 严禁建议或修改 volfrac / volume_fraction。体积分数是用户定义的问题约束，不属于反馈修正参数。
- 如果结果清晰且无明显缺陷（黑白分明、结构连通、无棋盘格、收敛良好），标记为无缺陷

请确保输出的 JSON 结构与字段完全匹配以下示例：
```json
{
  "has_defects": true,
  "defect_types": ["gray_elements"],
  "severity": "minor",
  "suggested_fixes": [
    {
      "parameter": "penal",
      "current_value": 3.0,
      "suggested_value": 4.0,
      "reason": "通过增大罚因子来抑制灰度单元"
    }
  ],
  "reasoning": "密度分布图中有轻微的灰色渐变区域，收敛曲线已基本平稳"
}
```
"""


def evaluate_result(state: AutoTopoState) -> dict[str, Any]:
    """视觉评估节点：LLM 检查结果图 → 缺陷报告。"""
    llm = get_llm(vision=True, structured_output=EvaluationResult)

    img_path = state["result_image_path"]
    img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode("utf-8")

    # 读取收敛历史图（可选）
    conv_img_path = state.get("convergence_image_path", "")
    conv_b64 = ""
    if conv_img_path and Path(conv_img_path).exists():
        conv_b64 = base64.b64encode(Path(conv_img_path).read_bytes()).decode("utf-8")

    current_params = state.get("current_params", {})
    iteration = state.get("iteration", 0)

    # 构建消息内容：文字 + 拓扑结果图 + 收敛曲线图
    content_parts: list[dict] = [
        {"type": "text", "text": (
            f"这是第 {iteration + 1} 次优化的结果。\n"
            f"当前参数：penal={current_params.get('penal', 3.0)}, "
            f"rmin={current_params.get('rmin', 0.05)} (Helmholtz 过滤半径比例), "
            f"volfrac={current_params.get('volfrac', 0.5)}\n"
            f"引擎: FEniCS + dolfin-adjoint, 优化器: {current_params.get('optimizer', 'SLSQP')}\n"
            f"第一张图是拓扑优化密度分布结果，第二张图是收敛历史曲线。\n"
            f"请综合两张图评估结果质量。"
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ]
    if conv_b64:
        content_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{conv_b64}"}}
        )

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=content_parts),
    ]

    evaluation: EvaluationResult = llm.invoke(messages)
    eval_dict = evaluation.model_dump(mode="json")

    # 更新历史记录
    history = list(state.get("history", []))
    history.append({
        "iteration": iteration,
        "params": current_params,
        "evaluation": eval_dict,
    })

    return {
        "evaluation": eval_dict,
        "iteration": iteration + 1,
        "history": history,
    }


def should_retry(state: AutoTopoState) -> str:
    """条件边函数：判断预览重试、最终精修或接受结果。"""
    evaluation = state.get("evaluation", {})
    iteration = state.get("iteration", 0)
    max_retries = state.get("max_retries", 2)
    solve_stage = state.get("solve_stage", "preview")
    solve_profile = state.get("solve_profile", "preview_refine")
    final_refine_done = state.get("final_refine_done", solve_profile != "preview_refine")

    if solve_stage == "final" or solve_profile == "final_only":
        return "accept"

    def finish_preview() -> str:
        if solve_profile == "preview_refine" and not final_refine_done:
            return "final"
        return "accept"

    if not evaluation.get("has_defects", False):
        return finish_preview()

    if evaluation.get("severity") == "minor":
        return finish_preview()

    if iteration >= max_retries:
        return finish_preview()  # 超过预览重试次数后进入最终精修或接受结果

    return "retry"


def prepare_final_refine(state: AutoTopoState) -> dict[str, Any]:
    """切换到最终精修阶段；最终阶段只求解一次，不再进入反馈长循环。"""
    return {
        "solve_stage": "final",
        "final_refine_done": True,
    }


def apply_fixes(state: AutoTopoState) -> dict[str, Any]:
    """反馈修正节点：根据评估建议调整参数。

    改进逻辑：
    - penal：惩罚因子原则上只增不减，防止灰色单元增多
    - rmin：Helmholtz 过滤半径依据缺陷双向更新
    - volfrac：体积分数是用户问题约束，反馈闭环严禁修改
    - 步进限幅：防止参数调整过猛导致收敛震荡
    """
    evaluation = state.get("evaluation", {})
    defect_types = evaluation.get("defect_types", [])
    current_params = dict(state.get("current_params", {}))

    # 安全边界 (rmin 是相对域尺寸的比例)
    BOUNDS = {
        "penal": (1.0, 10.0),
        "rmin": (0.01, 0.2),        # Helmholtz 过滤半径比例
    }
    SUPPORTED_PARAMS = set(BOUNDS)

    for fix in evaluation.get("suggested_fixes", []):
        param_name = fix["parameter"]
        if param_name not in SUPPORTED_PARAMS:
            continue

        suggested = fix["suggested_value"]
        current = current_params.get(param_name, suggested)

        if param_name == "penal":
            # penal 只增不减
            target = max(current, suggested)
            # 步进限幅：每次最多增加 50%
            max_step = current * 0.5
            target = min(target, current + max_step)
        elif param_name == "rmin":
            # rmin 依据具体缺陷进行双向调整
            if "checkerboard" in defect_types:
                # 存在棋盘格时，必须增大以平滑
                target = max(current, suggested)
                max_step = current * 0.5
                target = min(target, current + max_step)
            elif "gray_elements" in defect_types and "checkerboard" not in defect_types:
                # 仅灰色单元时允许减小
                target = min(current, suggested)
                max_step = current * 0.5
                target = max(target, current - max_step)
            else:
                target = suggested
                max_step = current * 0.5
                target = min(max(target, current - max_step), current + max_step)
        lo, hi = BOUNDS.get(param_name, (float("-inf"), float("inf")))
        current_params[param_name] = round(min(max(target, lo), hi), 4)

    return {"current_params": current_params}
