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
from autotopo.schemas import EvaluationResult, ParameterAdjustment
from autotopo.state import AutoTopoState


EVALUATOR_SYSTEM_PROMPT = """\
你是一个拓扑优化结果质量评估专家。你需要分析拓扑优化的密度分布结果图，深色部分为实体，白色部分为孔洞
识别以下常见缺陷并以 JSON 格式给出修正建议：

## 缺陷类型

1. **灰度单元 (gray_elements)**：密度值处于 0 和 1 之间的中间区域，
   表现为灰色区域。修正方案：增大 SIMP 惩罚因子 (penal)，或增大过滤半径 (rmin)，或启用 Heaviside 投影。

2. **棋盘格 (checkerboard)**：相邻单元密度值交替为 0 和 1，
   呈现棋盘状图案。修正方案：增大过滤半径 (rmin)。

3. **孤岛 (island)**：出现与主结构断开的小块材料区域。
   修正方案：降低体积分数约束或增加连通性约束。

4. **断裂 (disconnection)**：结构在载荷传递路径上出现断裂。
   通常是边界条件或载荷设置有误。

## 输出要求
- 仔细观察图片中的密度分布
- 判断是否存在上述任何缺陷
- 如果有缺陷，评估严重程度 (minor/moderate/severe)
- 给出具体的参数调整建议（参数名、当前值、建议值、理由）
- 如果结果清晰且无明显缺陷（黑白分明、结构连通、无棋盘格），标记为无缺陷

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
  "reasoning": "密度分布图中有轻微的灰色渐变区域"
}
```
"""


def evaluate_result(state: AutoTopoState) -> dict[str, Any]:
    """视觉评估节点：LLM 检查结果图 → 缺陷报告。"""
    llm = get_llm(vision=True, structured_output=EvaluationResult)

    img_path = state["result_image_path"]
    img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode("utf-8")

    current_params = state.get("current_params", {})
    iteration = state.get("iteration", 0)

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": (
                f"这是第 {iteration + 1} 次优化的结果图。\n"
                f"当前参数：penal={current_params.get('penal', 3.0)}, "
                f"rmin={current_params.get('rmin', 1.5)}, "
                f"volfrac={current_params.get('volfrac', 0.5)}\n"
                f"请评估结果质量。"
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]),
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
    """条件边函数：判断是否需要重试。"""
    evaluation = state.get("evaluation", {})
    iteration = state.get("iteration", 0)
    max_retries = state.get("max_retries", 3)

    if not evaluation.get("has_defects", False):
        return "accept"

    if iteration >= max_retries:
        return "accept"  # 超过最大重试次数，接受当前结果

    return "retry"


def apply_fixes(state: AutoTopoState) -> dict[str, Any]:
    """反馈修正节点：根据评估建议调整参数。

    改进逻辑：
    - penal：惩罚因子原则上只增不减（强制单向增大），防止回退导致灰色单元增多
    - rmin：过滤半径依据缺陷双向更新，有棋盘格时增大，仅有灰色单元时减小，防止过度模糊
    - 步进限幅：防止参数调整过猛导致收敛震荡
    """
    evaluation = state.get("evaluation", {})
    defect_types = evaluation.get("defect_types", [])
    current_params = dict(state.get("current_params", {}))

    # 安全边界
    BOUNDS = {
        "penal": (1.0, 10.0),
        "rmin": (0.5, 4.0),
        "volfrac": (0.1, 0.9),
    }

    for fix in evaluation.get("suggested_fixes", []):
        param_name = fix["parameter"]
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
                # 存在棋盘格时，必须增大以平滑数值不稳定
                target = max(current, suggested)
                # 步进限幅：最多增加 50%
                max_step = current * 0.5
                target = min(target, current + max_step)
            elif "gray_elements" in defect_types and not any(d == "checkerboard" for d in defect_types):
                # 仅存在灰色单元且没有棋盘格时，允许减小 rmin 以收紧过渡边界，减少灰色单元
                target = min(current, suggested)
                # 步进限幅：最多减小 50%
                max_step = current * 0.5
                target = max(target, current - max_step)
            else:
                # 其它情况（如无相关缺陷或混合缺陷），允许双向调整
                target = suggested
                # 步进限幅：最多调整 50%
                max_step = current * 0.5
                target = min(max(target, current - max_step), current + max_step)
        elif param_name == "volfrac":
            target = suggested
        else:
            current_params[param_name] = suggested
            continue

        lo, hi = BOUNDS.get(param_name, (float("-inf"), float("inf")))
        current_params[param_name] = round(min(max(target, lo), hi), 2)

    return {"current_params": current_params}

