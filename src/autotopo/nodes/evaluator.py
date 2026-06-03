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
你是一个拓扑优化结果质量评估专家。你将收到两张图片：
1. **拓扑优化密度分布结果图**（第一张）：深色部分为实体，白色部分为孔洞
2. **收敛历史曲线图**（第二张）：包含柔度（Compliance）和体积分数（Volume Fraction）随迭代步的变化

请综合分析两张图片，识别以下常见缺陷并以 JSON 格式给出修正建议：

## 缺陷类型

1. **灰度单元 (gray_elements)**：密度值处于 0 和 1 之间的中间区域，
   表现为灰色区域。修正方案：增大 SIMP 惩罚因子 (penal)，或增大过滤半径 (rmin)，或启用 Heaviside 投影（通过建议参数 ft=2）。

2. **棋盘格 (checkerboard)**：相邻单元密度值交替为 0 和 1，
   呈现棋盘状图案。修正方案：增大过滤半径 (rmin)。

3. **孤岛 (island)**：出现与主结构断开的小块材料区域。
   修正方案：降低体积分数约束或增加连通性约束。

4. **断裂 (disconnection)**：结构在载荷传递路径上出现断裂。
   通常是边界条件或载荷设置有误。

## 收敛曲线分析要点
- 柔度曲线是否已平稳收敛？若末段仍在剧烈波动，说明需增加迭代次数或调整参数
- 体积分数是否稳定在目标值附近？若偏差较大，说明优化约束可能有问题
- 若柔度曲线出现突变跳跃，可能是 β-continuation 的 β 翻倍导致的正常现象

## 输出要求
- 综合分析密度分布图和收敛曲线
- 判断是否存在上述任何缺陷
- 如果有缺陷，评估严重程度 (minor/moderate/severe)
- 给出具体的参数调整建议（参数名、当前值、建议值、理由）
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
            f"rmin={current_params.get('rmin', 1.5)}, "
            f"volfrac={current_params.get('volfrac', 0.5)}, "
            f"ft={current_params.get('ft', 1)} (1=密度过滤, 2=Heaviside投影), "
            # f"beta={current_params.get('beta', 1.0)}, "
            # f"beta_max={current_params.get('beta_max', 64.0)}, "
            # f"beta_interval={current_params.get('beta_interval', 40)}\n"
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
        # "beta": (1.0, 1.0),   # beta 初始值锁定为 1.0，通过 continuation 自动增长
        # "beta_max": (8.0, 128.0),
        # "beta_interval": (10, 80),
        # "eta": (0.3, 0.7),
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
        # elif param_name == "ft":
        #     # ft (过滤类型) 必须是整型
        #     current_params[param_name] = int(suggested)
        #     continue
        # elif param_name == "beta_interval":
        #     # beta_interval 必须是整型
        #     target = int(suggested)
        #     lo, hi = BOUNDS.get(param_name, (10, 80))
        #     current_params[param_name] = min(max(target, lo), hi)
        #     continue
        # elif param_name == "beta":
        #     # beta 初始值锁定为 1.0
        #     continue
        # elif param_name in ("beta_max", "eta"):
        #     # Heaviside 参数：直接采纳建议值，只做边界约束
        #     target = suggested
        else:
            current_params[param_name] = suggested
            continue

        lo, hi = BOUNDS.get(param_name, (float("-inf"), float("inf")))
        current_params[param_name] = round(min(max(target, lo), hi), 2)

    return {"current_params": current_params}

