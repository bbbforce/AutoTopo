"""多模态输入解析节点。

接收用户文本描述 + 设计域/非设计域示意图，
通过视觉 LLM 解析为结构化的 OptimizationProblem。

适配 FEniCS (DOLFIN) + dolfin-adjoint 后端。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.llm_factory import get_llm
from autotopo.schemas import OptimizationProblem
from autotopo.state import AutoTopoState


PARSER_SYSTEM_PROMPT = """\
你是一个拓扑优化问题解析专家。你的任务是将用户的自然语言描述（可能附带设计域示意图）
解析为结构化的 JSON 拓扑优化问题定义。

本项目使用 FEniCS (DOLFIN) + dolfin-adjoint 引擎，采用 Gmsh 生成三角形非结构网格。

关键要求：
1. 精确识别设计域尺寸、材料参数、边界条件、载荷和约束
2. 如果用户提供了设计域/非设计域的示意图，识别图中的非设计域区域（如孔洞、固定区域）
3. 对于未明确指定的参数，使用合理的默认值
4. 边界条件位置使用描述性字符串（如 "left_edge", "bottom_left_corner"）
5. 载荷方向使用 [fx, fy] 格式的方向向量
6. mesh_resolution 控制网格精细程度（值越小越精细），典型值 0.5~2.0

常见拓扑优化问题模式：
- MBB 梁：左端下角固定x,y，右端下角固定y，顶部中点施加向下集中力
- 半对称 MBB 梁（对称面在左侧）：左边界 fixed_x，右下角 fixed_y，左上角向下力
- 半对称 MBB 梁（对称面在右侧）：右边界 fixed_x，左下角 fixed_y，右上角向下力
- 悬臂梁：左端 fixed，右端中点施加向下集中力
- 桥梁：底部两端铰接，顶部分布载荷

请确保输出的 JSON 结构与字段完全匹配以下示例：
```json
{
  "description": "悬臂梁设计问题说明...",
  "domain": {
    "width": 60.0,
    "height": 20.0,
    "mesh_resolution": 1.0,
    "non_design_regions": []
  },
  "material": {
    "youngs_modulus": 1.0,
    "poissons_ratio": 0.3
  },
  "boundary_conditions": [
    {
      "type": "fixed",
      "location": "left_edge"
    }
  ],
  "loads": [
    {
      "type": "point_force",
      "location": "right_center",
      "magnitude": 1.0,
      "direction": [0.0, -1.0]
    }
  ],
  "objective": "minimize_compliance",
  "constraints": [
    {
      "type": "volume_fraction",
      "value": 0.5,
      "description": "体积分数约束"
    }
  ],
  "parameters": {
    "penal": 3.0,
    "rmin": 0.05,
    "max_iter": 200,
    "tol": 1e-6,
    "optimizer": "SLSQP"
  }
}
```
"""


def _encode_image(path: str) -> str:
    """将图片文件编码为 base64 字符串。"""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def parse_input(state: AutoTopoState) -> dict[str, Any]:
    """多模态解析节点：文本 + 图片 → 结构化问题定义。"""
    llm = get_llm(vision=True, structured_output=OptimizationProblem)

    # 构建多模态消息
    content: list[dict] = [
        {"type": "text", "text": state["user_input"]},
    ]

    # 如果有设计域示意图，附加图片
    for img_path in state.get("image_paths", []):
        if Path(img_path).exists():
            b64 = _encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    messages = [
        SystemMessage(content=PARSER_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    problem: OptimizationProblem = llm.invoke(messages)
    problem_dict = problem.model_dump(mode="json")

    return {
        "problem_definition": problem_dict,
        "problem_yaml": yaml.dump(problem_dict, allow_unicode=True, default_flow_style=False),
    }
