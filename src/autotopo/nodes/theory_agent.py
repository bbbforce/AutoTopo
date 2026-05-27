"""理论推导 Agent 节点。

负责对标准库未覆盖的约束/目标函数进行力学公式推导，
输出灵敏度分析的数学表达式。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.llm_factory import get_llm
from autotopo.state import AutoTopoState


THEORY_SYSTEM_PROMPT = """\
你是一个结构力学与拓扑优化理论专家。你的任务是为给定的约束类型或目标函数
推导其在 SIMP (Solid Isotropic Material with Penalization) 框架下的灵敏度分析公式。

输出要求：
1. 明确写出约束/目标函数的数学表达式
2. 推导其对密度变量 ρ_e 的灵敏度 (偏导数)
3. 给出伴随方法 (adjoint method) 的具体步骤（如果适用）
4. 用 LaTeX 格式写数学公式
5. 用伪代码描述计算流程
6. 注明是否可以利用自动微分 (AD) 替代手动推导

示例 — von Mises 应力约束：
- 目标：σ_vm(ρ) ≤ σ_allow
- 灵敏度：dσ_vm/dρ_e 通过伴随方法求解
- 需要额外求解伴随方程 K·λ = ∂σ_vm/∂u
"""


def theory_derivation(state: AutoTopoState) -> dict[str, Any]:
    """理论推导节点：推导自定义约束的灵敏度公式。"""
    llm = get_llm()

    unknown = state.get("unknown_constraints", [])
    problem_yaml = state.get("problem_yaml", "")

    prompt = f"""\
请为以下拓扑优化问题中的非标准约束推导灵敏度分析公式：

## 非标准约束类型
{', '.join(unknown)}

## 问题上下文
```yaml
{problem_yaml}
```

请按照系统提示中的格式要求，给出完整的数学推导和伪代码。
"""

    messages = [
        SystemMessage(content=THEORY_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    result = llm.invoke(messages)
    return {"theory_result": result.content}
