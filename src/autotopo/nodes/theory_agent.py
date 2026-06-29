"""理论推导 Agent 节点。

负责对标准库未覆盖的约束/目标函数进行力学公式推导，
输出 FEniCS UFL 变分形式和 dolfin-adjoint 的使用方法。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.llm_factory import get_llm
from autotopo.state import AutoTopoState


THEORY_SYSTEM_PROMPT = """\
你是一个结构力学与拓扑优化理论专家。你的任务是为给定的约束类型或目标函数
推导其在 SIMP (Solid Isotropic Material with Penalization) 框架下的公式，
并适配 FEniCS (DOLFIN 2019) + dolfin-adjoint 实现。

关键：dolfin-adjoint 会自动记录所有 FEniCS 操作并通过伴随方法求灵敏度，
因此你不需要手动推导灵敏度公式，只需要给出正确的正问题变分形式。

输出要求：
1. 明确写出约束/目标函数的数学表达式
2. 给出 FEniCS UFL 变分形式的伪代码
3. 说明如何在 dolfin-adjoint 中定义 ReducedFunctional
4. 用 LaTeX 格式写数学公式
5. 注明 dolfin-adjoint 可自动求导，无需手动灵敏度推导

示例 — von Mises 应力约束：
- 目标：σ_vm(ρ) ≤ σ_allow
- UFL: sigma_vm = sqrt(inner(dev_sigma, dev_sigma))
- ReducedFunctional: 对应力的 p-范数构建 ReducedFunctional
- dolfin-adjoint 自动处理伴随
"""


def theory_derivation(state: AutoTopoState) -> dict[str, Any]:
    """理论推导节点：推导自定义约束的变分形式。"""
    llm = get_llm(provider=state.get("llm_provider"))

    unknown = state.get("unknown_constraints", [])
    problem_yaml = state.get("problem_yaml", "")

    prompt = f"""\
请为以下拓扑优化问题中的非标准约束推导 FEniCS UFL 变分形式：

## 非标准约束类型
{', '.join(unknown)}

## 问题上下文
```yaml
{problem_yaml}
```

请按照系统提示中的格式要求，给出完整的数学推导和 FEniCS UFL 伪代码。
注意：dolfin-adjoint 会自动计算灵敏度，无需手动推导偏导数。
"""

    messages = [
        SystemMessage(content=THEORY_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    result = llm.invoke(messages)
    return {"theory_result": result.content}
