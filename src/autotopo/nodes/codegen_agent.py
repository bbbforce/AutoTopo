"""代码生成 Agent 节点。

基于理论推导结果，生成符合仿真引擎接口规范的 Python 代码。
"""

from __future__ import annotations

import ast
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.llm_factory import get_llm
from autotopo.state import AutoTopoState


CODEGEN_SYSTEM_PROMPT = """\
你是一个拓扑优化代码生成专家。基于力学理论推导结果，
生成符合 JAX-FEM 引擎接口的 Python 代码。

代码规范要求：
1. 使用 JAX (jax.numpy) 编写可微分的约束/目标函数
2. 函数签名必须符合 TopoEngine 接口：
   - def custom_objective(densities, displacement, ke) -> float
   - def custom_constraint(densities, displacement, ke) -> float
   - def custom_sensitivity(densities, displacement, ke) -> np.ndarray
3. 利用 jax.grad 或 jax.jacrev 进行自动微分（优先于手动实现）
4. 代码必须是纯函数式的（无副作用），适配 JAX 的 JIT 编译
5. 只输出 Python 代码块，不要包含解释文字

输出格式：
```python
# 你的代码
```
"""


def _validate_syntax(code: str) -> str:
    """AST 语法校验，确保生成的代码可解析。"""
    # 提取代码块
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    code = code.strip()
    ast.parse(code)  # 语法错误会抛异常
    return code


def code_generation(state: AutoTopoState) -> dict[str, Any]:
    """代码生成节点：理论公式 → 引擎兼容代码。"""
    llm = get_llm()

    theory = state.get("theory_result", "")
    problem_yaml = state.get("problem_yaml", "")

    prompt = f"""\
请基于以下力学理论推导，生成符合 JAX-FEM 引擎接口的 Python 代码。

## 理论推导
{theory}

## 问题定义
```yaml
{problem_yaml}
```

请生成可直接使用的 Python 代码。
"""

    messages = [
        SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    result = llm.invoke(messages)

    try:
        validated_code = _validate_syntax(result.content)
    except SyntaxError as e:
        return {
            "generated_code": "",
            "error": f"生成的代码存在语法错误: {e}",
        }

    return {"generated_code": validated_code}
