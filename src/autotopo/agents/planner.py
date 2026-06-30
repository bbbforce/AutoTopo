"""Planner agent：把 CaseSpec 拆成可执行任务。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.agents.llm_utils import AgentTrace, try_invoke_structured
from autotopo.schemas import BenchmarkMethod, CaseSpec, CodePlan, RetrievedEvidence


PLANNER_SYSTEM_PROMPT = """\
你是 AutoTopo research_graph 的 Planner agent。

## 职责
把已经规范化的 CaseSpec 和当前 benchmark method 拆解成可执行的 CodePlan。
你的计划用于最小研究 workflow，不负责生成自由代码，也不负责执行求解。

## 输入上下文
- case_spec：Scientist 和本地模板重建后的规范化 CaseSpec。
- method：当前 benchmark 对比方法。
- retrieved_evidence：本地 RAG 返回的证据片段，可用于补充 steps 和 evidence_ids。

## 输出格式
只输出一个严格匹配 CodePlan schema 的 JSON 对象，不要 Markdown、解释文字或额外字段。
必须保持：
- case_id 与输入 case_spec.case_id 一致。
- method 与输入 method 一致。
- engine 必须是 "python_simp_mma"。
- optimizer 必须是 "MMA"。
- allow_generated_code 必须是 false。
- template_id 必须跟随 case_spec.benchmark_type，只能是 "cantilever"、"mbb"、"l_shape"。
- parameters 使用 CaseSpec 中的 nelx、nely、volume_fraction、penal、rmin、max_iter、tol。

## 关键判断标准
- steps 应清楚说明选择结构化 benchmark 模板、实例化 PythonSimpMMAEngine、运行 SIMP/MMA、保存标准产物。
- evidence_ids 只能引用 retrieved_evidence 中真实存在的 evidence_id。
- 可以优化步骤表述和证据选择，但不能改变执行契约。

## 硬性边界
- 不得发明新 engine、新 optimizer、新 template、新代码路径或外部服务。
- 不得把 allow_generated_code 设为 true。
- 不得修改 CaseSpec 参数或提出越权修复。

## 失败/不确定时如何输出
如果证据不足，仍输出固定安全模板计划；steps 使用保守描述，evidence_ids 可为空列表。
"""


def plan_code(
    case_spec: CaseSpec,
    method: BenchmarkMethod,
    evidence: list[RetrievedEvidence] | None = None,
    *,
    use_llm: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> CodePlan:
    """为最小实验选择固定后端和模板。"""

    evidence = evidence or []
    deterministic = CodePlan(
        case_id=case_spec.case_id,
        method=method,
        engine="python_simp_mma",
        template_id=case_spec.benchmark_type.value,
        optimizer="MMA",
        allow_generated_code=False,
        evidence_ids=[item.evidence_id for item in evidence],
        parameters={
            "nelx": case_spec.nelx,
            "nely": case_spec.nely,
            "volfrac": case_spec.volume_fraction,
            "penal": case_spec.penal,
            "rmin": case_spec.rmin,
            "max_iter": case_spec.max_iter,
            "tol": case_spec.tol,
        },
        steps=[
            "选择结构化 benchmark 模板",
            "实例化 PythonSimpMMAEngine",
            "运行 SIMP/MMA 优化",
            "保存标准产物",
        ],
    )
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请规划这个最小拓扑优化运行。\n"
                f"case_spec: {case_spec.model_dump(mode='json')}\n"
                f"method: {method.value}\n"
                f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                "返回的 CodePlan 必须保持已批准的 engine、optimizer、template 和 generated-code 策略。"
            )
        ),
    ]
    llm_plan = try_invoke_structured(
        agent="planner",
        messages=messages,
        output_model=CodePlan,
        provider=llm_provider,
        llm=llm,
        use_llm=use_llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    if not isinstance(llm_plan, CodePlan):
        return deterministic

    evidence_ids = {item.evidence_id for item in evidence}
    safe_evidence_ids = [item for item in llm_plan.evidence_ids if item in evidence_ids]
    return deterministic.model_copy(
        update={
            "steps": llm_plan.steps or deterministic.steps,
            "evidence_ids": safe_evidence_ids or deterministic.evidence_ids,
        }
    )
