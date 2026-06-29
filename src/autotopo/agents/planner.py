"""Planner agent：把 CaseSpec 拆成可执行任务。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.agents.llm_utils import AgentTrace, try_invoke_structured
from autotopo.schemas import BenchmarkMethod, CaseSpec, CodePlan, RetrievedEvidence


PLANNER_SYSTEM_PROMPT = """\
You are the Planner agent for AutoTopo's minimal research workflow.

Create a CodePlan for an already-normalized CaseSpec. This workflow only allows
engine="python_simp_mma", optimizer="MMA", allow_generated_code=false, and one of
the supported benchmark templates. Do not invent new code paths. Return JSON
matching the CodePlan schema.
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
                "Plan this minimal topology optimization run.\n"
                f"case_spec: {case_spec.model_dump(mode='json')}\n"
                f"method: {method.value}\n"
                f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                "The returned plan must keep the approved engine, optimizer, "
                "template, and generated-code policy."
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
