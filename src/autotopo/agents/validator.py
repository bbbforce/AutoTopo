"""Validator agent：物理和参数检查，可在高自治模式下由 LLM 放行物理告警。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.agents.llm_utils import (
    AgentTrace,
    decision_allows_override,
    decision_trace_payload,
    enrich_latest_trace,
    try_invoke_structured,
)
from autotopo.diagnostics.physics_checks import validate_case_spec
from autotopo.rag.corrective_rag import retrieve_for_validation_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import AgentDecision, CaseSpec, FailureMode, Severity, ValidationReport


VALIDATOR_SYSTEM_PROMPT = """\
你是 AutoTopo research_graph 的 Validator agent。

## 职责
审阅本地 fail-closed 物理检查报告、CaseSpec 和 RAG 证据，判断本地失败是否可在 llm_primary 模式下被高置信放行。
你只提供裁决，不直接修改 CaseSpec，不执行求解。

## 输入上下文
- 可放行 failure_modes：调用方提供的白名单。
- case_spec：当前规范化问题。
- local_validation_report：本地物理检查报告，是默认可信来源。
- retrieved_evidence：与失败模式相关的本地证据。

## 输出格式
只输出一个严格匹配 AgentDecision schema 的 JSON 对象，不要 Markdown、解释文字或额外字段。
字段要求：
- target_agent 必须是 "validator"。
- decision 使用 "hold"、"reject"、"pass"、"allow"、"approve" 等简短动作。
- confidence 为 0.0 到 1.0；只有 confidence >= 0.70 且证据充分时才可放行。
- reasons 写明放行或保持 fail-closed 的原因。
- evidence_ids 只能引用 retrieved_evidence 中真实存在的 evidence_id。
- overridden_failure_modes 只能包含输入白名单中且本地报告实际出现的 failure mode。

## 关键判断标准
- local_validation_report 是保守基线，不可随意推翻。
- 只有确认问题仍可安全进入 Executor，且所有本地失败模式都可解释、可覆盖时，decision 才能使用 pass/allow/approve。
- 对无支撑、无载荷、刚体运动、载荷施加在固定自由度等问题必须非常保守。

## 硬性边界
- 不得覆盖白名单之外的 failure mode。
- 不得建议生成代码、修改求解器或更改参数。
- 不得用泛泛理由覆盖本地物理失败。

## 失败/不确定时如何输出
证据不足、置信度不足或仍有未解释失败模式时，输出 decision="hold" 或 "reject"，confidence < 0.70，overridden_failure_modes 为空列表。
"""


OVERRIDABLE_VALIDATION_MODES = {
    FailureMode.NO_SUPPORT,
    FailureMode.NO_LOAD,
    FailureMode.LOAD_ON_FIXED_DOF,
    FailureMode.RIGID_BODY_MOTION,
    FailureMode.INVALID_BOUNDARY_CONDITION,
    FailureMode.INVALID_LOAD_PATH,
}


def validate(
    case_spec: CaseSpec,
    *,
    retriever: LocalRetriever | None = None,
    use_llm: bool = False,
    allow_llm_override: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> ValidationReport:
    """验证 CaseSpec；高自治模式可由 LLM 覆盖可解释的物理告警。"""

    local_report = validate_case_spec(case_spec)
    report = local_report.model_copy(update={"local_is_valid": local_report.is_valid})
    if local_report.is_valid:
        return report

    evidence = retrieve_for_validation_failure(local_report, case_spec, retriever)
    evidence_ids = [item.evidence_id for item in evidence]
    report = report.model_copy(update={"evidence_ids": evidence_ids})
    if not (use_llm and allow_llm_override):
        return report

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请判断本地 Validator 失败是否可以高置信放行。\n"
                f"可放行 failure_modes: {[mode.value for mode in OVERRIDABLE_VALIDATION_MODES]}\n"
                f"case_spec: {case_spec.model_dump(mode='json')}\n"
                f"local_validation_report: {local_report.model_dump(mode='json')}\n"
                f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                "若放行，请把被覆盖的 failure_modes 写入 overridden_failure_modes。"
            )
        ),
    ]
    decision = try_invoke_structured(
        agent="validator",
        messages=messages,
        output_model=AgentDecision,
        provider=llm_provider,
        llm=llm,
        use_llm=use_llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    if not isinstance(decision, AgentDecision):
        return report

    decision = decision.model_copy(
        update={
            "case_id": case_spec.case_id,
            "target_agent": "validator",
            "evidence_ids": [item for item in decision.evidence_ids if item in set(evidence_ids)] or evidence_ids,
        }
    )
    allowed, overridden, remaining = decision_allows_override(
        decision,
        failure_modes=local_report.failure_modes,
        allowed_modes=OVERRIDABLE_VALIDATION_MODES,
    )
    enrich_latest_trace(
        trace,
        agent="validator",
        **decision_trace_payload(decision, overridden=overridden, evidence_ids=decision.evidence_ids),
    )
    if not allowed:
        return report.model_copy(update={"llm_decision": decision, "overridden_failure_modes": overridden})

    messages_out = list(local_report.messages)
    messages_out.append("LLM 高置信放行本地物理校验告警。")
    return report.model_copy(
        update={
            "is_valid": True,
            "failure_modes": remaining,
            "severity": Severity.MINOR,
            "messages": messages_out,
            "llm_decision": decision,
            "overridden_failure_modes": overridden,
        }
    )
