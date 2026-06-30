"""Reviewer agent：运行失败后的诊断与修复计划。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autotopo.agents.llm_utils import AgentTrace, try_invoke_structured
from autotopo.diagnostics.failure_modes import diagnose_execution_report
from autotopo.diagnostics.repair_rules import build_repair_plan
from autotopo.rag.corrective_rag import retrieve_for_execution_failure
from autotopo.rag.retriever import LocalRetriever
from autotopo.schemas import CaseSpec, ExecutionReport, FailureDiagnosis, FailureMode, RepairPlan, RetrievedEvidence


REVIEWER_SYSTEM_PROMPT = """\
你是 AutoTopo research_graph 的 Reviewer agent。

## 职责
诊断失败的 PythonSimpMMAEngine 或 generated_script 执行结果，输出结构化 FailureDiagnosis。
你负责解释失败原因和给出保守修复建议；真正的 RepairPlan 会由本地 build_repair_plan 规则生成并限制。

## 输入上下文
- allowed_failure_modes：允许使用的 FailureMode 枚举全集。
- case_spec：当前规范化 CaseSpec。
- execution_report：Executor 捕获的失败报告，包含异常、traceback、stdout/stderr 和产物路径。
- deterministic_diagnosis：本地基线诊断，是默认可信起点。
- retrieved_evidence：与执行失败相关的本地 RAG 证据。

## 输出格式
只输出一个严格匹配 FailureDiagnosis schema 的 JSON 对象，不要 Markdown、解释文字或额外字段。
字段要求：
- case_id 必须与 case_spec.case_id 一致。
- has_failure 通常为 true，除非 execution_report 明确不是失败。
- failure_modes 只能来自 allowed_failure_modes。
- severity 使用 minor、moderate、severe。
- likely_causes 写具体原因，优先对应异常文本、traceback 和证据。
- repair_suggestions 写保守建议；生成脚本失败时可明确写入 "fallback_template" 或 "regenerate_code"。
- auto_repair_allowed 只有在本地有界修复明显安全时才为 true。
- evidence_ids 只能引用 retrieved_evidence 中真实存在的 evidence_id。

## 关键判断标准
- deterministic_diagnosis 是基线；LLM 可以补充更清晰原因，但不要无证据推翻。
- 区分依赖缺失、shape mismatch、奇异刚度矩阵、非法边界条件、数值发散、生成脚本契约错误等失败。
- 对生成脚本错误，优先建议回退模板；只有错误明显可由重新生成脚本解决时才建议 regenerate_code。

## 硬性边界
- 不得提出任意代码执行、网络访问、外部依赖安装或越界文件操作。
- 不得直接给出 RepairPlan 或修改 CaseSpec。
- 不得使用 allowed_failure_modes 之外的 failure mode。

## 失败/不确定时如何输出
如果证据不足，沿用 deterministic_diagnosis 的 failure_modes，severity 取保守级别，auto_repair_allowed=false，并在 likely_causes 中说明信息不足。
"""


def _normalize_llm_diagnosis(
    diagnosis: FailureDiagnosis,
    *,
    case_spec: CaseSpec,
    deterministic: FailureDiagnosis,
    evidence_ids: list[str],
) -> FailureDiagnosis:
    if not diagnosis.has_failure or not diagnosis.failure_modes:
        return deterministic
    return diagnosis.model_copy(
        update={
            "case_id": case_spec.case_id,
            "has_failure": True,
            "likely_causes": diagnosis.likely_causes or deterministic.likely_causes,
            "repair_suggestions": diagnosis.repair_suggestions or deterministic.repair_suggestions,
            "evidence_ids": evidence_ids,
        }
    )


def review_execution_failure(
    case_spec: CaseSpec,
    report: ExecutionReport,
    *,
    repair_iteration: int,
    max_repair_rounds: int,
    retriever: LocalRetriever | None = None,
    use_llm: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> tuple[FailureDiagnosis, RepairPlan, list[RetrievedEvidence]]:
    """诊断执行失败并生成有界修复计划。"""

    evidence = retrieve_for_execution_failure(report, retriever, case_spec)
    evidence_ids = [item.evidence_id for item in evidence]
    deterministic = diagnose_execution_report(report).model_copy(update={"evidence_ids": evidence_ids})
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请审阅这次失败的拓扑优化执行。\n"
                f"allowed_failure_modes: {[mode.value for mode in FailureMode]}\n"
                f"case_spec: {case_spec.model_dump(mode='json')}\n"
                f"execution_report: {report.model_dump(mode='json')}\n"
                f"deterministic_diagnosis: {deterministic.model_dump(mode='json')}\n"
                f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                "请返回 FailureDiagnosis，并保持 auto_repair_allowed 保守。"
            )
        ),
    ]
    llm_diagnosis = try_invoke_structured(
        agent="reviewer",
        messages=messages,
        output_model=FailureDiagnosis,
        provider=llm_provider,
        llm=llm,
        use_llm=use_llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    diagnosis = (
        _normalize_llm_diagnosis(
            llm_diagnosis,
            case_spec=case_spec,
            deterministic=deterministic,
            evidence_ids=evidence_ids,
        )
        if isinstance(llm_diagnosis, FailureDiagnosis)
        else deterministic
    )
    repair_plan = build_repair_plan(
        case_spec,
        diagnosis.failure_modes,
        repair_iteration=repair_iteration,
        max_repair_rounds=max_repair_rounds,
        evidence_ids=diagnosis.evidence_ids,
    )
    return diagnosis, repair_plan, evidence
