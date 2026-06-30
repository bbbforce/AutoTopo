"""Coder agent：选择模板，或在高自治模式下生成独立求解脚本。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from autotopo.agents.llm_utils import AgentTrace, decision_trace_payload, enrich_latest_trace, try_invoke_structured
from autotopo.schemas import AgentAuthority, AgentDecision, CaseSpec, CodePlan, RetrievedEvidence


CODER_SYSTEM_PROMPT = """\
你是 AutoTopo research_graph 的 Coder agent。

## 职责
在默认路径中确认使用本地模板执行；只有在 llm_primary 且 allow_generated_code=true 时，才可以生成一个独立 Python 求解脚本草案。
你的输出是 GeneratedSolverDraft，后续 Executor 会做静态检查、沙箱式执行和 ExecutionReport 校验。

## 输入上下文
- case_spec：当前规范化 CaseSpec。
- code_plan：Planner 给出的执行计划。
- retrieved_evidence：本地 RAG 证据，可用于解释生成或回退理由。

## 输出格式
只输出一个严格匹配 GeneratedSolverDraft schema 的 JSON 对象，不要在 JSON 外添加 Markdown 或解释文字。
字段要求：
- decision：AgentDecision；target_agent 应为 "coder"。
- code：完整 Python 源码字符串；如果不安全或不需要生成，填空字符串。
- steps：生成、接口、产物和安全假设的简短步骤列表。
- evidence_ids：只能引用 retrieved_evidence 中真实存在的 evidence_id。

## 关键判断标准
- 只有确认脚本能遵守 JSON 脚本接口时，decision.decision 才能使用 "generate"。
- 独立脚本必须接收 --case-spec、--code-plan、--output-dir 三个参数。
- 脚本必须只在 output-dir 内写 execution_report.json，并使其匹配 AutoTopo 的 ExecutionReport JSON 结构。
- 脚本应尽量复用输入 JSON 和安全的标准库能力，不要绕过 workflow 的产物契约。

## 硬性边界
- 禁止网络访问、子进程、动态执行、交互输入、任意导入、任意文件读取写入和访问 output-dir 之外的路径。
- 禁止使用 eval、exec、compile、open、__import__、input、os.system、subprocess、requests、socket 等危险能力。
- 不得改变 code_plan 的 case_id、method 或输出目录契约。

## 失败/不确定时如何输出
如果无法高置信生成安全脚本，返回 decision.decision="fallback_template"，code=""，并在 reasons/steps 中说明回退本地模板执行。
"""


class GeneratedSolverDraft(BaseModel):
    """Coder LLM 生成的独立脚本草稿。"""

    decision: AgentDecision = Field(default_factory=lambda: AgentDecision(target_agent="coder"))
    code: str = ""
    steps: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


def _extract_code(code: str) -> str:
    if "```python" in code:
        return code.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in code:
        return code.split("```", 1)[1].split("```", 1)[0].strip()
    return code.strip()


def _template_plan(code_plan: CodePlan, *, reason: str = "确认不生成自由形式求解器代码") -> CodePlan:
    steps = list(code_plan.steps)
    if reason and reason not in steps:
        steps.append(reason)
    return code_plan.model_copy(
        update={
            "allow_generated_code": False,
            "execution_mode": "template",
            "generated_code_path": "",
            "generated_code_manifest_path": "",
            "steps": steps,
        }
    )


def _write_generated_artifacts(
    *,
    output_dir: Path,
    case_spec: CaseSpec,
    code_plan: CodePlan,
    draft: GeneratedSolverDraft,
) -> tuple[Path, Path]:
    code_path = output_dir / "generated_solver.py"
    manifest_path = output_dir / "generated_code_manifest.json"
    code = _extract_code(draft.code)
    code_path.write_text(code + "\n", encoding="utf-8")
    manifest = {
        "case_id": case_spec.case_id,
        "code_path": str(code_path),
        "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "decision": draft.decision.model_dump(mode="json"),
        "steps": draft.steps,
        "evidence_ids": draft.evidence_ids,
        "contract": "json_script_v1",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return code_path, manifest_path


def select_or_generate_code(
    code_plan: CodePlan,
    *,
    case_spec: CaseSpec | None = None,
    evidence: list[RetrievedEvidence] | None = None,
    output_dir: str | Path | None = None,
    agent_authority: AgentAuthority | str = AgentAuthority.DETERMINISTIC,
    allow_generated_code: bool = False,
    use_llm: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> CodePlan:
    """返回可执行计划；只有高自治模式会落盘 LLM 生成脚本。"""

    authority = AgentAuthority(agent_authority)
    if (
        authority != AgentAuthority.LLM_PRIMARY
        or not allow_generated_code
        or not use_llm
        or case_spec is None
        or output_dir is None
    ):
        return _template_plan(code_plan)

    evidence = evidence or []
    evidence_ids = {item.evidence_id for item in evidence}
    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请为这个 research_graph case 生成 JSON 脚本接口求解器。\n"
                f"case_spec: {case_spec.model_dump(mode='json')}\n"
                f"code_plan: {code_plan.model_dump(mode='json')}\n"
                f"retrieved_evidence: {[item.model_dump(mode='json') for item in evidence]}\n"
                "返回 GeneratedSolverDraft。"
            )
        ),
    ]
    draft = try_invoke_structured(
        agent="coder",
        messages=messages,
        output_model=GeneratedSolverDraft,
        provider=llm_provider,
        llm=llm,
        use_llm=use_llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    if not isinstance(draft, GeneratedSolverDraft) or not draft.code.strip():
        enrich_latest_trace(trace, agent="coder", action="fallback_template", fallback_reason="未得到可执行代码草稿")
        return _template_plan(code_plan, reason="LLM 未生成脚本，回退模板执行")

    safe_evidence_ids = [item for item in draft.evidence_ids if item in evidence_ids]
    decision = draft.decision.model_copy(
        update={
            "case_id": case_spec.case_id,
            "target_agent": "coder",
            "evidence_ids": safe_evidence_ids or draft.decision.evidence_ids,
        }
    )
    draft = draft.model_copy(update={"decision": decision, "evidence_ids": safe_evidence_ids})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    code_path, manifest_path = _write_generated_artifacts(
        output_dir=out,
        case_spec=case_spec,
        code_plan=code_plan,
        draft=draft,
    )
    enrich_latest_trace(
        trace,
        agent="coder",
        **decision_trace_payload(decision, evidence_ids=safe_evidence_ids),
        generated_code_path=str(code_path),
        generated_code_manifest_path=str(manifest_path),
    )
    steps = draft.steps or code_plan.steps
    return code_plan.model_copy(
        update={
            "allow_generated_code": True,
            "execution_mode": "generated_script",
            "generated_code_path": str(code_path),
            "generated_code_manifest_path": str(manifest_path),
            "steps": steps,
            "evidence_ids": safe_evidence_ids or code_plan.evidence_ids,
            "generation_decision": decision,
        }
    )
