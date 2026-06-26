"""Shared helpers for optional LLM-backed research agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from autotopo.llm_factory import clean_json_string
from autotopo.schemas import AgentDecision, FailureMode


AgentTrace = list[dict[str, Any]]
DECISION_CONFIDENCE_THRESHOLD = 0.70
PASS_DECISIONS = {"pass", "allow", "approve", "accept", "override", "continue"}


def llm_enabled(*, use_llm: bool = False, provider: str | None = None, llm: Any = None) -> bool:
    """Return whether an agent should attempt an LLM call."""

    return bool(use_llm or provider or llm is not None)


def append_trace(
    trace: AgentTrace | None,
    *,
    agent: str,
    enabled: bool,
    used_llm: bool,
    fallback_reason: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """追加一条紧凑的 LLM/fallback 审计记录。"""

    if trace is None or not enabled:
        return
    item = {
        "agent": agent,
        "enabled": enabled,
        "used_llm": used_llm,
        "fallback_reason": fallback_reason,
    }
    if extra:
        item.update(extra)
    trace.append(item)


def enrich_latest_trace(trace: AgentTrace | None, *, agent: str, **extra: Any) -> None:
    """把结构化裁决、证据和沙箱信息补到最近一次 agent trace。"""

    if trace is None:
        return
    for item in reversed(trace):
        if item.get("agent") == agent:
            item.update(extra)
            return


def _mode_values(modes: list[FailureMode]) -> list[str]:
    return [mode.value for mode in modes]


def decision_allows_override(
    decision: AgentDecision | None,
    *,
    failure_modes: list[FailureMode],
    allowed_modes: set[FailureMode],
    threshold: float = DECISION_CONFIDENCE_THRESHOLD,
) -> tuple[bool, list[FailureMode], list[FailureMode]]:
    """判断 LLM 裁决是否足够可信，并返回被覆盖和剩余失败模式。"""

    if decision is None or decision.confidence < threshold:
        return False, [], failure_modes
    if decision.decision.lower() not in PASS_DECISIONS:
        return False, [], failure_modes

    requested = [mode for mode in decision.overridden_failure_modes if mode in failure_modes]
    if not requested:
        requested = [mode for mode in failure_modes if mode in allowed_modes]
    overridden = [mode for mode in requested if mode in allowed_modes]
    remaining = [mode for mode in failure_modes if mode not in overridden]
    return bool(overridden) and not remaining, overridden, remaining


def decision_trace_payload(
    decision: AgentDecision | None,
    *,
    overridden: list[FailureMode] | None = None,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """把 LLM 裁决压缩成 trace 友好的 JSON 字段。"""

    if decision is None:
        return {}
    return {
        "decision": decision.decision,
        "confidence": decision.confidence,
        "reasons": decision.reasons,
        "evidence_ids": evidence_ids if evidence_ids is not None else decision.evidence_ids,
        "overridden_failure_modes": _mode_values(overridden or decision.overridden_failure_modes),
        "action": decision.action,
    }


def _coerce_structured_result(result: Any, output_model: type[BaseModel]) -> BaseModel:
    if isinstance(result, output_model):
        return result
    if isinstance(result, dict):
        return output_model.model_validate(result)
    if hasattr(result, "content"):
        cleaned = clean_json_string(str(result.content))
        return output_model.model_validate_json(cleaned)
    return output_model.model_validate(result)


def try_invoke_structured(
    *,
    agent: str,
    messages: list[Any],
    output_model: type[BaseModel],
    provider: str | None = None,
    llm: Any = None,
    use_llm: bool = False,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> BaseModel | None:
    """Invoke an optional structured LLM and return None on any fallback-worthy error."""

    enabled = llm_enabled(use_llm=use_llm, provider=provider, llm=llm)
    if not enabled:
        return None

    try:
        runnable = llm
        if runnable is None:
            from autotopo.llm_factory import get_llm

            runnable = get_llm(
                provider=provider,
                structured_output=output_model,
                **(llm_overrides or {}),
            )
        result = runnable.invoke(messages)
        coerced = _coerce_structured_result(result, output_model)
    except Exception as exc:
        append_trace(
            trace,
            agent=agent,
            enabled=enabled,
            used_llm=False,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
        return None

    append_trace(trace, agent=agent, enabled=enabled, used_llm=True)
    return coerced
