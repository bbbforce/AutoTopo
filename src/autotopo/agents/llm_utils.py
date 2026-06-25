"""Shared helpers for optional LLM-backed research agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from autotopo.llm_factory import clean_json_string


AgentTrace = list[dict[str, Any]]


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
) -> None:
    """Append a compact per-agent LLM/fallback trace entry."""

    if trace is None or not enabled:
        return
    trace.append(
        {
            "agent": agent,
            "enabled": enabled,
            "used_llm": used_llm,
            "fallback_reason": fallback_reason,
        }
    )


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
