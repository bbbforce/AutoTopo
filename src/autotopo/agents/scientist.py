"""Scientist agent：把自然语言和结构化参数合并成 CaseSpec。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from autotopo.agents.llm_utils import AgentTrace, try_invoke_structured
from autotopo.engines.structured_benchmarks import default_case_spec, case_to_problem
from autotopo.schemas import BenchmarkType, CaseSpec, MaterialSpec


SCIENTIST_SYSTEM_PROMPT = """\
You are the Scientist agent for AutoTopo's minimal research workflow.

Classify the user request into one supported structured topology optimization
benchmark and propose only conservative numeric parameters. Supported benchmark_type
values are: cantilever, mbb, l_shape. The downstream solver will rebuild the
canonical benchmark problem locally, so do not invent custom boundary conditions
or custom simulator code.

Return JSON matching the requested schema. Use null for unknown optional fields.
"""


class CaseSpecDraft(BaseModel):
    """LLM draft for Scientist; deterministic templates fill the final CaseSpec."""

    case_id: str | None = None
    benchmark_type: BenchmarkType
    variant: str = "clear"
    nelx: int | None = Field(default=None, ge=2, le=240)
    nely: int | None = Field(default=None, ge=2, le=160)
    volume_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    penal: float | None = Field(default=None, gt=0.0, le=8.0)
    rmin: float | None = Field(default=None, gt=0.0, le=10.0)
    max_iter: int | None = Field(default=None, ge=1, le=300)
    tol: float | None = Field(default=None, ge=1e-8, le=1e-1)
    optimizer: str | None = None
    material: MaterialSpec | None = None


def infer_benchmark_type(text: str, structured_params: dict[str, Any] | None = None) -> BenchmarkType:
    """从结构化参数优先、自然语言补充推断 benchmark 类型。"""

    params = structured_params or {}
    if params.get("benchmark_type"):
        return BenchmarkType(params["benchmark_type"])
    lower = text.lower()
    if "l-shape" in lower or "l shape" in lower or "l型" in lower or "l 型" in lower:
        return BenchmarkType.L_SHAPE
    if "mbb" in lower:
        return BenchmarkType.MBB
    if "cantilever" in lower or "悬臂" in lower:
        return BenchmarkType.CANTILEVER
    return BenchmarkType.CANTILEVER


def _case_spec_from_parts(
    natural_language: str,
    *,
    benchmark: BenchmarkType,
    variant: str,
    case_id: str | None,
    quick: bool,
    structured_params: dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
) -> CaseSpec:
    params = dict(structured_params or {})
    if params.get("benchmark_type"):
        benchmark = BenchmarkType(params["benchmark_type"])
    if "variant" in params:
        variant = str(params["variant"])
    if "case_id" in params:
        case_id = params["case_id"]

    for key in ("benchmark_type", "variant", "case_id", "natural_language"):
        params.pop(key, None)

    merged_overrides = dict(overrides or {})
    merged_overrides.update(params)
    spec = default_case_spec(
        benchmark,
        variant=variant,
        quick=quick,
        case_id=case_id,
        natural_language=natural_language,
        overrides=merged_overrides,
    )
    spec = spec.model_copy(update={"structured_params": structured_params or {}})
    return spec.model_copy(update={"problem": case_to_problem(spec)})


def _draft_to_case_spec(
    draft: CaseSpecDraft,
    natural_language: str,
    *,
    structured_params: dict[str, Any] | None,
    quick: bool,
) -> CaseSpec:
    data = draft.model_dump(mode="python", exclude_none=True)
    benchmark = BenchmarkType(data.pop("benchmark_type"))
    variant = str(data.pop("variant", "clear"))
    case_id = data.pop("case_id", None)
    return _case_spec_from_parts(
        natural_language,
        benchmark=benchmark,
        variant=variant,
        case_id=case_id,
        quick=quick,
        structured_params=structured_params,
        overrides=data,
    )


def _build_case_spec_deterministic(
    natural_language: str,
    *,
    structured_params: dict[str, Any] | None = None,
    quick: bool = False,
) -> CaseSpec:
    benchmark = infer_benchmark_type(natural_language, structured_params)
    params = structured_params or {}
    return _case_spec_from_parts(
        natural_language,
        benchmark=benchmark,
        variant=str(params.get("variant", "clear")),
        case_id=params.get("case_id"),
        quick=quick,
        structured_params=structured_params,
    )


def build_case_spec(
    natural_language: str,
    *,
    structured_params: dict[str, Any] | None = None,
    quick: bool = False,
    use_llm: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> CaseSpec:
    """构造 CaseSpec，结构化参数覆盖自然语言推断结果。"""

    messages = [
        SystemMessage(content=SCIENTIST_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Build a CaseSpecDraft for this minimal benchmark request.\n"
                f"natural_language: {natural_language}\n"
                f"structured_params: {structured_params or {}}\n"
                f"quick_mode: {quick}\n"
                "Respect structured_params over natural_language when they conflict."
            )
        ),
    ]
    draft = try_invoke_structured(
        agent="scientist",
        messages=messages,
        output_model=CaseSpecDraft,
        provider=llm_provider,
        llm=llm,
        use_llm=use_llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    if isinstance(draft, CaseSpecDraft):
        return _draft_to_case_spec(
            draft,
            natural_language,
            structured_params=structured_params,
            quick=quick,
        )

    return _build_case_spec_deterministic(
        natural_language,
        structured_params=structured_params,
        quick=quick,
    )
