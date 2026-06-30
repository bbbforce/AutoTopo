"""Scientist agent：把自然语言和结构化参数合并成 CaseSpec。"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from autotopo.agents.llm_utils import AgentTrace, try_invoke_structured
from autotopo.engines.structured_benchmarks import default_case_spec, case_to_problem
from autotopo.schemas import BenchmarkType, CaseSpec, MaterialSpec


SCIENTIST_SYSTEM_PROMPT = """\
你是 AutoTopo research_graph 的 Scientist agent。

## 职责
把用户自然语言和 structured_params 合并理解为一个最小研究 workflow 支持的拓扑优化基准问题草案。
你的输出只是 CaseSpecDraft，后续系统会用本地 canonical benchmark 模板重建最终 CaseSpec 和 problem。

## 输入上下文
- natural_language：用户的原始需求，可能包含 benchmark 类型、网格尺寸、体积分数、SIMP 参数和迭代参数。
- structured_params：调用方显式传入的结构化参数，优先级高于自然语言。
- quick_mode：快速实验开关；你只需理解上下文，不要自行改写 quick 模板。

## 输出格式
只输出一个严格匹配 CaseSpecDraft schema 的 JSON 对象，不要 Markdown、解释文字或额外字段。
字段要求：
- case_id：可为用户给定或简短稳定标识；未知时用 null。
- benchmark_type：只能是 "cantilever"、"mbb"、"l_shape"。
- variant：默认 "clear"，除非用户明确指定。
- nelx、nely、volume_fraction、penal、rmin、max_iter、tol、optimizer、material：只在用户或 structured_params 明确给出，或能高置信保守推断时填写；未知可选字段用 null。

## 关键判断标准
- structured_params 与 natural_language 冲突时，以 structured_params 为准。
- 只选择最接近用户意图的已支持 benchmark，不要创造新 benchmark。
- 数值参数必须保守、物理合理，并落在 CaseSpecDraft 的 Pydantic 约束范围内。
- 用户明确写出的网格、体积分数、罚因子、过滤半径、迭代上限和容差要尽量保留。

## 硬性边界
- 不得发明自定义边界条件、载荷路径、非设计域、求解器、优化器流程或 simulator code。
- 不得输出最终 CaseSpec、CodePlan、ExecutionReport 或 Python 代码。
- 不得把不确定信息伪装成确定参数。

## 失败/不确定时如何输出
如果无法判断类型，选择最保守的 cantilever；如果无法确定某个可选参数，填 null。
如果用户请求超出最小 benchmark 范围，仍映射到最接近的支持类型，并保持参数保守。
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


def _first_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group("value"))
    return None


def _extract_explicit_params(text: str) -> dict[str, Any]:
    """抽取用户在自然语言里明写的安全标量参数。"""

    normalized = (
        text.replace("×", "x")
        .replace("X", "x")
        .replace("：", ":")
        .replace("，", ",")
        .replace("＝", "=")
    )
    overrides: dict[str, Any] = {}

    size_match = re.search(
        r"(?:设计域|域|网格|尺寸|分辨率)[^0-9]{0,12}(?P<nelx>\d+)\s*[x*]\s*(?P<nely>\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not size_match:
        size_match = re.search(r"(?P<nelx>\d+)\s*x\s*(?P<nely>\d+)", normalized, flags=re.IGNORECASE)
    if size_match:
        overrides["nelx"] = int(size_match.group("nelx"))
        overrides["nely"] = int(size_match.group("nely"))

    value = _first_float(
        [
            r"(?:体积分数|体积约束|volume[_\s-]?fraction|volfrac)\s*(?:=|为|:)?\s*(?P<value>\d+(?:\.\d+)?)",
            r"(?:体积分数约束)\s*(?:=|为|:)?\s*(?P<value>\d+(?:\.\d+)?)",
        ],
        normalized,
    )
    if value is not None:
        overrides["volume_fraction"] = value

    value = _first_float(
        [
            r"(?:惩罚因子|罚因子)\s*(?:p|penal)?\s*(?:=|为|:)?\s*(?P<value>\d+(?:\.\d+)?)",
            r"\b(?:penal|p)\s*(?:=|:)\s*(?P<value>\d+(?:\.\d+)?)",
        ],
        normalized,
    )
    if value is not None:
        overrides["penal"] = value

    value = _first_float(
        [
            r"(?:过滤半径|滤波半径)\s*(?:rmin|r)?\s*(?:=|为|:)?\s*(?P<value>\d+(?:\.\d+)?)",
            r"\b(?:rmin|r)\s*(?:=|:)\s*(?P<value>\d+(?:\.\d+)?)",
        ],
        normalized,
    )
    if value is not None:
        overrides["rmin"] = value

    value = _first_float(
        [
            r"(?:最大迭代次数|迭代上限|max[_\s-]?iter)\s*(?:=|为|:)?\s*(?P<value>\d+(?:\.\d+)?)",
        ],
        normalized,
    )
    if value is not None:
        overrides["max_iter"] = int(value)

    value = _first_float(
        [
            r"(?:收敛容差|容差|tol)\s*(?:=|为|:)?\s*(?P<value>\d+(?:\.\d+)?)",
        ],
        normalized,
    )
    if value is not None:
        overrides["tol"] = value

    return overrides


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
    merged_overrides.update(_extract_explicit_params(natural_language))
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

    spec, _causality = build_case_spec_with_causality(
        natural_language,
        structured_params=structured_params,
        quick=quick,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm=llm,
        llm_overrides=llm_overrides,
        trace=trace,
    )
    return spec


def build_case_spec_with_causality(
    natural_language: str,
    *,
    structured_params: dict[str, Any] | None = None,
    quick: bool = False,
    use_llm: bool = False,
    llm_provider: str | None = None,
    llm: Any = None,
    llm_overrides: dict[str, Any] | None = None,
    trace: AgentTrace | None = None,
) -> tuple[CaseSpec, dict[str, Any]]:
    """构造 CaseSpec，并返回 raw/normalized causality 元数据。"""

    messages = [
        SystemMessage(content=SCIENTIST_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请为这个最小 benchmark 请求构建 CaseSpecDraft。\n"
                f"natural_language: {natural_language}\n"
                f"structured_params: {structured_params or {}}\n"
                f"quick_mode: {quick}\n"
                "当 structured_params 与 natural_language 冲突时，必须优先遵守 structured_params。"
            )
        ),
    ]
    raw_layer: dict[str, Any] = {
        "input_type": "natural_language",
        "natural_language": natural_language,
        "structured_params": structured_params or {},
        "quick": quick,
        "llm_requested": bool(use_llm or llm_provider or llm is not None),
        "source": "deterministic",
        "case_spec_draft": None,
    }
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
        raw_layer["source"] = "llm"
        raw_layer["case_spec_draft"] = draft.model_dump(mode="json")
        spec = _draft_to_case_spec(
            draft,
            natural_language,
            structured_params=structured_params,
            quick=quick,
        )
    else:
        if raw_layer["llm_requested"]:
            raw_layer["source"] = "deterministic_fallback"
        spec = _build_case_spec_deterministic(
            natural_language,
            structured_params=structured_params,
            quick=quick,
        )

    spec_dump = spec.model_dump(mode="json")
    return spec, {
        "raw": raw_layer,
        "normalized": {
            "artifact": "case_spec.json",
            "case_spec": spec_dump,
        },
        "repair": {
            "artifact": "case_spec_repaired.json",
            "applications": [],
            "final_case_spec": spec_dump,
        },
    }
