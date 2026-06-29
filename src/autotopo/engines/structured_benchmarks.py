"""二维结构化 benchmark case 生成。"""

from __future__ import annotations

from typing import Any

import numpy as np

from autotopo.schemas import BenchmarkType, CaseSpec


def l_shape_passive_void_mask(nelx: int, nely: int) -> list[list[bool]]:
    """L 型梁默认挖掉右上象限。"""

    mask = np.zeros((nely, nelx), dtype=bool)
    mask[: nely // 2, nelx // 2 :] = True
    return mask.tolist()


def default_case_spec(
    benchmark_type: BenchmarkType | str,
    *,
    variant: str = "clear",
    quick: bool = False,
    case_id: str | None = None,
    natural_language: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> CaseSpec:
    """返回可直接求解的默认 CaseSpec。"""

    benchmark = BenchmarkType(benchmark_type)
    if quick and benchmark != BenchmarkType.L_SHAPE:
        nelx, nely, max_iter = 12, 4, 10
    elif quick and benchmark == BenchmarkType.L_SHAPE:
        nelx, nely, max_iter = 10, 10, 10
    elif benchmark == BenchmarkType.L_SHAPE:
        nelx, nely, max_iter = 40, 40, 100
    else:
        nelx, nely, max_iter = 90, 30, 100

    descriptions = {
        BenchmarkType.CANTILEVER: "悬臂梁：左边固定，右端中点向下集中力。",
        BenchmarkType.MBB: "半 MBB 梁：左侧对称约束，右下竖向支撑，左上向下集中力。",
        BenchmarkType.L_SHAPE: "L 型梁：右上象限为空洞，左边固定，右下端向下集中力。",
    }
    if variant == "fuzzy":
        descriptions = {
            BenchmarkType.CANTILEVER: "做一个常见悬臂梁拓扑优化，右侧受向下力。",
            BenchmarkType.MBB: "做一个 MBB 类梁，支撑和载荷按标准模板补齐。",
            BenchmarkType.L_SHAPE: "做一个带缺口的 L 型梁，边界和载荷用默认工程模板。",
        }

    data: dict[str, Any] = {
        "case_id": case_id or f"{benchmark.value}_{variant}",
        "benchmark_type": benchmark,
        "natural_language": natural_language or descriptions[benchmark],
        "variant": variant,
        "nelx": nelx,
        "nely": nely,
        "volume_fraction": 0.5 if benchmark != BenchmarkType.CANTILEVER else 0.4,
        "penal": 3.0,
        "rmin": 1.5,
        "max_iter": max_iter,
        "tol": 2e-2 if not quick else 1e-2,
        "optimizer": "MMA",
    }
    if overrides:
        data.update(overrides)

    spec = CaseSpec(**data)
    return spec.model_copy(update={"problem": case_to_problem(spec)})


def case_to_problem(case_spec: CaseSpec) -> dict[str, Any]:
    """将 CaseSpec 转成 PythonSimpMMAEngine 可读问题字典。"""

    common = {
        "description": case_spec.natural_language,
        "domain": {
            "nelx": case_spec.nelx,
            "nely": case_spec.nely,
            "width": float(case_spec.nelx),
            "height": float(case_spec.nely),
        },
        "material": case_spec.material.model_dump(mode="json"),
        "objective": "minimize_compliance",
        "constraints": [
            {
                "type": "volume_fraction",
                "value": case_spec.volume_fraction,
                "description": "体积分数约束",
            }
        ],
        "parameters": {
            "penal": case_spec.penal,
            "rmin": case_spec.rmin,
            "max_iter": case_spec.max_iter,
            "tol": case_spec.tol,
            "optimizer": case_spec.optimizer,
            "ft": 1,
        },
    }

    if case_spec.benchmark_type == BenchmarkType.CANTILEVER:
        common["boundary_conditions"] = [{"type": "fixed", "location": "left_edge"}]
        common["loads"] = [
            {"type": "point_force", "location": "right_center", "magnitude": 1.0, "direction": [0, -1]}
        ]
    elif case_spec.benchmark_type == BenchmarkType.MBB:
        common["boundary_conditions"] = [
            {"type": "fixed_x", "location": "left_edge"},
            {"type": "fixed_y", "location": "bottom_right"},
        ]
        common["loads"] = [
            {"type": "point_force", "location": "top_left", "magnitude": 1.0, "direction": [0, -1]}
        ]
    elif case_spec.benchmark_type == BenchmarkType.L_SHAPE:
        common["domain"]["passive_void_mask"] = l_shape_passive_void_mask(case_spec.nelx, case_spec.nely)
        common["boundary_conditions"] = [{"type": "fixed", "location": "left_edge"}]
        common["loads"] = [
            {"type": "point_force", "location": "bottom_right", "magnitude": 1.0, "direction": [0, -1]}
        ]
    else:
        raise ValueError(f"不支持的 benchmark: {case_spec.benchmark_type}")

    return common


def minimal_benchmark_cases(*, quick: bool = False) -> list[CaseSpec]:
    """返回 3 类 benchmark × clear/fuzzy 的最小 case 矩阵。"""

    cases: list[CaseSpec] = []
    for benchmark in (BenchmarkType.MBB, BenchmarkType.CANTILEVER, BenchmarkType.L_SHAPE):
        for variant in ("clear", "fuzzy"):
            cases.append(default_case_spec(benchmark, variant=variant, quick=quick))
    return cases
