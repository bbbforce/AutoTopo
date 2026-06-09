"""仿真调度节点。

负责初始化 FEniCS + dolfin-adjoint 引擎、执行优化迭代、生成结果图。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from autotopo.engines.base import TopoEngine
from autotopo.state import AutoTopoState


def _load_config() -> dict:
    for p in [Path("config/settings.yaml"), Path(__file__).resolve().parents[3] / "config" / "settings.yaml"]:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    return {}


def _get_engine() -> TopoEngine:
    """根据配置实例化仿真引擎。"""
    config = _load_config()
    backend = config.get("engine", {}).get("backend", "dolfin_adjoint")

    if backend == "dolfin_adjoint":
        from autotopo.engines.dolfin_adjoint_engine import DolfinAdjointEngine
        return DolfinAdjointEngine()
    else:
        raise ValueError(f"不支持的引擎后端: {backend}")


def _volume_constraint_value(problem: dict[str, Any]) -> float | None:
    """从问题约束中读取体积分数。"""
    for constraint in problem.get("constraints", []):
        if constraint.get("type") == "volume_fraction":
            return constraint.get("value")
    return None


def _sync_volume_constraint(problem: dict[str, Any], volfrac: float) -> None:
    """将 parameters.volfrac 同步回体积分数约束，供容器求解器读取。"""
    constraints = list(problem.get("constraints", []))
    for constraint in constraints:
        if constraint.get("type") == "volume_fraction":
            constraint["value"] = volfrac
            problem["constraints"] = constraints
            return

    constraints.append({
        "type": "volume_fraction",
        "value": volfrac,
        "description": "体积分数约束",
    })
    problem["constraints"] = constraints


def _merge_parameters(
    problem: dict[str, Any],
    defaults: dict[str, Any],
    current_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """按反馈参数 > 体积分数约束 > 问题参数 > 配置默认值合并求解参数。"""
    params = dict(defaults)
    params.update(problem.get("parameters", {}))

    constraint_volfrac = _volume_constraint_value(problem)
    if constraint_volfrac is not None:
        params["volfrac"] = constraint_volfrac

    if current_params:
        params.update(current_params)

    return params


def run_simulation(state: AutoTopoState) -> dict[str, Any]:
    """仿真节点：初始化引擎 → 执行优化 → 导出结果图。"""
    config = _load_config()
    output_cfg = config.get("output", {})
    engine_defaults = config.get("engine", {}).get("default_params", {})

    problem = deepcopy(state["problem_definition"])
    params = _merge_parameters(problem, engine_defaults, state.get("current_params"))
    problem["parameters"] = params
    _sync_volume_constraint(problem, params.get("volfrac", 0.5))

    # 每轮从均匀密度场重新开始优化
    engine = _get_engine()
    engine.setup(problem)

    # 执行优化
    result = engine.optimize(
        max_iter=params.get("max_iter", 200),
        tol=params.get("tol", 1e-6),
        penal=params.get("penal", 3.0),
        rmin=params.get("rmin", 0.05),
        volfrac=params.get("volfrac", 0.5),
    )

    # 导出结果图
    output_dir = Path(state.get("output_path", output_cfg.get("dir", "./output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    iteration = state.get("iteration", 0)
    img_path = str(output_dir / f"result_iter_{iteration}.{output_cfg.get('image_format', 'png')}")
    engine.export_image(img_path, dpi=output_cfg.get("dpi", 300))

    # 导出当前轮收敛历史图
    convergence_img_path = str(output_dir / f"convergence_iter_{iteration}.png")
    if hasattr(engine, 'get_convergence_image'):
        engine.get_convergence_image(convergence_img_path)
    else:
        from autotopo.utils.visualization import plot_convergence_history
        plot_convergence_history(
            result.compliance_history,
            result.volume_history,
            convergence_img_path,
        )

    return {
        "density_field": result.densities,
        "result_image_path": img_path,
        "convergence_image_path": convergence_img_path,
        "solve_result": {
            "compliance_history": result.compliance_history,
            "volume_history": result.volume_history,
            "iterations": result.iterations,
            "converged": result.converged,
            "mesh_info": result.mesh_info,
        },
        "current_params": params,
    }
