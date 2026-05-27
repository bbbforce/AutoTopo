"""仿真调度节点。

负责初始化引擎、执行优化迭代、生成结果图。
"""

from __future__ import annotations

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
    backend = config.get("engine", {}).get("backend", "jax_fem")

    if backend == "jax_fem":
        from autotopo.engines.jax_fem_engine import JaxFemEngine
        return JaxFemEngine()
    else:
        raise ValueError(f"不支持的引擎后端: {backend}")


def run_simulation(state: AutoTopoState) -> dict[str, Any]:
    """仿真节点：初始化引擎 → 执行优化 → 导出结果图。"""
    config = _load_config()
    output_cfg = config.get("output", {})
    engine_defaults = config.get("engine", {}).get("default_params", {})

    # 合并参数：问题定义 > 反馈修正 > 配置默认
    problem = state["problem_definition"]
    params = {**engine_defaults}
    params.update(problem.get("parameters", {}))
    if state.get("current_params"):
        params.update(state["current_params"])

    # 初始化引擎
    engine = _get_engine()
    engine.setup(problem)

    # 执行优化
    result = engine.optimize(
        max_iter=params.get("max_iter", 200),
        tol=params.get("tol", 0.01),
        penal=params.get("penal", 3.0),
        rmin=params.get("rmin", 1.5),
        volfrac=params.get("volfrac", 0.5),
    )

    # 导出结果图
    output_dir = Path(state.get("output_path", output_cfg.get("dir", "./output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    iteration = state.get("iteration", 0)
    img_path = str(output_dir / f"result_iter_{iteration}.{output_cfg.get('image_format', 'png')}")
    engine.export_image(img_path, dpi=output_cfg.get("dpi", 300))

    return {
        "density_field": result.densities,
        "result_image_path": img_path,
        "solve_result": {
            "compliance_history": result.compliance_history,
            "volume_history": result.volume_history,
            "iterations": result.iterations,
            "converged": result.converged,
        },
        "current_params": params,
    }
