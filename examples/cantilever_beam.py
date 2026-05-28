"""示例：经典悬臂梁拓扑优化（纯引擎，不依赖 LLM）。

左端全固定，右端中点施加向下单位力。
"""

from pathlib import Path

from autotopo.engines.jax_fem_engine import JaxFemEngine

problem = {
    "domain": {"nelx": 80, "nely": 40},
    "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
    "boundary_conditions": [
        {"type": "fixed", "location": "left_edge"},
    ],
    "loads": [
        {"type": "point_force", "location": "right_center",
         "magnitude": 1.0, "direction": [0, -1]},
    ],
    "constraints": [
        {"type": "volume_fraction", "value": 0.4},
    ],
    "parameters": {"penal": 3.0, "rmin": 2.0},
}

if __name__ == "__main__":
    engine = JaxFemEngine()
    engine.setup(problem)
    result = engine.optimize(max_iter=150)

    output = Path("output")
    output.mkdir(exist_ok=True)

    engine.export_image(str(output / "cantilever_beam.png"))
    print(f"✅ 悬臂梁优化完成: {result.iterations}次迭代, 柔度={result.compliance_history[-1]:.4f}")
