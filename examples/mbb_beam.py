"""示例：MBB 梁拓扑优化（纯引擎，不依赖 LLM）。

利用对称性建模半梁：
- 左端固定水平位移（对称轴）
- 右下角固定竖直位移（支座）
- 左上角施加向下单位力
"""

from pathlib import Path

from autotopo.engines.jax_fem_engine import JaxFemEngine

problem = {
    "domain": {"nelx": 90, "nely": 30},
    "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
    "boundary_conditions": [
        {"type": "fixed_x", "location": "left_edge"},
        {"type": "fixed_y", "location": "bottom_right"},
    ],
    "loads": [
        {"type": "point_force", "location": "top_left",
         "magnitude": 1.0, "direction": [0, -1]},
    ],
    "constraints": [
        {"type": "volume_fraction", "value": 0.5},
    ],
    "parameters": {"penal": 3.0, "rmin": 1.5},
}

if __name__ == "__main__":
    engine = JaxFemEngine()
    engine.setup(problem)
    result = engine.optimize(max_iter=200)

    output = Path("output")
    output.mkdir(exist_ok=True)

    engine.export_image(str(output / "mbb_beam.png"))
    print(f"✅ MBB梁优化完成: {result.iterations}次迭代, 柔度={result.compliance_history[-1]:.4f}")
