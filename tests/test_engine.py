"""SIMP 仿真引擎测试。"""

import numpy as np
import pytest

from autotopo.engines.jax_fem_engine import JaxFemEngine


def _cantilever_problem(nelx=30, nely=10, volfrac=0.5, max_iter=50):
    return {
        "domain": {"nelx": nelx, "nely": nely},
        "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
        "boundary_conditions": [{"type": "fixed", "location": "left_edge"}],
        "loads": [{"type": "point_force", "location": "right_center",
                   "magnitude": 1.0, "direction": [0, -1]}],
        "constraints": [{"type": "volume_fraction", "value": volfrac}],
        "parameters": {"penal": 3.0, "rmin": 1.5, "max_iter": max_iter},
    }


class TestJaxFemEngine:

    def test_setup(self):
        engine = JaxFemEngine()
        engine.setup(_cantilever_problem())
        assert engine.nelx == 30
        assert engine.nely == 10
        assert engine.densities is not None
        assert engine.densities.shape == (10, 30)

    def test_optimize_runs(self):
        engine = JaxFemEngine()
        engine.setup(_cantilever_problem(max_iter=20))
        result = engine.optimize(max_iter=20)

        assert result.iterations == 20
        assert len(result.compliance_history) == 20
        assert len(result.volume_history) == 20
        assert result.densities.shape == (10, 30)

    def test_compliance_decreases(self):
        """柔度应大体单调递减。"""
        engine = JaxFemEngine()
        engine.setup(_cantilever_problem(max_iter=30))
        result = engine.optimize(max_iter=30)

        # 前5次 vs 后5次的均值
        early = np.mean(result.compliance_history[:5])
        late = np.mean(result.compliance_history[-5:])
        assert late < early, "优化应使柔度降低"

    def test_volume_fraction_respected(self):
        """最终体积分数应接近目标值。"""
        volfrac = 0.4
        engine = JaxFemEngine()
        engine.setup(_cantilever_problem(volfrac=volfrac, max_iter=50))
        result = engine.optimize(max_iter=50, volfrac=volfrac)

        actual_vol = float(np.mean(result.densities))
        assert abs(actual_vol - volfrac) < 0.05, f"体积分数 {actual_vol:.3f} 偏离目标 {volfrac}"

    def test_density_bounds(self):
        """密度值应在 [0, 1] 范围内。"""
        engine = JaxFemEngine()
        engine.setup(_cantilever_problem(max_iter=30))
        result = engine.optimize(max_iter=30)

        assert np.all(result.densities >= 0)
        assert np.all(result.densities <= 1.001)

    def test_different_penal(self):
        """不同罚因子应产生不同结果。"""
        engine1 = JaxFemEngine()
        engine1.setup(_cantilever_problem(max_iter=30))
        r1 = engine1.optimize(max_iter=30, penal=2.0)

        engine2 = JaxFemEngine()
        engine2.setup(_cantilever_problem(max_iter=30))
        r2 = engine2.optimize(max_iter=30, penal=5.0)

        # 高罚因子应产生更接近 0-1 的密度分布
        gray1 = np.sum((r1.densities > 0.1) & (r1.densities < 0.9))
        gray2 = np.sum((r2.densities > 0.1) & (r2.densities < 0.9))
        assert gray2 <= gray1, "更高罚因子应减少灰度单元"

    def test_export_image(self, tmp_path):
        engine = JaxFemEngine()
        engine.setup(_cantilever_problem(max_iter=10))
        engine.optimize(max_iter=10)

        img_path = str(tmp_path / "test_result.png")
        returned_path = engine.export_image(img_path)
        assert returned_path == img_path
        assert (tmp_path / "test_result.png").exists()

    def test_mbb_beam(self):
        """MBB 梁问题应正常求解。"""
        problem = {
            "domain": {"nelx": 30, "nely": 10},
            "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
            "boundary_conditions": [
                {"type": "fixed_x", "location": "left_edge"},
                {"type": "fixed_y", "location": "bottom_right"},
            ],
            "loads": [{"type": "point_force", "location": "top_left",
                       "magnitude": 1.0, "direction": [0, -1]}],
            "constraints": [{"type": "volume_fraction", "value": 0.5}],
            "parameters": {"penal": 3.0, "rmin": 1.5},
        }

        engine = JaxFemEngine()
        engine.setup(problem)
        result = engine.optimize(max_iter=30)

        assert result.iterations == 30
        assert result.densities.shape == (10, 30)
