"""FEniCS + dolfin-adjoint Docker proxy tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autotopo.engines.dolfin_adjoint_engine import DolfinAdjointEngine


def _cantilever_problem(volfrac=0.4):
    return {
        "domain": {"width": 60.0, "height": 20.0, "mesh_resolution": 1.0},
        "material": {"youngs_modulus": 1.0, "poissons_ratio": 0.3},
        "boundary_conditions": [{"type": "fixed", "location": "left_edge"}],
        "loads": [{"type": "point_force", "location": "right_center",
                   "magnitude": 1.0, "direction": [0, -1]}],
        "constraints": [{"type": "volume_fraction", "value": volfrac}],
        "parameters": {"penal": 3.0, "rmin": 0.05, "max_iter": 20, "optimizer": "SLSQP"},
    }


class FakeDolfinAdjointEngine(DolfinAdjointEngine):
    def __init__(self):
        super().__init__(container="fake-dolfin")
        self.sent_problem = None

    def _deploy_solver(self) -> None:
        return None

    def _docker_exec(self, cmd: str, *, capture: bool = False):
        if capture:
            return 0, "fake solve complete", ""
        return 0, "", ""

    def _docker_cp_to(self, local_path: str, container_path: str) -> None:
        if local_path.endswith("problem.json"):
            self.sent_problem = json.loads(Path(local_path).read_text())

    def _docker_cp_from(self, container_path: str, local_path: str) -> None:
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.name == "result.json":
            path.write_text(json.dumps({
                "solve_stage": "preview",
                "iterations": 3,
                "converged": True,
                "early_stopped": True,
                "timings": {"mesh": 0.1, "setup": 0.2, "optimization": 0.3, "export": 0.4, "total": 1.0},
                "compliance_history": [10.0, 7.0, 5.0],
                "volume_history": [0.4, 0.4, 0.4],
                "mesh_info": {"num_cells": 12, "num_vertices": 9},
            }))
        elif path.name == "density_grid.npy":
            np.save(path, np.full((2, 3), 0.4))
        else:
            path.write_bytes(b"fake image")


class TestDolfinAdjointEngine:
    def test_setup_syncs_constraint_volfrac_into_params(self):
        problem = _cantilever_problem(volfrac=0.35)
        problem["parameters"]["volfrac"] = 0.8

        engine = FakeDolfinAdjointEngine()
        engine.setup(problem)

        assert engine._problem["parameters"]["volfrac"] == 0.35

    def test_setup_converts_legacy_nelx_to_mesh_resolution(self):
        problem = _cantilever_problem()
        problem["domain"] = {"width": 60.0, "height": 20.0, "nelx": 30, "nely": 10}

        engine = FakeDolfinAdjointEngine()
        engine.setup(problem)

        assert engine._problem["domain"]["mesh_resolution"] == 2.0

    def test_optimize_sends_synced_problem_and_loads_result(self):
        engine = FakeDolfinAdjointEngine()
        engine.setup(_cantilever_problem(volfrac=0.4))

        result = engine.optimize(max_iter=7, penal=4.0, rmin=0.08, volfrac=0.3)

        assert engine.sent_problem["parameters"]["max_iter"] == 7
        assert engine.sent_problem["parameters"]["penal"] == 4.0
        assert engine.sent_problem["parameters"]["rmin"] == 0.08
        assert engine.sent_problem["parameters"]["volfrac"] == 0.3
        assert engine.sent_problem["constraints"][0]["value"] == 0.3
        assert result.iterations == 3
        assert result.densities.shape == (2, 3)
        assert result.compliance_history[-1] == 5.0
        assert result.extra["solve_stage"] == "preview"
        assert result.extra["early_stopped"] is True
        assert result.extra["timings"]["total"] == 1.0

    def test_optimize_sends_early_stop_config(self):
        problem = _cantilever_problem(volfrac=0.4)
        problem["early_stop"] = {"enabled": True, "min_iter": 5, "window": 3, "rel_delta": 0.01}
        engine = FakeDolfinAdjointEngine()
        engine.setup(problem)

        engine.optimize(max_iter=7)

        assert engine.sent_problem["early_stop"]["min_iter"] == 5
        assert engine.sent_problem["early_stop"]["window"] == 3

    def test_export_images_after_optimize(self, tmp_path):
        engine = FakeDolfinAdjointEngine()
        engine.setup(_cantilever_problem())
        engine.optimize(max_iter=3)

        density_path = tmp_path / "density.png"
        convergence_path = tmp_path / "convergence.png"

        assert engine.export_image(str(density_path)) == str(density_path)
        assert engine.get_convergence_image(str(convergence_path)) == str(convergence_path)
        assert density_path.read_bytes() == b"fake image"
        assert convergence_path.read_bytes() == b"fake image"
