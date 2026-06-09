"""Simulator parameter merge/profile tests."""

from autotopo.nodes.simulator import _apply_runtime_profile, _merge_parameters


def test_feedback_cannot_override_volume_constraint():
    problem = {
        "constraints": [{"type": "volume_fraction", "value": 0.5}],
        "parameters": {"volfrac": 0.4, "penal": 3.0, "rmin": 0.05},
    }
    defaults = {"volfrac": 0.3, "penal": 2.0, "rmin": 0.1}
    feedback = {"volfrac": 0.35, "penal": 4.0}

    params = _merge_parameters(problem, defaults, feedback)

    assert params["volfrac"] == 0.5
    assert params["penal"] == 4.0
    assert params["rmin"] == 0.05


def test_preview_profile_overrides_runtime_mesh_and_solver_params():
    problem = {
        "domain": {"width": 60.0, "height": 20.0, "mesh_resolution": 1.0},
        "parameters": {"max_iter": 200, "tol": 1e-6},
    }
    params = {"max_iter": 200, "tol": 1e-6, "penal": 3.0}
    profile = {"mesh_resolution": 2.0, "max_iter": 60, "tol": 1e-4, "output_dpi": 120}

    runtime_params = _apply_runtime_profile(problem, params, profile, stage="preview")

    assert problem["domain"]["mesh_resolution"] == 2.0
    assert runtime_params["max_iter"] == 60
    assert runtime_params["tol"] == 1e-4
    assert runtime_params["output_dpi"] == 120
    assert runtime_params["solve_stage"] == "preview"


def test_final_profile_keeps_original_mesh_resolution():
    problem = {
        "domain": {"width": 60.0, "height": 20.0, "mesh_resolution": 1.0},
        "parameters": {"max_iter": 100, "tol": 1e-5},
    }
    params = {"max_iter": 100, "tol": 1e-5, "penal": 3.0}
    profile = {"max_iter": 200, "tol": 1e-6, "output_dpi": 300}

    runtime_params = _apply_runtime_profile(problem, params, profile, stage="final")

    assert problem["domain"]["mesh_resolution"] == 1.0
    assert runtime_params["max_iter"] == 200
    assert runtime_params["tol"] == 1e-6
    assert runtime_params["output_dpi"] == 300
    assert runtime_params["solve_stage"] == "final"
