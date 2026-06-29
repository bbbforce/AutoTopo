"""PythonSimpMMAEngine smoke tests。"""

from __future__ import annotations

from pathlib import Path

import pytest

from autotopo.engines.python_simp_mma_engine import PythonSimpMMAEngine
from autotopo.engines.structured_benchmarks import case_to_problem, default_case_spec


@pytest.mark.parametrize("benchmark", ["cantilever", "mbb", "l_shape"])
def test_python_simp_mma_engine_runs_small_grid(benchmark, tmp_path):
    spec = default_case_spec(benchmark, quick=True)
    engine = PythonSimpMMAEngine()
    engine.setup(case_to_problem(spec))

    result = engine.optimize(max_iter=1, tol=0.0)
    files = engine.save_outputs(tmp_path / benchmark, result)

    assert result.iterations == 1
    assert result.densities.shape == (spec.nely, spec.nelx)
    assert result.extra["optimizer"] == "MMA"
    assert result.extra["optimizer_fallback"] is None
    for path in files.values():
        assert Path(path).exists()

