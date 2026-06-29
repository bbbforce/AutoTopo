# Python SIMP MMA Solver API

PythonSimpMMAEngine accepts a dictionary problem with `domain.nelx`, `domain.nely`,
`boundary_conditions`, `loads`, `constraints`, and `parameters`.

Required optimizer for the minimal research experiment is `MMA`. OC fallback is not
an accepted final success condition.

Important parameters:
- `volfrac` or a `volume_fraction` constraint controls active design material ratio.
- `penal` should usually be between 2.0 and 5.0.
- `rmin` is an element-radius filter value and should be at least 1.0.
- `max_iter` and `tol` control termination.

Outputs:
- `density.npy`
- `density.png`
- `optimization_history.csv`
- `result.json`

