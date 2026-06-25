# Default Physics Rules

The minimal benchmark uses compliance minimization, SIMP interpolation, density
filtering, and MMA updates.

Default safe values:
- `volume_fraction`: 0.4 to 0.5
- `penal`: 3.0
- `rmin`: 1.5 for small structured grids
- `tol`: 0.01 for smoke tests

Corrective parameter rules:
- grayness repair should use penal continuation and avoid large one-step jumps.
- checkerboard repair should increase `rmin` before changing the benchmark
  template.
- volume repair should project the active design variables back to the target
  volume fraction.

Structured parameters always override natural-language inference. Natural language
is only used to infer the benchmark type and missing defaults.
