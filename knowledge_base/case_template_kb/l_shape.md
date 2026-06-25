# L-shape Beam Template

Benchmark type: `l_shape`.

Default layout for the minimal experiment:
- start from a square rectangular grid
- remove the upper-right quadrant as passive void elements
- left edge fixed
- point force at the bottom-right corner
- downward load direction `[0, -1]`

The passive void region must remain density 0 and should be excluded from the
active volume-fraction calculation.

