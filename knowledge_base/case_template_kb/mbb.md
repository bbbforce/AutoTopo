# MBB Beam Template

Benchmark type: `mbb`.

Default half-MBB layout:
- rectangular domain
- left edge fixed in x direction as the symmetry support
- bottom right fixed in y direction
- point force at top left
- downward load direction `[0, -1]`
- minimize compliance with a volume fraction constraint

Good smoke grid: `nelx=12`, `nely=4`.

