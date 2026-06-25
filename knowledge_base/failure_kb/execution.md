# Execution Failure Modes

`missing_dependency`: import error for numpy, scipy, matplotlib, or local packages.
Repair: keep optional imports lazy or install the dependency outside the benchmark
workflow.

`shape_mismatch`: density, mask, FE matrix, or element count dimensions disagree.
Repair: rebuild the mesh-dependent arrays after changing `nelx`, `nely`, or `rmin`.

`singular_stiffness_matrix`: supports are missing (`no_support`), supports allow
`rigid_body_motion`, loads are disconnected from the active design domain, or the
active density collapsed.
Repair: validate supports and loads before solving; increase minimum stiffness or
repair invalid boundary conditions.

`invalid_boundary_condition`: location strings cannot be mapped to nodes or DOFs.
Repair: choose a supported location such as `left_edge`, `right_center`,
`top_left`, or `bottom_right`.
