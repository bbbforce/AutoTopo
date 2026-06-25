# Optimization Failure Modes

`non_convergence`: the optimizer reached `max_iter` without satisfying tolerance.
Repair: increase `max_iter`, loosen smoke-test tolerance, or reduce parameter jumps.

`compliance_nan_or_inf`: objective is not finite.
Repair: check stiffness assembly, supports, minimum stiffness, and density bounds.

`volume_constraint_violation`: final active volume differs from the target volume.
Repair: project the design variables back to the target active volume fraction.

`density_collapse`: almost all active cells become void or solid.
Repair: validate gradients and volume projection.

`mma_oscillation`: objective increases abnormally across repeated iterations.
Repair: reduce MMA move limit and apply bounded repair.

