# Physics and Modeling Failure Modes

`no_support`: the case has no boundary condition.
Repair: add at least one fixed, fixed_x, or fixed_y boundary condition.

`no_load`: the case has no nonzero load.
Repair: add a point force at a supported benchmark location.

`load_on_fixed_dof`: the same degree of freedom is both loaded and fixed.
Repair: move the load or relax the support component.

`rigid_body_motion`: supports do not constrain enough global motion.
Repair: ensure x and y motion are constrained by the benchmark support template.

`invalid_volume_fraction`: volume fraction must be between 0 and 1.
Repair: clamp to the benchmark default, usually 0.4 or 0.5.

`invalid_filter_radius`: filter radius must be positive and normally at least one
element for structured SIMP.
Repair: set `rmin` to 1.5 or larger.

`invalid_penal`: SIMP penalty must be positive and is normally 2.0 to 5.0.
Repair: set `penal` to 3.0.

