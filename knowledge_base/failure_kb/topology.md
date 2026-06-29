# Topology Quality Failure Modes

`grayness_too_high`: many cells remain between solid and void.
Repair: use bounded penal continuation, increase `penal` gradually, and keep the
step bounded.

`checkerboard`: alternating neighboring cells indicate numerical instability.
Repair: increase `rmin` or the density filter radius.

`disconnected_islands`: solid material is split into disconnected components.
Repair: increase `rmin` or repair the load path.

`too_thin_members`: structural members are too narrow for the filter scale.
Repair: increase filter radius.

`invalid_load_path`: there is no continuous material path from load to supports.
Repair: validate loads, supports, and passive void regions.
