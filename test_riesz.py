import dolfin
import dolfin_adjoint as da
mesh=da.UnitSquareMesh(10,10)
V=dolfin.FunctionSpace(mesh,'CG',1)
rho=da.Function(V)
control=da.Control(rho)
J=da.assemble(rho**2*dolfin.dx)
Jhat=da.ReducedFunctional(J,control)
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
Jhat_np=ReducedFunctionalNumPy(Jhat)

import types
def _ad_convert_riesz(self, gradient, **kwargs):
    f=dolfin.Function(self.function_space())
    f.vector()[:]=gradient
    return f

# Patch the instance directly!
rho._ad_convert_riesz = types.MethodType(_ad_convert_riesz, rho)
dJ=Jhat_np.derivative()
print('SUCCESS')
