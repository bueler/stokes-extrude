from firedrake import *
from stokesextrude import *

basemesh = UnitIntervalMesh(20)
se = StokesExtrude(basemesh, mz=10)
se.mixed_TaylorHood()
se.viscosity_constant(1.0)
u, p = split(se.up)
v, q = TestFunctions(se.Z)
f_body = Constant((1.0, -1.0))
F = (
    inner(2.0 * se.nu * se.D(u), se.D(v)) - p * div(v) - q * div(u) - inner(f_body, v)
) * dx
se.dirichlet(("bottom",), Constant((0.0, 0.0)))
params = SolverParams["newton"]
params.update(SolverParams["mumps"])
params["snes_converged_reason"] = None
u, p = se.solve(F=F, par=params)
se.savesolution("result.pvd")
