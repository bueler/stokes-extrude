# test scalability of GMG and Schur solvers on 3D cube
# Stokes problem is lid-driven cavity with constant viscosity and stress-free base

from firedrake import *
from stokesextrude import *

mx = 10
mz = 5

basemesh = UnitSquareMesh(mx, mx, diagonal="crossed")
se = StokesExtrude(basemesh, mz=mz)

se.mixed_TaylorHood()
u, p = split(se.up)
v, q = TestFunctions(se.Z)

# linear Stokes with viscosity nu = 1.0
se.viscosity_constant(1.0)  # some solvers use this value
f_body = Constant((1.0, 1.0, -1.0))
F = (
    inner(2.0 * se.D(u), se.D(v)) - p * div(v) - q * div(u) - inner(f_body, v)
) * dx

# drive lid in 45 degree direction with maximum speed of 1.0, and
#   with driving speed going to zero on lid boundaries
x, y, _ = SpatialCoordinate(se.mesh)
bump = 16.0 * x * (1.0 - x) * y * (1.0 - y) / sqrt(2.0)
se.dirichlet(("top",), as_vector([bump, bump, 0.0]))

# zero velocity on sides (but no stress on bottom)
se.dirichlet((1, 2, 3, 4), Constant((0.0, 0.0, 0.0)))

params = SolverParams["newton"]
params["ksp_converged_reason"] = None
params["snes_monitor"] = None
params["snes_converged_reason"] = None

#params.update(SolverParams["mumps"])
#params.update(SolverParams["schur_nonscalable"])
#params.update(SolverParams["schur_nonscalable_selfp"])
#params.update(SolverParams["schur_nonscalable_mass"])
params.update(SolverParams["schur_hypre_mass"])

# FIXME try GMG solver in 3D; convince myself it is worth it in this no-pinch case!

try:
    u, p = se.solve(F=F, par=params)
except firedrake.ConvergenceError:
    print("convergence error caught")

se.savesolution("result.pvd")
