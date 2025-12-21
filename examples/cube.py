# test scalability of GMG and Schur solvers on a 3D cube
# Stokes problem: lid-driven cavity with constant viscosity and stress-free base

from firedrake import *
from stokesextrude import *
import sys
import time

if len(sys.argv) > 1:
    solvetype = sys.argv[1]
else:
    solvetype = "mumps"

bmx = 4
bmz = 2
levs = 3

# serial solve times (seconds) on lemur (41.6GB; 6GB background) with 4x4x2 base mesh:
#   levs | mumps   memory  | gmg     memory
#   2    | 0.50    ..      | 2.09    ..
#   3    | 5.79    ..      | 8.69    ..
#   4    | 199.0   23GB    | 70.33   9.2GB
#   5    |                 | 594.1   29GB
# [meshes: 2->8x8x4, 3->16x16x8, 4->32x32x16, 5->64x64x32]

mx = bmx * 2**(levs - 1)
mz = bmz * 2**(levs - 1)
if solvetype == "gmg":
    coarsebasemesh = UnitSquareMesh(bmx, bmx)
    #coarsebasemesh = UnitSquareMesh(bmx, bmx, diagonal="crossed")
    #coarsebasemesh = UnitSquareMesh(bmx, bmx, quadrilateral=True)
    basehierarchy = MeshHierarchy(coarsebasemesh, levs - 1)
    meshhierarchy = ExtrudedMeshHierarchy(basehierarchy, 1.0, base_layer=bmz, refinement_ratio=2)
    se = StokesExtrude(basehierarchy[-1], mz=mz, mesh=meshhierarchy[-1])
else:
    basemesh = UnitSquareMesh(mx, mx)
    #basemesh = UnitSquareMesh(mx, mx, diagonal="crossed")
    se = StokesExtrude(basemesh, mz=mz)
se.reset_elevations(Constant(0.0), Constant(1.0))

se.mixed_TaylorHood()
u, p = split(se.up)
v, q = TestFunctions(se.Z)

# linear Stokes with viscosity nu = 1.0
se.viscosity_constant(1.0)  # _mass solvers use this value
f_body = Constant((1.0, 1.0, -1.0))
F = (
    inner(2.0 * se.D(u), se.D(v)) - p * div(v) - q * div(u) - inner(f_body, v)
) * dx

# drive lid in 45 degree direction with maximum speed of 1.0,
#   with lid speed zero on boundaries to reduce pressure singularity
x, y, _ = SpatialCoordinate(se.mesh)
bump = 16.0 * x * (1.0 - x) * y * (1.0 - y) / sqrt(2.0)
se.dirichlet(("top",), as_vector([bump, bump, 0.0]))

# zero velocity on sides (but no stress on bottom)
se.dirichlet((1, 2, 3, 4), Constant((0.0, 0.0, 0.0)))

params = SolverParams["newton"]
params["ksp_converged_reason"] = None
params["snes_monitor"] = None
params["snes_converged_reason"] = None

if solvetype == "gmg":
    #params.update(SolverParams["schur_gmg_mass"])
    params.update(SolverParams["schur_gmg_selfp"])
else:
    params.update(SolverParams[solvetype])
    #params.update(SolverParams["mumps"])
    #params.update(SolverParams["schur_nonscalable"])
    #params.update(SolverParams["schur_nonscalable_selfp"])
    #params.update(SolverParams["schur_nonscalable_mass"])
    #params.update(SolverParams["schur_hypre_mass"])

n_u, n_p = se.V.dim(), se.W.dim()
printpar(f"solving for {mx} x {mx} x {mz} mesh with sizes n_u = {n_u}, n_p = {n_p} ...")

start_time = time.perf_counter()
try:
    u, p = se.solve(F=F, par=params, pinch=False)
except firedrake.ConvergenceError:
    print("convergence error caught")
end_time = time.perf_counter()
printpar(f"solve time = {end_time - start_time:.2f} seconds")

se.savesolution("result.pvd")
