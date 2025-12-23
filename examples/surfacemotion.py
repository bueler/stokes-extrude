# Construct a glacier with a Halfar profile, solve the Glen-law
# Stokes problem, and compute its surface map output Phi(s) = - u|_s . n_s.
# As much as possible of this code is dimension-independent; the base mesh
# can be 1D or 2D.  Note that zero-thickness columns are dealt with
# in the solve() method by trivializing those equations.

import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextrude import *

# FIXME "gmg" solver choice has two problems:
#   1) transfer operator needs hmin>0 (and largish?) to be defined
#   2) is a crappy solver with current schur_gmg_selfp settings

# run as:
#   python3 surfacemotion.py DIM LEVS METHOD
# default: python3 surfacemotion.py 2 3 mumps
import sys
if len(sys.argv) > 1:
    dim = int(sys.argv[1])
else:
    dim = 2
if len(sys.argv) > 2:
    levs = int(sys.argv[2])
else:
    levs = 3
if len(sys.argv) > 3:
    method = sys.argv[3]
else:
    method = "mumps"

# mesh parameters
assert dim in [2, 3]
if dim == 2:
    cmx = 25
    cmz = 4
else:
    cmx = 4
    cmz = 1
mx = cmx * 2**(levs - 1)
mz = cmz * 2**(levs - 1)

# map-plane region dimensio
L = 100.0e3  # 2D: domain is [-L,L];  3D: domain is [-L,L] x [-L,L]

# extruded mesh via refinement of coarse base mesh
if dim == 2:
    printpar(f"generating 2D {mx}x{mz} extruded mesh from {cmx}x{cmz} coarse and {levs} levels ...")
else:
    printpar(f"generating 3D {mx}x{mx}x{mz} extruded mesh from {cmx}x{cmx}x{cmz} coarse and {levs} levels ...")
if dim == 2:
    coarsebasemesh = IntervalMesh(cmx, -L, L)
else:
    coarsebasemesh = RectangleMesh(cmx, cmx, L, L, originX=-L, originY=-L, diagonal="crossed")
coarsebasemesh.topology_dm.viewFromOptions("-dm_view")
se = StokesExtrude(coarsebasemesh, mz=cmz, levs=levs)

# physics parameters
secpera = 31556926.0  # seconds per year
g, rho = 9.81, 910.0  # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24  # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3 ** (-1.0 / 3.0)  # Pa s(1/3);  ice hardness
eps = 0.01
Dtyp = 2.0 / secpera  # 2 a-1
qq = 1.0 / nglen - 1.0

# set surface geometry from Halfar time-dependent SIA geometry solutions,
# a dome with zero SMB; reference:
#   * P. Halfar (1981), On the dynamics of the ice sheets, J. Geophys. Res. 86 (C11), 11065--11072
#   * P. Halfar (1983), On the dynamics of the ice sheets 2, J. Geophys. Res. 88, 6043--6051
# The solution is evaluated at t = t0.
pp = 1.0 + 1.0 / nglen
rr = nglen / (2.0 * nglen + 1.0)
R0 = 70.0e3  # Halfar dome radius
H0 = 1200.0  # Halfar dome height
s = [None for j in range(se.levs)]
for j in range(se.levs):
    P1b = FunctionSpace(se.basehier[j], "P", 1)
    s[j] = Function(P1b)  # set to zero
    x = SpatialCoordinate(se.basehier[j])
    hmin = 10.0 if method == "gmg" else 0.0  # FIXME
    if dim == 2:
        s[j].interpolate(conditional(abs(x[0]) < R0,
                                     H0 * (1.0 - abs(x[0] / R0) ** pp) ** rr,
                                     hmin))
    else:
        r = sqrt(x[0] * x[0] + x[1] * x[1])
        s[j].interpolate(conditional(r < R0,
                                     H0 * (1.0 - abs(r / R0) ** pp) ** rr,
                                     hmin))
se.reset_elevations(0.0, s)

# function spaces
se.mixed_TaylorHood()
n_u, n_p = se.V.dim(), se.W.dim()
printpar(f"  sizes: n_u = {n_u}, n_p = {n_p}")

# boundary conditions;  wrong if ice advances to margin
if dim == 2:
    se.dirichlet((1, 2), Constant((0.0, 0.0)))
    se.dirichlet(("bottom",), Constant((0.0, 0.0)))
else:
    se.dirichlet((1, 2), Constant((0.0, 0.0, 0.0)))
    se.dirichlet(("bottom",), Constant((0.0, 0.0, 0.0)))


def _form_stokes(se):
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(se.D(u), se.D(u)) + (eps * Dtyp) ** 2.0
    if dim == 2:
        f_body = Constant((0.0, -rho * g))
    else:
        f_body = Constant((0.0, 0.0, -rho * g))
    F = (
        inner(B3 * Du2 ** (qq / 2.0) * se.D(u), se.D(v))
        - p * div(v)
        - div(u) * q
        - inner(f_body, v)
    ) * dx(degree=4)
    return F


# viscosity scale needed in solvers which use pc_Mass
Du2_0 = 10.0 * (eps * Dtyp) ** 2.0  # throw in factor of 10?
nu_0 = B3 * Du2_0 ** (qq / 2.0)
se.viscosity_constant(nu_0)

params = SolverParams["newton"]
if method == "gmg":
    params.update(SolverParams["schur_gmg_selfp"])
else:
    params.update(SolverParams["mumps"])
# params.update(SolverParams['schur_hypre_mass']) # FIXME not working for now
params.update({"snes_monitor": None, "snes_converged_reason": None})
printpar(f"solving {dim}D Stokes by newton-{method} method ...")
u, p = se.solve(F=_form_stokes(se), par=params, pinch=True)
se.save_solution(name="result.pvd")
printpar(f"u, p solution norms = {norm(u):8.3e}, {norm(p):8.3e}")

# output surface elevation in P1 ...
x = SpatialCoordinate(se.mesh)
sbm = trace_scalar_to_p1(se.basehier[-1], se.mesh, x[dim - 1])  # z = x[dim-1]
sbm.rename("surface elevation (m)")

# surface velocity in P2 ...
ubm = trace_vector_to_p2(se.basehier[-1], se.mesh, u, dim=dim)
ubm.rename("surface velocity (m s-1)")

# and surface motion in DG0
if dim == 2:
    ns = as_vector([-sbm.dx(0), Constant(1.0)])
else:
    ns = as_vector([-sbm.dx(0), -sbm.dx(1), Constant(1.0)])
DG0bm = FunctionSpace(se.basehier[-1], "DG", 0)
Phibm = Function(DG0bm).project(dot(ubm, ns))
Phibm.rename("surface motion map Phi = u|_s . n_s (m s-1)")

# .pvd result only in 3D
if dim == 3:
    bmname = "result_base.pvd"
    if coarsebasemesh.comm.size > 1:
        printpar("saving s,u,Phi,rank at top surface to %s" % bmname)
        rankbm = Function(FunctionSpace(se.basehier[-1], "DG", 0))
        rankbm.dat.data[:] = se.basehier[-1].comm.rank
        rankbm.rename("rank")
        VTKFile(bmname).write(sbm, ubm, Phibm, rankbm)
    else:
        printpar("saving s,u,Phi at top surface to %s" % bmname)
        VTKFile(bmname).write(sbm, ubm, Phibm)

# .png figure with s(x) and Phi(s)(x) only in 2D and in serial
if dim == 2 and coarsebasemesh.comm.size == 1:
    xx = se.basehier[-1].coordinates.dat.data_ro
    xm = (xx[1:] + xx[:-1]) / 2.0
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(xx / 1.0e3, sbm.dat.data, color="C1", label="s")
    ax1.legend(loc="upper left")
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel("elevation (m)")
    ax2.plot(xm / 1.0e3, Phibm.dat.data * secpera, ".", color="C2", label=r"$\Phi(s)$")
    ax2.legend(loc="upper right")
    ax2.set_ylabel(r"$\Phi$ (m a-1)")
    ax2.grid(visible=True)
    plt.xlabel("x (km)")
    plt.show()
