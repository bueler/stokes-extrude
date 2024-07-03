# Construct a glacier with a Halfar profile, solve the Glen-law
# Stokes problem, and compute its surface map output Phi(s) = - u|_s . n_s.
# As much as possible of this code is dimension-independent.

import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *

if True:
    # 2D: mx x mz mesh
    dim = 2
    mx = 101
    mz = 15
else:
    # 3D: mx x mx x mz mesh
    dim = 3
    mx = 20
    mz = 7
L = 100.0e3             # 2D: domain is [-L,L];  3D: domain is [-L,L] x [-L,L]
R0 = 70.0e3             # Halfar dome radius
H0 = 1200.0             # Halfar dome height

# would like to make these True, so that there is no fake ice
extrudeemptycols = False
nominicethickness = False
Hmin = 50.0

# base mesh and extruded mesh (but before initial geometry)
if dim == 2:
    basemesh = IntervalMesh(mx, -L, L)
    xb = basemesh.coordinates.dat.data_ro
else:
    basemesh = RectangleMesh(mx, mx, 2*L, 2*L, diagonal='crossed')
    # coordinate kludge; see warning https://www.firedrakeproject.org/mesh-coordinates.html
    basemesh.coordinates.dat.data[:, :] -= L
    xb = basemesh.coordinates.dat.data_ro[:,0]
    yb = basemesh.coordinates.dat.data_ro[:,1]
basemesh.topology_dm.viewFromOptions('-dm_view_basemesh')
if extrudeemptycols:
    # extruded mesh with empty columns where there is no ice (sb == 0)
    assert dim == 2
    layers = np.zeros((mx, 2))
    layers[:, 1] = mz
    xbc = (xb[1:] + xb[:-1]) / 2.0  # coords of centers of cells
    layers[abs(xbc) > R0, 1] = 0
    mesh = ExtrudedMesh(basemesh, layers=layers, layer_height=1.0/mz)
else:
    mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)
# note mesh.topology_dm.viewFromOptions() gives same result as for basemesh

# physics parameters
secpera = 31556926.0    # seconds per year
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.01
Dtyp = 2.0 / secpera    # 2 a-1

def _form_stokes(mesh, se):
    def D(w):
        return 0.5 * (grad(w) + grad(w).T)
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
    rrr = 1.0 / nglen - 1.0
    F = ( inner(B3 * Du2**(rrr / 2.0) * D(u), D(v)) \
              - p * div(v) - div(u) * q - inner(se.f_body, v) ) * dx(degree=4)
    return F

# the Halfar time-dependent SIA geometry solutions, a dome with zero SMB,
# are from:
#   * P. Halfar (1981), On the dynamics of the ice sheets,
#     J. Geophys. Res. 86 (C11), 11065--11072
#   * P. Halfar (1983), On the dynamics of the ice sheets 2,
#     J. Geophys. Res., 88, 6043--6051
# The solution is evaluated at t = t0.
pp = 1.0 + 1.0 / nglen
rr = nglen / (2.0 * nglen + 1.0)
sb = np.zeros(np.shape(xb))
# following seems to work in parallel!
if dim == 2:
    sb[abs(xb) < R0] = H0 * (1.0 - abs(xb[abs(xb) < R0] / R0)**pp)**rr
else:
    rb = np.sqrt(xb * xb + yb * yb)
    sb[rb < R0] = H0 * (1.0 - abs(rb[rb < R0] / R0)**pp)**rr
if not nominicethickness:
    sb[sb < Hmin] = Hmin

# extend sbase, defined on the base mesh, to the extruded mesh using the
#   'R' constant-in-the-vertical space
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(P1R)
s.dat.data[:] = sb
Vcoord = mesh.coordinates.function_space()
if dim == 2:
    x, z = SpatialCoordinate(mesh)
    newcoord = Function(Vcoord).interpolate(as_vector([x, s * z]))
else:
    x, y, z = SpatialCoordinate(mesh)
    newcoord = Function(Vcoord).interpolate(as_vector([x, y, s * z]))
mesh.coordinates.assign(newcoord)

se = StokesExtruded(mesh)
se.mixed_TaylorHood()
if dim == 2:
    se.body_force(Constant((0.0, - rho * g)))
    se.dirichlet((1,2), Constant((0.0,0.0)))  # wrong if ice advances to margin
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
else:
    se.body_force(Constant((0.0, 0.0, - rho * g)))
    se.dirichlet((1,2), Constant((0.0,0.0,0.0)))  # wrong if ice advances to margin
    se.dirichlet(('bottom',), Constant((0.0,0.0,0.0)))

params = SolverParams['newton']
params.update(SolverParams['mumps'])
params.update({'snes_monitor': None,
               'snes_converged_reason': None})
if dim == 2:
    printpar(f'solving Stokes on {mx} x {mz} extruded mesh ...')
else:
    printpar(f'solving Stokes on {mx} x {mx} x {mz} extruded mesh ...')
n_u, n_p = se.V.dim(), se.W.dim()
printpar(f'  sizes: n_u = {n_u}, n_p = {n_p}')
u, p = se.solve(par=params, F=_form_stokes(mesh, se))
se.savesolution(name='result.pvd')
printpar(f'u, p solution norms = {norm(u):8.3e}, {norm(p):8.3e}')

sbm = trace_scalar_to_p1(basemesh, mesh, z)
sbm.rename('surface elevation (m)')
if dim == 2:
    ns = [-sbm.dx(0), Constant(1.0)]
else:
    ns = [-sbm.dx(0), -sbm.dx(1), Constant(1.0)]
ubm = trace_vector_to_p2(basemesh, mesh, u, dim=dim)
ubm.rename('surface velocity (m s-1)')

P1bm = FunctionSpace(basemesh, 'CG', 1)
# observation: if interpolate() is used in next line then asymmetry results
Phibm = Function(P1bm).project(- dot(ubm, as_vector(ns)))
Phibm.rename('surface motion map Phi (m s-1)')

if dim == 3:
    bmname = 'result_base.pvd'
    if basemesh.comm.size > 1:
        printpar('saving s,u,Phi,rank at top surface to %s' % bmname)
        rankbm = Function(FunctionSpace(basemesh,'DG',0))
        rankbm.dat.data[:] = basemesh.comm.rank
        rankbm.rename('rank')
        VTKFile(bmname).write(sbm, ubm, Phibm, rankbm)
    else:
        printpar('saving s,u,Phi at top surface to %s' % bmname)
        VTKFile(bmname).write(sbm, ubm, Phibm)

if dim == 2 and basemesh.comm.size == 1:
    # figure with s(x) and Phi(s)  [serial only]
    xx = basemesh.coordinates.dat.data
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(xx / 1.0e3, sbm.dat.data, color='C1', label='s')
    ax1.legend(loc='upper left')
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel('elevation (m)')
    ax2.plot(xx / 1.0e3, Phibm.dat.data * secpera, color='C2', label=r'$\Phi(s)$')
    ax2.legend(loc='upper right')
    ax2.set_ylabel(r'$\Phi$ (m a-1)')
    ax2.grid(visible=True)
    plt.xlabel('x (km)')
    plt.show()
