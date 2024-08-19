# Construct a glacier with a Halfar profile, solve the Glen-law
# Stokes problem, and compute its surface map output Phi(s) = - u|_s . n_s.
# As much as possible of this code is dimension-independent; the base mesh
# can be 1D or 2D.  Note that zero-thickness columns are dealt with
# in the solve() method by trivializing those equations; see
# IceFreeConditionXX() methods.

import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextrude import *

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
mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)
mesh.topology_dm.viewFromOptions('-dm_view')  # base mesh DMPlex info only

# physics parameters
secpera = 31556926.0    # seconds per year
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.01
Dtyp = 2.0 / secpera    # 2 a-1
qq = 1.0 / nglen - 1.0

def _form_stokes(mesh, se):
    def D(w):
        return 0.5 * (grad(w) + grad(w).T)
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
    F = ( inner(B3 * Du2**(qq / 2.0) * D(u), D(v)) \
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

# extend sbase, defined on the base mesh, to the extruded mesh using the
#   'R' constant-in-the-vertical space
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
sR = Function(P1R)
sR.dat.data[:] = sb
Vcoord = mesh.coordinates.function_space()
if dim == 2:
    x, z = SpatialCoordinate(mesh)
    newcoord = Function(Vcoord).interpolate(as_vector([x, sR * z]))
else:
    x, y, z = SpatialCoordinate(mesh)
    newcoord = Function(Vcoord).interpolate(as_vector([x, y, sR * z]))
mesh.coordinates.assign(newcoord)

# set up Stokes mixed method
se = StokesExtrude(mesh)
se.mixed_TaylorHood()
#se.mixed_PkDG()

# boundary conditions
if dim == 2:
    se.body_force(Constant((0.0, - rho * g)))
    se.dirichlet((1,2), Constant((0.0,0.0)))  # wrong if ice advances to margin
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
else:
    se.body_force(Constant((0.0, 0.0, - rho * g)))
    se.dirichlet((1,2), Constant((0.0,0.0,0.0)))  # wrong if ice advances to margin
    se.dirichlet(('bottom',), Constant((0.0,0.0,0.0)))

# deal with zero-thickness columns
class IceFreeConditionVelocity(DirichletBC):
    @utils.cached_property
    def nodes(self):
        # return vector P2 nodes in columns with surface elevation less than 1.0 meter
        # warning: assumes velocity space is P2
        P2scalar = FunctionSpace(self.function_space().mesh(), 'CG', 2)
        sU = Function(P2scalar).interpolate(sR)
        if dim == 2:
            ssU = Function(self.function_space()).interpolate(as_vector([sU, sU]))
            return np.where(ssU.dat.data_ro < 1.0)[0]
        else:
            sssU = Function(self.function_space()).interpolate(as_vector([sU, sU, sU]))
            return np.where(sssU.dat.data_ro < 1.0)[0]
class IceFreeConditionPressure(DirichletBC):
    @utils.cached_property
    def nodes(self):
        # return P1 nodes in columns with surface elevation less than 1.0 meter
        sP = Function(self.function_space()).interpolate(sR)
        return np.where(sP.dat.data_ro < 1.0)[0]
#print(IceFreeConditionVelocity(se.Z.sub(0), as_vector([0.0, 0.0]), None).nodes)
#print(IceFreeConditionPressure(se.Z.sub(1), 0.0, None).nodes)
zerovec = as_vector([0.0, 0.0]) if dim == 2 else as_vector([0.0, 0.0, 0.0])
se.addcondition(IceFreeConditionVelocity(se.Z.sub(0), zerovec, None))
se.addcondition(IceFreeConditionPressure(se.Z.sub(1), 0.0, None))

# viscosity scale needed in solvers which use pc_Mass
Du2_0 = 10.0 * (eps * Dtyp)**2.0  # throw in factor of 10?
nu_0 = B3 * Du2_0**(qq / 2.0)
se.viscosity_constant(nu_0)

params = SolverParams['newton']
params.update(SolverParams['mumps'])
#params.update(SolverParams['schur_hypre_mass']) # FIXME not working for now
params.update({'snes_monitor': None,
               'snes_converged_reason': None})
if dim == 2:
    printpar(f'solving 2D Stokes on {mx} x {mz} extruded mesh ...')
else:
    printpar(f'solving 3D Stokes on {mx} x {mx} x {mz} extruded mesh ...')
n_u, n_p = se.V.dim(), se.W.dim()
printpar(f'  sizes: n_u = {n_u}, n_p = {n_p}')
u, p = se.solve(par=params, F=_form_stokes(mesh, se))
se.savesolution(name='result.pvd')
printpar(f'u, p solution norms = {norm(u):8.3e}, {norm(p):8.3e}')

# output surface elevation in P1 ...
sbm = trace_scalar_to_p1(basemesh, mesh, z)
sbm.rename('surface elevation (m)')

# surface velocity in P2 ...
ubm = trace_vector_to_p2(basemesh, mesh, u, dim=dim)
ubm.rename('surface velocity (m s-1)')

# and surface motion in DG0
if dim == 2:
    ns = as_vector([-sbm.dx(0), Constant(1.0)])
else:
    ns = as_vector([-sbm.dx(0), -sbm.dx(1), Constant(1.0)])
DG0bm = FunctionSpace(basemesh, 'DG', 0)
Phibm = Function(DG0bm).project(- dot(ubm, ns))
Phibm.rename('surface motion map Phi (m s-1)')

# .pvd result only in 3D
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

# .png figure with s(x) and Phi(s)(x) only in 2D and in serial
if dim == 2 and basemesh.comm.size == 1:
    xx = basemesh.coordinates.dat.data_ro
    xm = (xx[1:] + xx[:-1]) / 2.0
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(xx / 1.0e3, sbm.dat.data, color='C1', label='s')
    ax1.legend(loc='upper left')
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel('elevation (m)')
    ax2.plot(xm / 1.0e3, Phibm.dat.data * secpera, '.', color='C2', label=r'$\Phi(s)$')
    ax2.legend(loc='upper right')
    ax2.set_ylabel(r'$\Phi$ (m a-1)')
    ax2.grid(visible=True)
    plt.xlabel('x (km)')
    plt.show()
