# Construct a glacier with a Halfar profile s(x), solve the Glen-law
# Stokes problem, and compute its surface map output Phi(s) = - u|_s . n_s.

import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *

mx = 90                 # mx x mz mesh, but with no-ice (empty) columns
mz = 10
L = 100.0e3             # domain is [-L,L]

# would like to make these True, so that there is no fake ice
extrudeemptycols = False
nominicethickness = False
Hmin = 50.0

secpera = 31556926.0    # seconds per year
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.01
Dtyp = 2.0 / secpera    # 2 a-1

R0 = 70.0e3
H0 = 1200.0

def _form_2d_ice(mesh, se):
    def D(w):
        return 0.5 * (grad(w) + grad(w).T)
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
    rrr = 1.0 / nglen - 1.0
    F = ( inner(B3 * Du2**(rrr / 2.0) * D(u), D(v)) \
              - p * div(v) - div(u) * q - inner(se.f_body, v) ) * dx(degree=4)
    return F

basemesh = IntervalMesh(mx, -L, L)
xb = basemesh.coordinates.dat.data_ro

# P. Halfar (1981), On the dynamics of the ice sheets,
#   J. Geophys. Res. 86 (C11), 11065--11072
# solution evaluated at t = t0, so "f(t)" = 1.0
pp = 1.0 + 1.0 / nglen
rr = nglen / (2.0 * nglen + 1.0)
sb = np.zeros(np.shape(xb))
sb[abs(xb) < R0] = H0 * (1.0 - abs(xb[abs(xb) < R0] / R0)**pp)**rr
if not nominicethickness:
    sb[sb < 100.0] = Hmin

if extrudeemptycols:
    # extruded mesh with empty columns where there is no ice (sb == 0)
    layers = np.zeros((mx, 2))
    layers[:, 1] = mz
    xbc = (xb[1:] + xb[:-1]) / 2.0  # coords of centers of cells
    layers[abs(xbc) > R0, 1] = 0
    mesh = ExtrudedMesh(basemesh, layers=layers, layer_height=1.0/mz)
else:
    mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)

# extend sbase, defined on the base mesh, to the extruded mesh using the
#   'R' constant-in-the-vertical space
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(P1R)
s.dat.data[:] = sb
Vcoord = mesh.coordinates.function_space()
x, z = SpatialCoordinate(mesh)
XZ = Function(Vcoord).interpolate(as_vector([x, s * z]))
mesh.coordinates.assign(XZ)

P1 = FunctionSpace(mesh, 'CG', 1)
dummy = Function(P1).interpolate(Constant(1.0))
VTKFile('result.pvd').write(dummy)

se = StokesExtruded(mesh)
se.mixed_TaylorHood()
se.body_force(Constant((0.0, - rho * g)))
se.dirichlet((1,2), Constant((0.0,0.0)))  # wrong if ice advances to margin
se.dirichlet(('bottom',), Constant((0.0,0.0)))

params = SolverParams['newton']
params.update(SolverParams['mumps'])
params.update({'snes_monitor': None,
               'snes_converged_reason': None})
u, p = se.solve(par=params, F=_form_2d_ice(mesh, se))
se.savesolution(name='result.pvd')
printpar(f'u, p solution norms = {norm(u):8.3e}, {norm(p):8.3e}')

# FIXME trace does not seem to work
# the following do, but only in serial

P1topbc = DirichletBC(P1, 0.0, 'top')
P1bm = FunctionSpace(basemesh, 'CG', 1)
sbm = Function(P1bm)
sbm.dat.data[:] = Function(P1).interpolate(z).dat.data_with_halos[P1topbc.nodes]
assert max(abs(sbm.dat.data - sb)) == 0.0

P2topbc = DirichletBC(u.function_space(), as_vector([0.0, 0.0]), 'top')
VP2bm = VectorFunctionSpace(basemesh, 'CG', 2, dim=2)
ubm = Function(VP2bm)
ubm.dat.data[:] = Function(u.function_space()).interpolate(u).dat.data_with_halos[P2topbc.nodes]

ns = [-sbm.dx(0), Constant(1.0)]
Phi = Function(P1bm).interpolate(- dot(ubm, as_vector(ns)))

# figure with s(x) and Phi(s)
xx = basemesh.coordinates.dat.data
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(xx / 1.0e3, sbm.dat.data, color='C1', label='s')
ax1.legend(loc='upper left')
ax1.set_xticklabels([])
ax1.grid(visible=True)
ax1.set_ylabel('elevation (m)')
ax2.plot(xx / 1.0e3, Phi.dat.data * secpera, color='C2', label=r'$\Phi(s)$')
ax2.legend(loc='upper right')
ax2.set_ylabel(r'$\Phi$ (m a-1)')
ax2.grid(visible=True)
plt.xlabel('x (km)')
plt.show()

