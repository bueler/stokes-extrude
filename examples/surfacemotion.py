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
Hmin = 100.0

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

P1 = FunctionSpace(mesh, 'P', 1)
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
               'snes_converged_reason': None,
               'ksp_view_mat': ':foo.m:ascii_matlab'})
u, p = se.solve(par=params, F=_form_2d_ice(mesh, se))
se.savesolution(name='result.pvd')

# FIXME trace_top() u to get u|_s
# FIXME trace_top() z to get s
# FIXME s.dx and n_s = (-s.dx,1)
# FIXME Phi = - u|_s . n_s

#ptop = trace_top(basemesh, mesh, p)
