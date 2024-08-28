'''This code does not use StokesExtrude.  It is a strange way to solve a
Poisson problem, but it illustrates the trivialization of zero-height extruded
columns, which is natural for Stokes on a glacier.

Solve Poisson equation (- del^2 u = C) on a half disc.  The half disc is an extruded
mesh which is pinched-off on 2/3 of its domain, with zero-area extruded elements.
Elements are quadrilaterals.  The height of the extruded mesh is
       / semicircle in (1,2)
  z = |
       \ zero in (0,1]+[2,3)
In parallel, it looks like one gets failure if one process owns only zero-area
elements, so this fails for P>3 processes.

Thanks to Colin Cotter and Lawrence Mitchell for help with this.'''

from firedrake import *
from firedrake import utils
from firedrake.petsc import PETSc
printpar = PETSc.Sys.Print
import numpy as np

C = 10.0
mx = 40
mz = 10
debug = False

# extruded mesh with constant height 1.0
basemesh = IntervalMesh(3 * mx, 3.0)  # 1d base mesh on (0,3)
mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)
mesh.topology_dm.viewFromOptions('-dm_view')  # note you only see the base mesh DMPlex

# set-up semi-circle geometry on (1,2)
xb = basemesh.coordinates.dat.data_ro
qb = np.abs(xb - 1.5)
zcircleb = np.zeros(np.shape(qb))
zcircleb[qb < 0.5] = np.sqrt(0.25 - qb[qb < 0.5]**2)
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(P1R)
s.dat.data[:] = zcircleb
Vcoord = mesh.coordinates.function_space()
x, z = SpatialCoordinate(mesh)
newcoord = Function(Vcoord).interpolate(as_vector([x, s * z]))
mesh.coordinates.assign(newcoord)  # now extruded mesh is rising sun shape

# spaces, weak form, boring
H = FunctionSpace(mesh, 'CG', 1)
u = Function(H, name='u(x,y)')
v = TestFunction(H)
F = (dot(grad(u), grad(v)) - C * v) * dx

# get extruded mesh node indices where x is not in (1,2)
class PinchedNodes(DirichletBC):
    @utils.cached_property
    def nodes(self):
        xy = self.function_space().mesh().coordinates.dat.data_ro
        return np.where(np.abs(xy[:,0] - 1.5) >= 0.5)[0]

# solve
bcs = [DirichletBC(H, Constant(0.0), (1, 2)),
       DirichletBC(H, Constant(0.0), 'top'),
       PinchedNodes(H, Constant(0.0), None)]
if debug: # reveal extruded mesh numberings
       print(DirichletBC(H, Constant(0.0), (1, 2)).nodes)
       print(DirichletBC(H, Constant(0.0), 'top').nodes)
       print(PinchedNodes(H, Constant(0.0), None).nodes)
solve(F == 0, u, bcs=bcs,
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

# numerical error and output
x, y = SpatialCoordinate(mesh)
r = sqrt((x - 1.5)**2 + (y - 0.0)**2)
uexact = Function(H).interpolate(conditional(r > 0.5, 0.0, (C / 4) * ((1 / 4) - r * r)))
uexact.rename('u_exact(x,y)')
printpar(f'mx,mz = {mx},{mz}: |u - u_exact|_2 / |u_exact|_2 = {errornorm(u, uexact):.2e}')
VTKFile("result.pvd").write(u, uexact)
