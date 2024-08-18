'''Solve Poisson on a half disc, - del^2 u = C.  The half disc is an extruded
mesh which is pinched-off on 2/3 of its domain, with zero area elements.
The height of the extruded mesh is
       / semicircle in (1,2)
  z = |
       \ zero in (0,1] union [2,3)
This is a strange way to solve a Poisson problem, but natural for Stokes on a glacier.

Current status: RUNS BUT SOLUTION IS FULL OF NANs.'''

from firedrake import *
from firedrake import utils
from firedrake.petsc import PETSc
printpar = PETSc.Sys.Print
import numpy as np

C = 10.0
N = 6

# build geometry by extruding
basemesh = IntervalMesh(3 * N, 3.0)  # 1d base mesh on (0,3)
xb = basemesh.coordinates.dat.data_ro
qb = np.abs(xb - 1.5)
zcircleb = np.zeros(np.shape(qb))
zcircleb[qb < 0.5] = np.sqrt(1.0 - (qb[qb < 0.5] / 0.5)**2)
mesh = ExtrudedMesh(basemesh, layers=N/2, layer_height=2/N)  # height=1.0 here
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(P1R)
s.dat.data[:] = zcircleb
Vcoord = mesh.coordinates.function_space()
x, z = SpatialCoordinate(mesh)
newcoord = Function(Vcoord).interpolate(as_vector([x, s * z]))
mesh.coordinates.assign(newcoord)  # now extruded mesh is rising sun shape

# spaces, weak form
H = FunctionSpace(mesh, 'CG', 1)
u = Function(H, name='u(x,y)')
v = TestFunction(H)
F = (dot(grad(u), grad(v)) - C * v) * dx

class PinchedC(DirichletBC):
    @utils.cached_property
    def nodes(self):
        '''Return the node indices where x is not in (1,2).'''
        return np.where(np.abs(xb - 1.5) >= 0.5)[0]

# solve
bc = [DirichletBC(H, Constant(0.0), (1, 2)),
      DirichletBC(H, Constant(0.0), 'top')]
pc = PinchedC(H, Constant(0.0), None)
print(pc.nodes)  # these are the expected nodes of the base mesh
solve(F == 0, u, bcs=bc.append(pc),
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

VTKFile("result.pvd").write(u)
