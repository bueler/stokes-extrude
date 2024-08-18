'''Solve Poisson on a disc, - del^2 u = C, by pinning d.o.f.s
in a square mesh.  No claim this is a good numerical approach,
but it illustrates how to subclass DirichletBC so that you can
set arbitrary interior d.o.f.s.  Even works in parallel!'''
from firedrake import *
from firedrake import utils
from firedrake.petsc import PETSc
printpar = PETSc.Sys.Print
import numpy as np

C = 10.0
N = 40

# mesh, spaces, weak form
mesh = UnitSquareMesh(N, N)
H = FunctionSpace(mesh, 'CG', 1)
u = Function(H, name='u(x,y)')
v = TestFunction(H)
F = (dot(grad(u), grad(v)) - C * v) * dx

# subclass DirichletBC and change how the node list is set
#   (inspired by https://mailman.ic.ac.uk/pipermail/firedrake/2015-August/001005.html)
class IndicatedC(DirichletBC):
    @utils.cached_property
    def nodes(self):
        '''Return the node indices where r > 0.5.'''
        xy = self.function_space().mesh().coordinates.dat.data_ro
        xs = xy[:,0] - 0.5
        ys = xy[:,1] - 0.5
        r = np.sqrt(xs * xs + ys * ys)
        return np.where(r > 0.5)[0]

# solve
bc = DirichletBC(H, Constant(0.0), (1, 2, 3, 4))
ic = IndicatedC(H, Constant(0.0), None)
#print(ic.nodes)
solve(F == 0, u, bcs=[bc, ic],
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

# compare exact solution from Laplacian in polar coordinates
x, y = SpatialCoordinate(mesh)
r = sqrt((x - 0.5)**2 + (y - 0.5)**2)
uexact = Function(H).interpolate(conditional(r > 0.5, 0.0, (C / 4) * ((1 / 4) - r * r)))
printpar(f'N = {N}: |u - u_exact|_2 / |u_exact|_2 = {errornorm(u, uexact):.2e}')
VTKFile("result.pvd").write(u)
