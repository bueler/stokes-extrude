# Attempt to solve Poisson  (- del^2 u = C)  on a disc, but within
# a square mesh.  The idea is to modify the weak form with an
# indicator function which is 1 where the dofs should be zero.
# It does not work well.

from firedrake import *

# mesh, spaces, source term
mesh = UnitSquareMesh(40, 40)
H = FunctionSpace(mesh, 'CG', 1)
u = Function(H, name='u(x,y)')
v = TestFunction(H)
C = 10.0
fsrc = Function(H).interpolate(Constant(C))

# element-wise indicator; intention: nodes should be pinned
# at u=0 if they are adjacent to indicated elements 
x, y = SpatialCoordinate(mesh)
r = sqrt((x - 0.5)**2 + (y - 0.5)**2)
DG0 = FunctionSpace(mesh, 'DG', 0)
indic = Function(DG0).interpolate(conditional(r > 0.5, 1.0, 0.0))

# attempt to weakly-enforce pinning
F = (1.0 - indic) * (dot(grad(u), grad(v)) - fsrc * v) * dx \
    + indic * u * v * dx

# solve
BCs = DirichletBC(H, Constant(0.0), (1, 2, 3, 4))
solve(F == 0, u, bcs=[BCs],
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

# output, with exact solution from Laplacian in polar coordinates; hopefully correct ...
uexact = Function(H).interpolate(conditional(r > 0.5, 0.0, (C / 4) * ((1 / 4) - r * r)))
uexact.rename('u_exact(x,y)')
VTKFile("result.pvd").write(u, uexact)
