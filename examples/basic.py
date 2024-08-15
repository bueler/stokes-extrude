from firedrake import *
from stokesextrude import *
basemesh = UnitIntervalMesh(10)
mesh = ExtrudedMesh(basemesh, 4, layer_height=1.0 / 4)
se = StokesExtrude(mesh)
se.mixed_TaylorHood()
se.viscosity_constant(1.0)
se.body_force(Constant((1.0, -1.0)))
se.dirichlet(('bottom',), Constant((0.0,0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
u, p = se.solve(par=params)
se.savesolution('result.pvd')
