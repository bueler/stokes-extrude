# FIXME rename and do something SKE related with this code

from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *

# FIXME set up Halfar profile in x
L = 1000.0e3

basemesh = IntervalMesh(36, -L, L)
mesh = ExtrudedMesh(basemesh, 6, layer_height=1.0 / 6)
se = StokesExtruded(mesh)
se.mixed_TaylorHood()
se.viscosity_constant(1.0)
se.body_force(Constant((1.0, -1.0)))
se.dirichlet(('bottom',), Constant((0.0,0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
u, p = se.solve(par=params)
se.savesolution(name='result.pvd')

# FIXME trace_top() u to get u|_s
# FIXME trace_top() z to get s
# FIXME s.dx and n_s = (-s.dx,1)
# FIXME Phi = - u|_s . n_s

#ptop = trace_top(basemesh, mesh, p)
