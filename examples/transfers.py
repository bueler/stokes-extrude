from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *

basemesh = UnitIntervalMesh(10)
mesh = ExtrudedMesh(basemesh, 4, layer_height=1.0 / 4)
se = StokesExtruded(mesh)
se.mixed_TaylorHood()
se.viscosity_constant(1.0)
se.body_force(Constant((1.0, -1.0)))
se.dirichlet(('bottom',), Constant((0.0,0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
_, p = se.solve(par=params)
p.rename('pressure p(x,z)')

ptop = trace_top(basemesh, mesh, p)
if False:
    xx = basemesh.coordinates.dat.data
    import matplotlib.pyplot as plt
    plt.plot(xx, ptop.dat.data, color="k")
    plt.xlabel('x')
    plt.title('p(x,z=1)')
    plt.show()

x, = SpatialCoordinate(basemesh)
P1base = FunctionSpace(basemesh, 'CG', 1)
pext = extend_p1_from_basemesh(mesh, ptop)
pext.rename('pext(x,z) = (p(x,z=1) extended to (x,z))')
pdiff = Function(p.function_space()).interpolate(p - pext)
pdiff.rename('pdiff = p(x,z) - pext(x,z)')
VTKFile('result.pvd').write(p,pext,pdiff)

assert norm(trace_top(basemesh, mesh, pdiff)) < 1.0e-14
