from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *

def extend_from_basemesh(mesh, f):
    '''On an extruded mesh, extend a P1 function f(x), defined for x
    in basemesh, to the extruded (x,z) mesh.  Returns a P1 function
    on mesh, in the 'R' constant-in-the-vertical space.'''
    P1R = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    fextend = Function(P1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def trace_top(basemesh, mesh, f, bottom=False):
    '''On an extruded mesh, compute the trace of a scalar function f(x,z)
    along the top.  (Trace along bottom if set to True.)  Returns a P1
    function on basemesh.'''
    P1R = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    v = TestFunction(P1R)
    ftop_cof = Cofunction(P1R.dual())
    if bottom:
        assemble(f * v * ds_b, tensor=ftop_cof)
    else:
        # re next line: "Cofunction(X).assemble(Y)" syntax not allowed for some reason
        assemble(f * v * ds_t, tensor=ftop_cof)
    ftop_fcn = ftop_cof.riesz_representation(riesz_map='L2')
    P1base = FunctionSpace(basemesh, 'CG', 1)
    ftop = Function(P1base)
    ftop.dat.data[:] = ftop_fcn.dat.data_ro[:]
    return ftop

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
pext = extend_from_basemesh(mesh, ptop)
pext.rename('pext(x,z) = (p(x,z=1) extended to (x,z))')
pdiff = Function(p.function_space()).interpolate(p - pext)
pdiff.rename('pdiff = p(x,z) - pext(x,z)')
VTKFile('result.pvd').write(p,pext,pdiff)

assert norm(trace_top(basemesh, mesh, pdiff)) < 1.0e-14
