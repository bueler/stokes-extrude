'''Tools for evaluating top/bottom traces on extruded meshes, and to
extend fields from base meshes to extruded meshes.  It is believed,
and partially tested, that these run correctly in parallel.'''

import firedrake as fd

def extend_p1_from_basemesh(mesh, f):
    '''On an extruded mesh, extend a P1 function f(x), defined for x
    in basemesh, to the extruded (x,z) mesh.  Returns a P1 function
    on mesh, in the 'R' constant-in-the-vertical space.'''
    P1R = fd.FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    fextend = fd.Function(P1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def trace_scalar_to_p1(basemesh, mesh, f, surface='top', nointerpolate=False):
    '''On an extruded mesh, compute the trace of any scalar function f
    along the surface='top' or surface='bottom' boundary at the P1 nodes.
    Set nointerpolate=True if f is already P1.  Returns a P1 function on
    basemesh.'''
    assert surface in ['top', 'bottom']
    P1 = fd.FunctionSpace(mesh, 'CG', 1)
    if nointerpolate:
        fP1 = f
    else:
        fP1 = fd.Function(P1).interpolate(f)
    bc = fd.DirichletBC(P1, 0.0, surface)
    P1basemesh = fd.FunctionSpace(basemesh, 'CG', 1)
    fbm = fd.Function(P1basemesh)
    fbm.dat.data_with_halos[:] = fP1.dat.data_with_halos[bc.nodes]
    return fbm

def trace_vector_to_p2(basemesh, mesh, u, surface='top', dim=2, nointerpolate=False):
    '''On an extruded mesh, compute the trace of any vector-valued function
    u along the surface='top' or surface='bottom' boundary at the P2 nodes.
    Set nointerpolate=True if u is already P2.  Returns a P2 vector-valued
    function with dim=dim on basemesh.'''
    assert surface in ['top', 'bottom']
    P2V = fd.VectorFunctionSpace(mesh, 'CG', 2, dim=dim)
    if nointerpolate:
        uP2 = u
    else:
        uP2 = fd.Function(P2V).interpolate(u)
    bc = fd.DirichletBC(P2V, 0.0, surface)
    P2Vbasemesh = fd.VectorFunctionSpace(basemesh, 'CG', 2, dim=dim)
    ubm = fd.Function(P2Vbasemesh)
    ubm.dat.data_with_halos[:] = uP2.dat.data_with_halos[bc.nodes]
    return ubm
