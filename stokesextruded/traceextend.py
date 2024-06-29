'''Tools for evaluating top/bottom traces on extruded meshes, and to
extend fields from base meshes to extruded meshes.'''

import firedrake as fd

def extend_p1_from_basemesh(mesh, f):
    '''On an extruded mesh, extend a P1 function f(x), defined for x
    in basemesh, to the extruded (x,z) mesh.  Returns a P1 function
    on mesh, in the 'R' constant-in-the-vertical space.'''
    P1R = fd.FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    fextend = fd.Function(P1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def trace_top(basemesh, mesh, f, bottom=False):
    '''On an extruded mesh, compute the trace of any scalar function f(x,z)
    along the top.  (Trace along bottom if set to True.)  Returns a P1
    function on basemesh.'''
    P1R = fd.FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    v = fd.TestFunction(P1R)
    ftop_cof = fd.Cofunction(P1R.dual())
    # re assemble() commands: "Cofunction(X).assemble(Y)" syntax not allowed for some reason
    if bottom:
        fd.assemble(f * v * fd.ds_b, tensor=ftop_cof)
    else:
        fd.assemble(f * v * fd.ds_t, tensor=ftop_cof)
    ftop_fcn = ftop_cof.riesz_representation(riesz_map='L2')
    P1base = fd.FunctionSpace(basemesh, 'CG', 1)
    ftop = fd.Function(P1base)
    ftop.dat.data[:] = ftop_fcn.dat.data_ro[:]
    return ftop

def trace_bottom(basemesh, mesh, f):
    return trace_top(basemesh, mesh, f, bottom=True)
