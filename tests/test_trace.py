from firedrake import *
from stokesextruded import *

def test_trace_extend_trace_2d():
    basemesh = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(basemesh, 3, layer_height=1.0 / 3)
    x, z = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, 'CG', 1)
    f = Function(P1).interpolate((x - z) * x)
    ftop = trace_top(basemesh, mesh, f)
    P1base = FunctionSpace(basemesh, 'CG', 1)
    fext = extend_p1_from_basemesh(mesh, ftop)
    fdiff = Function(f.function_space()).interpolate(f - fext)
    assert norm(trace_top(basemesh, mesh, fdiff)) < 1.0e-14

def test_extend_trace_3d():
    basemesh = UnitSquareMesh(3,3)
    mesh = ExtrudedMesh(basemesh, 2)
    x, y = SpatialCoordinate(basemesh)
    P1base = FunctionSpace(basemesh, 'CG', 1)
    fbase = Function(P1base).interpolate((x - y) * x + y * y)
    fext = extend_p1_from_basemesh(mesh, fbase)
    fbase2 = trace_bottom(basemesh, mesh, fext)
    print(errornorm(fbase, fbase2))
    assert errornorm(fbase, fbase2) < 1.0e-14
