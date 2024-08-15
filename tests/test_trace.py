from firedrake import *
from stokesextrude import *

def test_trace_extend_trace_2d():
    basemesh = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(basemesh, 3, layer_height=1.0 / 3)
    x, z = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, 'CG', 1)
    f = Function(P1).interpolate((x - z) * x)
    ftop = trace_scalar_to_p1(basemesh, mesh, f)
    P1base = FunctionSpace(basemesh, 'CG', 1)
    fext = extend_p1_from_basemesh(mesh, ftop)
    fdiff = Function(f.function_space()).interpolate(f - fext)
    assert norm(trace_scalar_to_p1(basemesh, mesh, fdiff)) < 1.0e-14

def test_extend_trace_3d():
    basemesh = UnitSquareMesh(3,3)
    mesh = ExtrudedMesh(basemesh, 2)
    x, y = SpatialCoordinate(basemesh)
    P1base = FunctionSpace(basemesh, 'CG', 1)
    fbase = Function(P1base).interpolate((x - y) * x + y * y)
    fext = extend_p1_from_basemesh(mesh, fbase)
    fbase2 = trace_scalar_to_p1(basemesh, mesh, fext, surface='bottom')
    assert errornorm(fbase, fbase2) < 1.0e-14

def test_extend_trace_3d_nointerpolate():
    basemesh = UnitSquareMesh(3,3)
    mesh = ExtrudedMesh(basemesh, 2)
    x, y = SpatialCoordinate(basemesh)
    P1base = FunctionSpace(basemesh, 'CG', 1)
    fbase = Function(P1base).interpolate((x - y) * x + y * y)
    fext = extend_p1_from_basemesh(mesh, fbase)
    P1 = FunctionSpace(mesh, 'CG', 1)
    fext2 = Function(P1).interpolate(fext)
    fbase2 = trace_scalar_to_p1(basemesh, mesh, fext2, surface='bottom', nointerpolate=True)
    assert errornorm(fbase, fbase2) < 1.0e-14

def test_trace_vector_2d():
    basemesh = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(basemesh, 3, layer_height=1.0 / 3)
    x, z = SpatialCoordinate(mesh)
    P1V = VectorFunctionSpace(mesh, 'CG', 1)
    u = Function(P1V).interpolate(as_vector([x, z]))
    utop = trace_vector_to_p2(basemesh, mesh, u)
    P2Vbm = VectorFunctionSpace(basemesh, 'CG', 2, dim=2)
    xb = SpatialCoordinate(basemesh)[0]
    udiffbm = Function(P2Vbm).interpolate(utop - as_vector([xb, 1.0]))
    assert norm(udiffbm) < 1.0e-14

# see also examples/partracedemo.py

if __name__ == "__main__":
    pass
    #test_trace_vector_2d()
