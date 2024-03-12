from firedrake import *
from stokesextruded import StokesExtruded

def test_setupmixed3d():
    m = 4
    basemesh = UnitSquareMesh(m, m)
    mesh = ExtrudedMesh(basemesh, m)       # prism elements
    se = StokesExtruded(mesh)
    udim, pdim = se.setupTaylorHood(kp=1)  # "P2xP1" but prism
    assert pdim == (m + 1)**se.dim
    assert udim == se.dim * (2 * m + 1)**se.dim

def test_solvelinear2dhydrostatic():
    m = 4
    basemesh = UnitIntervalMesh(m)
    mesh = ExtrudedMesh(basemesh, m)  # Q1 elements
    se = StokesExtruded(mesh)
    se.setupTaylorHood(kp=1)          # Q2xQ1"
    udim, pdim = se.setupTaylorHood(kp=1)
    assert pdim == (m + 1)**se.dim
    assert udim == se.dim * (2 * m + 1)**se.dim
    se.dirichlet((1,2,'bottom'),Constant((0.0,0.0)))
    se.solve()
    x, z = SpatialCoordinate(mesh)
    u, p = se.up.subfunctions[0], se.up.subfunctions[1]
    assert norm(u) < 1.0e-10
    pexact = Function(p.function_space()).interpolate(1.0 - z)
    assert errornorm(pexact, p)
    #se.savesolution('result.pvd')

if __name__ == "__main__":
   test_solvelinear2dhydrostatic()
