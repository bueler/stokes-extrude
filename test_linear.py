from firedrake import *
from stokesextruded import StokesExtruded, par_defaults, par_mumps

def test_setupmixed2d():
    m = 2
    basemesh = UnitIntervalMesh(m)
    mesh = ExtrudedMesh(basemesh, m)       # quad elements
    se = StokesExtruded(mesh)
    udim, pdim = se.setupTaylorHood(kp=2)  # Q3 x Q2
    assert pdim == (2 * m + 1)**se.dim
    assert udim == se.dim * (3 * m + 1)**se.dim

def test_setupmixed3d():
    m = 2
    basemesh = UnitSquareMesh(m, m)
    mesh = ExtrudedMesh(basemesh, m)       # prism elements
    se = StokesExtruded(mesh)
    udim, pdim = se.setupTaylorHood()      # "P2 x P1" but prism
    assert pdim == (m + 1)**se.dim
    assert udim == se.dim * (2 * m + 1)**se.dim

def test_solvelinear2dhydrostatic():
    m = 4
    basemesh = UnitIntervalMesh(m)
    mesh = ExtrudedMesh(basemesh, m)
    se = StokesExtruded(mesh)
    se.setupTaylorHood()
    se.dirichlet((1,2,'bottom'),Constant((0.0,0.0)))
    p = par_defaults.copy()
    p.update(par_mumps)
    u, p = se.solve(par=p)
    _, z = SpatialCoordinate(mesh)
    assert norm(u) < 1.0e-10
    pexact = Function(p.function_space()).interpolate(1.0 - z)
    assert errornorm(pexact, p) < 1.0e-10
    #se.savesolution('result.pvd')

if __name__ == "__main__":
   test_solvelinear2dhydrostatic()
