from firedrake import *
from stokesextruded import StokesExtruded, par_newton, par_mumps

def test_setup_2d_th():
    m, k = 2, 2
    basemesh = UnitIntervalMesh(m)         # 1d base mesh
    mesh = ExtrudedMesh(basemesh, m)       # quad elements
    se = StokesExtruded(mesh)
    udim, pdim = se.mixed_TaylorHood(kp=k)  # Q3 x Q2
    assert pdim == (k * m + 1)**se.dim
    assert udim == se.dim * ((k+1) * m + 1)**se.dim

def test_setup_3d_th():
    m, k = 2, 1
    basemesh = UnitSquareMesh(m, m)        # 2d base mesh
    mesh = ExtrudedMesh(basemesh, m)       # prism elements
    se = StokesExtruded(mesh)
    udim, pdim = se.mixed_TaylorHood(kp=k)  # "P2 x P1" but prism
    assert pdim == (k * m + 1)**se.dim
    assert udim == se.dim * ((k+1) * m + 1)**se.dim

def test_solve_2d_hydrostatic_mumps():
    m = 4
    basemesh = UnitIntervalMesh(m)
    mesh = ExtrudedMesh(basemesh, m)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    se.viscosity(1.0)
    se.body_force(Constant((0.0, -1.0)))
    se.dirichlet((1,2,'bottom'), Constant((0.0,0.0)))
    p = par_newton.copy()
    p.update(par_mumps)
    u, p = se.solve(par=p)
    _, z = SpatialCoordinate(mesh)
    assert norm(u) < 1.0e-10
    pexact = Function(p.function_space()).interpolate(1.0 - z)
    assert errornorm(pexact, p) < 1.0e-10
    #se.savesolution('result.pvd')

def test_solve_2d_slab_mumps():
    mx, mz = 6, 4
    L, H = 10.0, 1.0
    alpha = 0.5    # tilt in radians
    g, rho0, nu0 = 9.8, 1.0, 1.0
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    se.viscosity(nu0)
    se.body_force(Constant((rho0 * g* sin(alpha), - rho0 * g * cos(alpha))))
    x, z = SpatialCoordinate(mesh)
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
    C0 = rho0 * g * sin(alpha) / nu0
    u_in = Function(se.V).interpolate(as_vector([C0 * z * (H - z / 2), Constant(0.0)]))
    se.dirichlet((1,), u_in)
    # FIXME add downstream boundary nonhomogeneous Neumann integral
    p = par_newton.copy()
    p.update(par_mumps)
    u, p = se.solve(par=p)
    # FIXME from here
    assert True
    #assert norm(u) < 1.0e-10
    #pexact = Function(p.function_space()).interpolate(1.0 - z)
    #assert errornorm(pexact, p) < 1.0e-10
    #se.savesolution('result.pvd')

if __name__ == "__main__":
   test_solve_2d_slab_mumps()
