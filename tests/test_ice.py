from firedrake import *
from stokesextruded import *

def _setup_physics_2d_iceslab(mesh, se, L, H, alpha):
    # essentially same settings as slab-on-slope example in
    #   https://github.com/bueler/mccarthy/tree/master/stokes
    secpera = 31556926.0    # seconds per year
    g, rho = 9.81, 910.0    # m s-2, kg m-3
    nglen = 3.0
    A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
    B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
    eps = 0.01
    Dtyp = 2.0 / secpera    # 2 a-1
    se.body_force(Constant((rho * g * sin(alpha), - rho * g * cos(alpha))))
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
    _, z = SpatialCoordinate(mesh)
    C = (2.0 / (nglen + 1.0)) \
        * (rho * g * sin(alpha) / B3)**nglen
    uin = as_vector([C * (H**(nglen + 1.0) - (H - z)**(nglen + 1.0)),
                     0.0])
    se.dirichlet((1,), uin)
    stressout = as_vector([- rho * g * cos(alpha) * (H - z),
                           rho * g * sin(alpha) * (H - z)])
    se.neumann((2,), stressout)
    def D(w):
        return 0.5 * (grad(w) + grad(w).T)
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
    rr = 1.0 / nglen - 1.0
    F = ( inner(B3 * Du2**(rr / 2.0) * D(u), D(v)) \
              - p * div(v) - div(u) * q - inner(se.f_body, v) ) * dx(degree=4)
    return F

def test_solve_2d_iceslab_mumps():
    mx, mz = 20, 5
    L, H = 3000.0, 400.0
    alpha = 0.1   # radians
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    F = _setup_physics_2d_iceslab(mesh, se, L, H, alpha)
    params = SolverParams['newton']
    params.update(SolverParams['mumps'])
    params['snes_linesearch_type'] = 'bt'
    #params['snes_converged_reason'] = None
    #params['snes_monitor'] = None
    u, p = se.solve(par=params, F=F)
    #se.savesolution('result.pvd')
    assert se.solver.snes.getIterationNumber() < 15
    g, rho = 9.81, 910.0
    _, z = SpatialCoordinate(mesh)
    pexact = Function(p.function_space()).interpolate(rho * g * (H - z))
    #print(errornorm(pexact, p) / norm(pexact))
    assert errornorm(pexact, p) / norm(pexact) < 0.01

def test_solve_2d_iceslab_mumps_dg():
    mx, mz = 20, 5
    L, H = 3000.0, 400.0
    alpha = 0.1   # radians
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_PkDG() # only change from test_solve_2d_iceslab_mumps()
    F = _setup_physics_2d_iceslab(mesh, se, L, H, alpha)
    params = SolverParams['newton']
    params.update(SolverParams['mumps'])
    params['snes_linesearch_type'] = 'bt'
    #params['snes_converged_reason'] = None
    #params['snes_monitor'] = None
    u, p = se.solve(par=params, F=F)
    #se.savesolution('resultdg.pvd')
    assert se.solver.snes.getIterationNumber() < 15
    g, rho = 9.81, 910.0
    _, z = SpatialCoordinate(mesh)
    pexact = Function(p.function_space()).interpolate(rho * g * (H - z))
    #print(errornorm(pexact, p) / norm(pexact))
    assert errornorm(pexact, p) / norm(pexact) < 0.01

if __name__ == "__main__":
    test_solve_2d_iceslab_mumps()
    test_solve_2d_iceslab_mumps_dg()
