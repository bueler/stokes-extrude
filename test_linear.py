from firedrake import *
from firedrake.petsc import PETSc
from stokesextruded import StokesExtruded, \
                           par_newton, par_mumps, par_schur_nonscalable, \
                           pc_Mass, par_schur_nonscalable_mass, \
                           par_schur_hypre_mass

printpar = PETSc.Sys.Print

def revealfullname(o):
    # https://petsc.org/release/manualpages/PC/PCPythonSetType/
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

def test_pc_mass_name():
    bar = pc_Mass()
    assert revealfullname(bar) == 'stokesextruded.pc_Mass'

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
    params = par_newton.copy()
    params.update(par_mumps)
    u, p = se.solve(par=params)
    _, z = SpatialCoordinate(mesh)
    assert norm(u) < 1.0e-10
    pexact = Function(p.function_space()).interpolate(1.0 - z)
    assert errornorm(pexact, p) < 1.0e-10
    #se.savesolution('result.pvd')

def setup_physics_2d_slab(se, L, H, z):
    alpha = 0.5    # tilt in radians
    g, rho0, nu0 = 9.8, 1.0, 1.0
    se.viscosity(nu0)
    CC = rho0 * g
    se.body_force(Constant((CC * sin(alpha), - CC * cos(alpha))))
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
    C0 = CC * sin(alpha) / nu0
    u_in = as_vector([C0 * z * (H - z / 2), Constant(0.0)])
    se.dirichlet((1,), u_in)
    stress_out = as_vector([- CC * cos(alpha) * (H - z),
                            CC * sin(alpha) * (H - z)])
    se.neumann((2,), stress_out)

def exact_2d_slab(V, W, L, H, z):
    alpha = 0.5    # tilt in radians
    g, rho0, nu0 = 9.8, 1.0, 1.0
    CC = rho0 * g
    C0 = CC * sin(alpha) / nu0
    uexact = Function(V).interpolate(as_vector([C0 * z * (H - z / 2), Constant(0.0)]))
    pexact = Function(W).interpolate(CC * cos(alpha) * (H - z))
    return uexact, pexact

def test_solve_2d_slab_mumps():
    mx, mz = 6, 4
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    x, z = SpatialCoordinate(mesh)
    setup_physics_2d_slab(se, L, H, z)
    params = par_newton.copy()
    params.update(par_mumps)
    u, p = se.solve(par=params)
    assert se.solver.snes.getIterationNumber() == 1
    uexact, pexact = exact_2d_slab(u.function_space(), p.function_space(), L, H, z)
    assert errornorm(uexact, u) < 1.0e-10
    assert errornorm(pexact, p) < 1.0e-10
    #se.savesolution('result.pvd')

def test_solve_2d_slab_schur_nonscalable():
    mx, mz = 6, 4
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    x, z = SpatialCoordinate(mesh)
    setup_physics_2d_slab(se, L, H, z)
    params = par_newton.copy()
    params.update(par_schur_nonscalable)
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() == 2  # guaranteed by theory
    assert se.solver.snes.getIterationNumber() == 1
    uexact, pexact = exact_2d_slab(u.function_space(), p.function_space(), L, H, z)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

class local_pc_Mass(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        nu = 1.0
        a = (1.0 / nu) * inner(test, trial) * dx
        bcs = None
        return (a, bcs)

def test_solve_2d_slab_schur_nonscalable_mass():
    mx, mz = 20, 2
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    x, z = SpatialCoordinate(mesh)
    setup_physics_2d_slab(se, L, H, z)
    params = par_newton.copy()
    params.update(par_schur_nonscalable_mass)
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() < 15
    assert se.solver.snes.getIterationNumber() == 2
    uexact, pexact = exact_2d_slab(u.function_space(), p.function_space(), L, H, z)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

def test_solve_2d_slab_schur_hypre_mass():
    mx, mz = 20, 2
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    x, z = SpatialCoordinate(mesh)
    setup_physics_2d_slab(se, L, H, z)
    params = par_newton.copy()
    params.update(par_schur_hypre_mass)
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() < 30
    assert se.solver.snes.getIterationNumber() == 2
    uexact, pexact = exact_2d_slab(u.function_space(), p.function_space(), L, H, z)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

def DEV_test_solve_2d_slab_schur_hypre_mass():
    mx, mz = 20, 2
    #mx, mz = 60, 6
    #mx, mz = 180, 18
    #mx, mz = 540, 54
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    mesh = ExtrudedMesh(basemesh, mz, layer_height=H / mz)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    x, z = SpatialCoordinate(mesh)
    setup_physics_2d_slab(se, L, H, z)
    params = par_newton.copy()
    params.update(par_schur_hypre_mass)
    params.update({'snes_converged_reason': None,
                   'ksp_converged_reason': None})
    u, p = se.solve(par=params)
    se.savesolution('result.pvd')
    #assert se.solver.snes.ksp.getIterationNumber() < 30
    #assert se.solver.snes.getIterationNumber() == 2
    uexact, pexact = exact_2d_slab(u.function_space(), p.function_space(), L, H, z)
    #assert errornorm(uexact, u) < 1.0e-8
    #assert errornorm(pexact, p) < 1.0e-8
    printpar(errornorm(uexact, u))
    printpar(errornorm(pexact, p))

if __name__ == "__main__":
   #test_pc_mass_name()
   #test_setup_2d_th()
   #test_setup_3d_th()
   #test_solve_2d_hydrostatic_mumps()
   #test_solve_2d_slab_mumps()
   #test_solve_2d_slab_schur_nonscalable()
   #test_solve_2d_slab_schur_nonscalable_mass()
   #test_solve_2d_slab_schur_hypre_mass()
   DEV_test_solve_2d_slab_schur_hypre_mass()
