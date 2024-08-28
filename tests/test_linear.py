from firedrake import *
from stokesextrude import *

def revealfullname(obj):
    # https://petsc.org/release/manualpages/PC/PCPythonSetType/
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    clas = obj.__class__
    module = clas.__module__
    if module == 'builtins':
        return clas.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + clas.__qualname__

def test_pc_mass_name():
    assert revealfullname(pc_Mass()) == 'stokesextrude.solverparams.pc_Mass'

def test_setup_2d_th():
    m, k = 2, 2
    basemesh = UnitIntervalMesh(m)           # 1d base mesh
    se = StokesExtrude(basemesh, mz=m)       # quad elements
    udim, pdim = se.mixed_TaylorHood(kp=k)   # Q3 x Q2
    assert pdim == (k * m + 1)**se.dim
    assert udim == se.dim * ((k+1) * m + 1)**se.dim

def test_setup_3d_th():
    m, k = 2, 1
    basemesh = UnitSquareMesh(m, m)        # 2d base mesh
    se = StokesExtrude(basemesh, mz=m)     # prism elements
    udim, pdim = se.mixed_TaylorHood(kp=k) # "P2 x P1" but prism
    assert pdim == (k * m + 1)**se.dim
    assert udim == se.dim * ((k+1) * m + 1)**se.dim

def test_solve_3d_hydrostatic_mumps():
    m = 3
    basemesh = UnitSquareMesh(m, m)
    se = StokesExtrude(basemesh, mz=m)   # prism elements
    se.mixed_TaylorHood()
    se.viscosity_constant(1.0)
    se.body_force(Constant((0.0, 0.0, -1.0)))
    se.dirichlet((1,2,3,4,'bottom'), Constant((0.0, 0.0, 0.0)))
    params = SolverParams['newton']
    params.update(SolverParams['mumps'])
    u, p = se.solve(par=params)
    _, _, z = SpatialCoordinate(se.mesh)
    assert norm(u) < 1.0e-10
    pexact = Function(p.function_space()).interpolate(1.0 - z)
    assert errornorm(pexact, p) < 1.0e-10

def _setup_physics_2d_slab(se, L, H):
    alpha = 0.5    # tilt in radians
    g, rho0, nu0 = 9.8, 1.0, 1.0
    se.viscosity_constant(nu0)
    CC = rho0 * g
    se.body_force(Constant((CC * sin(alpha), - CC * cos(alpha))))
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
    C0 = CC * sin(alpha) / nu0
    _, z = SpatialCoordinate(se.mesh)
    u_in = as_vector([C0 * z * (H - z / 2), Constant(0.0)])
    se.dirichlet((1,), u_in)
    stress_out = as_vector([- CC * cos(alpha) * (H - z),
                            CC * sin(alpha) * (H - z)])
    se.neumann((2,), stress_out)

def _exact_2d_slab(mesh, V, W, L, H):
    alpha = 0.5    # tilt in radians
    g, rho0, nu0 = 9.8, 1.0, 1.0
    CC = rho0 * g
    C0 = CC * sin(alpha) / nu0
    _, z = SpatialCoordinate(mesh)
    uexact = Function(V).interpolate(as_vector([C0 * z * (H - z / 2), Constant(0.0)]))
    pexact = Function(W).interpolate(CC * cos(alpha) * (H - z))
    return uexact, pexact

def test_solve_2d_slab_mumps():
    mx, mz = 6, 4
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    se = StokesExtrude(basemesh, mz=mz)
    se.reset_elevations(Constant(0.0), Constant(H))
    se.mixed_TaylorHood()
    _setup_physics_2d_slab(se, L, H)
    params = SolverParams['newton']
    params.update(SolverParams['mumps'])
    u, p = se.solve(par=params)
    assert se.solver.snes.getIterationNumber() == 1
    uexact, pexact = _exact_2d_slab(se.mesh, u.function_space(), p.function_space(), L, H)
    assert errornorm(uexact, u) < 1.0e-10
    assert errornorm(pexact, p) < 1.0e-10

def test_solve_2d_slab_schur_nonscalable():
    mx, mz = 6, 4
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    se = StokesExtrude(basemesh, mz=mz)
    se.reset_elevations(Constant(0.0), Constant(H))
    se.mixed_TaylorHood()
    _setup_physics_2d_slab(se, L, H)
    params = SolverParams['newton']
    params.update(SolverParams['schur_nonscalable'])
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() == 2  # guaranteed by theory
    assert se.solver.snes.getIterationNumber() == 1
    uexact, pexact = _exact_2d_slab(se.mesh, u.function_space(), p.function_space(), L, H)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

def test_solve_2d_slab_schur_nonscalable_mass():
    mx, mz = 20, 2
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    se = StokesExtrude(basemesh, mz=mz)
    se.reset_elevations(Constant(0.0), Constant(H))
    se.mixed_TaylorHood()
    _setup_physics_2d_slab(se, L, H)
    params = SolverParams['newton']
    params.update(SolverParams['schur_nonscalable_mass'])
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() < 15
    assert se.solver.snes.getIterationNumber() == 2
    uexact, pexact = _exact_2d_slab(se.mesh, u.function_space(), p.function_space(), L, H)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

def test_solve_2d_slab_schur_hypre_mass():
    mx, mz = 20, 2
    L, H = 10.0, 1.0
    basemesh = IntervalMesh(mx, L)
    se = StokesExtrude(basemesh, mz=mz)
    se.reset_elevations(Constant(0.0), Constant(H))
    se.mixed_TaylorHood()
    _setup_physics_2d_slab(se, L, H)
    params = SolverParams['newton']
    params.update(SolverParams['schur_hypre_mass'])
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() < 30
    assert se.solver.snes.getIterationNumber() == 2
    uexact, pexact = _exact_2d_slab(se.mesh, u.function_space(), p.function_space(), L, H)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

def test_solve_2d_slab_schur_gmg_mass():
    mx, mz = 20, 2
    levs = 3
    L, H = 10.0, 1.0
    basebasemesh = IntervalMesh(mx, L)
    basehierarchy = MeshHierarchy(basebasemesh, levs - 1)
    meshhierarchy = ExtrudedMeshHierarchy(basehierarchy, H, base_layer=mz, refinement_ratio=2)
    mesh = meshhierarchy[-1]
    se = StokesExtrude(basehierarchy[-1], mz=mz*2**(levs-1), mesh=mesh)
    se.reset_elevations(Constant(0.0), Constant(H))
    se.mixed_TaylorHood()
    _setup_physics_2d_slab(se, L, H)
    params = SolverParams['newton']
    params.update(SolverParams['schur_gmg_mass'])
    u, p = se.solve(par=params)
    assert se.solver.snes.ksp.getIterationNumber() < 30
    assert se.solver.snes.getIterationNumber() == 2
    uexact, pexact = _exact_2d_slab(se.mesh, u.function_space(), p.function_space(), L, H)
    assert errornorm(uexact, u) < 1.0e-8
    assert errornorm(pexact, p) < 1.0e-8

def _halfdisc_zeroheight_mumps(zeroheight):
    mx, mz = 18, 4
    # 1d base mesh on (0,3)
    basemesh = IntervalMesh(3 * mx, 3.0)
    # numpy array: semi-circle on (1,2), but zero on (0,1) union (2,3)
    xb = basemesh.coordinates.dat.data_ro
    qb = np.abs(xb - 1.5)
    sb = np.zeros(np.shape(qb))
    sb[qb < 0.5] = np.sqrt(0.25 - qb[qb < 0.5]**2)
    # extrude mesh and set geometry
    se = StokesExtrude(basemesh, mz=mz, htol=1.0e-6)
    P1bm = FunctionSpace(basemesh, 'P', 1)
    s = Function(P1bm)
    s.dat.data[:] = sb
    se.reset_elevations(Constant(0.0), s)
    # solve Stokes
    se.mixed_TaylorHood()
    g, rho0, nu0 = 9.8, 1.0, 1.0
    se.viscosity_constant(nu0)
    se.body_force(Constant((0.0, - rho0 * g)))
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
    params = SolverParams['newton']
    params.update(SolverParams['mumps'])
    #params['ksp_converged_reason'] = None
    #params['snes_converged_reason'] = None
    _, p = se.solve(par=params, zeroheight=zeroheight)
    #print(norm(p))
    assert abs(norm(p) - 1.2188) < 1.0e-3
    assert se.solver.snes.ksp.getIterationNumber() == 1
    assert se.solver.snes.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.CONVERGED_ITS
    assert se.solver.snes.getIterationNumber() == 1
    assert se.solver.snes.getConvergedReason() == PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS
    return se

def test_zeroheight_mumps_indices():
    se = _halfdisc_zeroheight_mumps('indices')
    #se.savesolution(name='result_indices.pvd')

def test_zeroheight_mumps_bounds():
    se = _halfdisc_zeroheight_mumps('bounds')
    #se.savesolution(name='result_bounds.pvd')

if __name__ == "__main__":
    pass
    #test_pc_mass_name()
    #test_setup_2d_th()
    #test_setup_3d_th()
    #test_solve_2d_hydrostatic_mumps()
    #test_solve_2d_slab_mumps()
    #test_solve_2d_slab_schur_nonscalable()
    #test_solve_2d_slab_schur_nonscalable_mass()
    #test_solve_2d_slab_schur_hypre_mass()
    #test_solve_2d_slab_schur_gmg_mass()
    #test_zeroheight_mumps_indices()
    #test_zeroheight_mumps_bounds()
