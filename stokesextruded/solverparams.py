"""Dictionary of dictionaries with PETSc solver parameters,
suitable for Stokes problems on extruded meshes."""

import firedrake as fd


class pc_Mass(fd.AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        actx = self.get_appctx(pc)  # appctx is kwarg to StokesExtruded.solve()
        nu = actx.get("stokesextruded_nu")  # breaks if this key not in dict.
        a = (1.0 / nu) * fd.inner(test, trial) * fd.dx
        bcs = None
        return (a, bcs)


SolverParams = {
    "newton": {  # Newton solve
        "snes_linesearch_type": "basic",
        "snes_max_it": 200,
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 0.0,
    },
    "mumps": {  # Newton steps by LU
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_shift_type": "inblocks",
        "pc_factor_mat_solver_type": "mumps",
    },
    "schur_nonscalable":  # Newton steps by GMRES + Schur with full formation and
    #   inversion in solving the Schur complement, and LU on both blocks
    {
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "pc_fieldsplit_schur_precondition": "full",  # nonscalable inversion here
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",  # LU on u/u block
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "lu",  # LU on Schur block
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    },
    "schur_nonscalable_mass":  # Newton steps by GMRES + Schur with mass-matrix preconditioning,
    #   but with LU on A00 block
    {
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "pc_fieldsplit_schur_precondition": "a11",  # the default
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "stokesextruded.pc_Mass",
        "fieldsplit_1_aux_pc_type": "bjacobi",
        "fieldsplit_1_aux_sub_pc_type": "icc",
    },
    "schur_hypre_mass":  # Newton steps by GMRES + Schur with mass-matrix preconditioning,
    #   and with hypre algebraic multigrid on A00 block
    # note one can see A00 mat with:  -pc_hypre_mat_view :foo.m:ascii_matlab
    {
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "pc_fieldsplit_schur_precondition": "a11",  # the default
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "stokesextruded.pc_Mass",
        "fieldsplit_1_aux_pc_type": "bjacobi",
        "fieldsplit_1_aux_sub_pc_type": "icc",
    },
    "schur_gmg_mass":  # Newton steps by GMRES + Schur with mass-matrix preconditioning,
    #   and with geometric multigrid on A00 block
    #   works with mesh built as follows (e.g.):
    #     bbmesh = [IntervalMesh()|RectangleMesh()]
    #     bhier = MeshHierarchy(bbmesh, levs - 1)
    #     mhier = ExtrudedMeshHierarchy(bhier, H, base_layer=bmz, refinement_ratio=2)
    #     mesh = mhier[-1]
    {
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "pc_fieldsplit_schur_precondition": "a11",  # the default
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "mg",
        "fieldsplit_0_mg_levels_ksp_type": "chebyshev",  #  the default
        "fieldsplit_0_mg_levels_pc_type": "sor",  # the default
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "stokesextruded.pc_Mass",
        "fieldsplit_1_aux_pc_type": "bjacobi",
        "fieldsplit_1_aux_sub_pc_type": "icc",
    },
    "schur_gmg_cgnone_mass":  # tentative state: unpreconditioned CG as smoother ... a few more
    # iterations than Cheb+SOR above, but just as fast?
    {
        "ksp_type": "fgmres",  # because CG+none as smoother is not fixed
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "pc_fieldsplit_schur_precondition": "a11",  # the default
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "mg",
        "fieldsplit_0_mg_levels_ksp_type": "cg",
        "fieldsplit_0_mg_levels_pc_type": "none",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "stokesextruded.pc_Mass",
        "fieldsplit_1_aux_pc_type": "bjacobi",
        "fieldsplit_1_aux_sub_pc_type": "icc",
    },
    "DEV_schur_gmgmf_mass": {  # NOTE: to head toward matrix-free application of GMG on A00 block, need to know that non-assembled (or minimally-assembled) PC works  #'mat_type': 'nest',  ???
        #'mat_type': 'matfree',
        "ksp_type": "fgmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "pc_fieldsplit_schur_precondition": "a11",
        #'fieldsplit_0_mat_type': 'matfree',
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "mg",
        "fieldsplit_0_mg_levels_ksp_type": "cg",
        "fieldsplit_0_mg_levels_pc_type": "none",
        #'fieldsplit_1_mat_type': 'aij',
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "stokesextruded.pc_Mass",
        "fieldsplit_1_aux_pc_type": "bjacobi",
        "fieldsplit_1_aux_sub_pc_type": "icc",
    },
}

# alternative to consider is "multigrid on outside and Schur complements
# as smoother on each level".  done for this problem (Stokes) at:
#   https://github.com/firedrakeproject/firedrake/blob/master/docs/notebooks/07-geometric-multigrid.ipynb

# "Matrix free FMG with Telescoping" done for Poisson at:
#   https://github.com/firedrakeproject/firedrake/blob/master/docs/notebooks/12-HPC_demo.ipynb
# to do this for Stokes: nest at top level or putting multigrid on outside?

# matfree for NS at:
#  https://github.com/firedrakeproject/firedrake/blob/master/demos/matrix_free/navier_stokes.py.rst

# FIXME Newton steps by GMG in vert, AMG in horizontal
