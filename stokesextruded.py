'''Class for solving the Stokes problem on an extruded mesh.'''

import numpy as np
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print

# Newton solve
par_newton = { \
    'snes_linesearch_type': 'basic',
    'snes_max_it': 200,
    'snes_rtol': 1.0e-8,
    'snes_atol': 1.0e-12,
    'snes_stol': 0.0,
    }

# Newton steps by LU
par_mumps = { \
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_shift_type': 'inblocks',
    'pc_factor_mat_solver_type': 'mumps',
    }

# Newton steps by GMRES + Schur with full formation and
#   inversion in solving the Schur complement, and LU on both blocks
par_schur_nonscalable = { \
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'pc_fieldsplit_schur_precondition': 'full',  # nonscalable inversion here
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'lu',                # LU on u/u block
    'fieldsplit_0_pc_factor_mat_solver_type': 'mumps',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'lu',                # LU on Schur block
    'fieldsplit_1_pc_factor_mat_solver_type': 'mumps',
    }

class pc_Mass(fd.AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        actx = self.get_appctx(pc)  # note appctx is kwarg to StokesExtruded.solve()
        nu = actx.get('stokesextruded_nu')  # breaks if this key not in dict.
        a = (1.0 / nu) * fd.inner(test, trial) * fd.dx
        bcs = None
        return (a, bcs)

# Newton steps by GMRES + Schur with mass-matrix preconditioning,
#   but with LU on A00 block
par_schur_nonscalable_mass = { \
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'pc_fieldsplit_schur_precondition': 'a11',  # the default
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'lu',
    'fieldsplit_0_pc_factor_mat_solver_type': 'mumps',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'python',
    'fieldsplit_1_pc_python_type': 'stokesextruded.pc_Mass',
    'fieldsplit_1_aux_pc_type': 'bjacobi',
    'fieldsplit_1_aux_sub_pc_type': 'icc',
    }

# Newton steps by GMRES + Schur with mass-matrix preconditioning,
#   and with hypre algebraic multigrid on A00 block
# note one can see A00 mat with:  -pc_hypre_mat_view :foo.m:ascii_matlab
par_schur_hypre_mass = { \
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'pc_fieldsplit_schur_precondition': 'a11',  # the default
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'hypre',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'python',
    'fieldsplit_1_pc_python_type': 'stokesextruded.pc_Mass',
    'fieldsplit_1_aux_pc_type': 'bjacobi',
    'fieldsplit_1_aux_sub_pc_type': 'icc',
    }

# Newton steps by GMRES + Schur with mass-matrix preconditioning,
#   and with geometric multigrid on A00 block
#   works with mesh built as follows (e.g.):
#     bbmesh = [IntervalMesh()|RectangleMesh()]
#     bhier = MeshHierarchy(bbmesh, levs - 1)
#     mhier = ExtrudedMeshHierarchy(bhier, H, base_layer=bmz, refinement_ratio=2)
#     mesh = mhier[-1]
par_schur_gmg_mass = { \
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'pc_fieldsplit_schur_precondition': 'a11',  # the default
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'mg',
    'fieldsplit_0_mg_levels_ksp_type': 'chebyshev', #  the default
    'fieldsplit_0_mg_levels_pc_type': 'sor',  # the default
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'python',
    'fieldsplit_1_pc_python_type': 'stokesextruded.pc_Mass',
    'fieldsplit_1_aux_pc_type': 'bjacobi',
    'fieldsplit_1_aux_sub_pc_type': 'icc',
    }

# tentative state: unpreconditioned CG as smoother ... a few more
# iterations than Cheb+SOR above, but just as fast?
par_schur_gmg_cgnone_mass = { \
    'ksp_type': 'fgmres',  # because CG+none as smoother is not fixed
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'pc_fieldsplit_schur_precondition': 'a11',  # the default
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'mg',
    'fieldsplit_0_mg_levels_ksp_type': 'cg',
    'fieldsplit_0_mg_levels_pc_type': 'none',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'python',
    'fieldsplit_1_pc_python_type': 'stokesextruded.pc_Mass',
    'fieldsplit_1_aux_pc_type': 'bjacobi',
    'fieldsplit_1_aux_sub_pc_type': 'icc',
    }

# alternative to consider is "multigrid on outside and Schur complements
# as smoother on each level".  done for this problem (Stokes) at:
#   https://github.com/firedrakeproject/firedrake/blob/master/docs/notebooks/07-geometric-multigrid.ipynb

# "Matrix free FMG with Telescoping" done for Poisson at:
#   https://github.com/firedrakeproject/firedrake/blob/master/docs/notebooks/12-HPC_demo.ipynb
# to do this for Stokes: nest at top level or putting multigrid on outside?

# matfree for NS at:
#  https://github.com/firedrakeproject/firedrake/blob/master/demos/matrix_free/navier_stokes.py.rst

# NOTE: to head toward matrix-free application of GMG on A00 block, need to know that non-assembled (or minimally-assembled) PC works
DEV_par_schur_gmgmf_mass = { \
    #'mat_type': 'nest',  ???
    #'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'pc_fieldsplit_schur_precondition': 'a11',
    #'fieldsplit_0_mat_type': 'matfree',
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'mg',
    'fieldsplit_0_mg_levels_ksp_type': 'cg',
    'fieldsplit_0_mg_levels_pc_type': 'none',
    #'fieldsplit_1_mat_type': 'aij',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'python',
    'fieldsplit_1_pc_python_type': 'stokesextruded.pc_Mass',
    'fieldsplit_1_aux_pc_type': 'bjacobi',
    'fieldsplit_1_aux_sub_pc_type': 'icc',
    }

# FIXME Newton steps by GMG in vert, AMG in horizontal

def _extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = fd.FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def _D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)

class StokesExtruded:
    '''Use Firedrake to solve a Stokes problem on an extruded mesh, exploiting a vertical mesh hierarchy for geometric multigrid in the vertical, and using algebraic multigrid over the coarse mesh.  See the documentation on extruded meshes at https://www.firedrakeproject.org/extruded-meshes.html'''

    def __init__(self, mesh):
        self.mesh = mesh
        self.bdim = mesh.cell_dimension()[0]
        self.dim = sum(mesh.cell_dimension())
        self.bcs = []
        self.F_neumann = []
        self.Z = None
        self.up = None
        self.nu = None
        self.f_body = None

    def mixed_TaylorHood(self, kp=1):
        # set up Taylor-Hood mixed method
        self.V = fd.VectorFunctionSpace(self.mesh, 'Lagrange', kp+1)
        self.W = fd.FunctionSpace(self.mesh, 'Lagrange', kp)
        self.Z = self.V * self.W
        self.up = fd.Function(self.Z)
        return self.V.dim(), self.W.dim()

    def dirichlet(self, ind, val):
        self.bcs += [ fd.DirichletBC(self.Z.sub(0), val, ind) ]  # append to list

    def neumann(self, ind, val):
        self.F_neumann += [ (val, ind) ]  # append to list

    def body_force(self, f):
        self.f_body = f

    def viscosity(self, nu):
        self.nu = nu

    def solve(self, par=None, appctx=None):
        '''Define weak form and solve the Stokes problem.'''
        assert self.Z != None
        assert self.up != None
        assert self.nu != None
        assert self.f_body != None
        assert len(self.bcs) > 0          # requires some Dirichlet boundary
        u, p = fd.split(self.up)          # get UFL objects
        v, q = fd.TestFunctions(self.Z)
        self.F = ( fd.inner(2.0 * self.nu * _D(u), _D(v)) \
                 - p * fd.div(v) - q * fd.div(u) \
                 - fd.inner(self.f_body, v) ) * fd.dx
        if len(self.F_neumann) > 0:
            # FIXME only implemented for side facets
            for ff in self.F_neumann:  # ff = (val, ind)
                self.F -= fd.inner(ff[0], v) * fd.ds_v(ff[1])
        self.problem = fd.NonlinearVariationalProblem( \
            self.F,
            self.up,
            bcs=self.bcs)
        if appctx == None:
            appctx = {'stokesextruded_nu': self.nu}
        else:
            appctx.update({'stokesextruded_nu': self.nu})
        self.solver = fd.NonlinearVariationalSolver( \
            self.problem,
            options_prefix='s',
            solver_parameters=par,
            appctx=appctx)
        self.solver.solve()
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        return u, p

    def savesolution(self, name=None):
        ''' Save u, p solution into .pvd file.'''
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        u.rename('velocity (m s-1)')
        p.rename('pressure (Pa)')
        if self.mesh.comm.size > 1:
            printpar('saving u,p,rank to %s' % name)
            rank = fd.Function(fd.FunctionSpace(self.mesh,'DG',0))
            rank.dat.data[:] = self.mesh.comm.rank
            rank.rename('rank')
            VTKFile(name).write(u,p,rank)
        else:
            print('saving u,p to %s' % name)
            VTKFile(name).write(u,p)
