'''Class for solving the Stokes problem on an extruded mesh.'''

import numpy as np
import firedrake as fd

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

nu_pc_Mass = 1.0

class pc_Mass(fd.AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        a = (1.0 / nu_pc_Mass) * fd.inner(test, trial) * fd.dx
        bcs = None
        return (a, bcs)

# Newton steps by GMRES + Schur with mass-matrix preconditioning
# FIXME doing hypre on A00 block is slower than LU on it
par_schur = { \
    'ksp_type': 'gmres',
    #'ksp_monitor': None,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'lower',
    'fieldsplit_0_ksp_type': 'preonly',
    #'fieldsplit_0_pc_type': 'lu',
    #'fieldsplit_0_pc_factor_mat_solver_type': 'mumps',
    'fieldsplit_0_pc_type': 'hypre',
    #'fieldsplit_0_pc_type': 'gamg',
    #'fieldsplit_0_pc_gamg_aggressive_square_graph': None,
    #'fieldsplit_0_pc_gamg_mis_k_minimum_degree_ordering': True,
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
        nu_pc_Mass = nu

    def solve(self, par=None):
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
        self.problem = fd.NonlinearVariationalProblem(self.F, self.up, bcs=self.bcs)
        self.solver = fd.NonlinearVariationalSolver(self.problem,
                 options_prefix='s', solver_parameters=par)
        self.solver.solve()
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        return u, p

    def savesolution(self, name=None):
        ''' Save u, p solution into .pvd file.'''
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        u.rename('velocity (m s-1)')
        p.rename('pressure (Pa)')
        print('saving u,p to %s' % name)
        fd.File(name).write(u,p)
