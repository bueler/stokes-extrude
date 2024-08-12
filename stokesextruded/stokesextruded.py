'''Class for solving the Stokes problems on extruded meshes.'''

import numpy as np
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print

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

    def mixed_PkDG(self, ku=2, kp=1):
        self.V = fd.VectorFunctionSpace(self.mesh, 'Lagrange', ku)
        self.W = fd.FunctionSpace(self.mesh, 'DQ', kp)
        self.Z = self.V * self.W
        self.up = fd.Function(self.Z)
        return self.V.dim(), self.W.dim()

    def dirichlet(self, ind, val):
        self.bcs += [ fd.DirichletBC(self.Z.sub(0), val, ind) ]  # append to list

    def neumann(self, ind, val):
        self.F_neumann += [ (val, ind) ]  # append to list

    def body_force(self, f):
        self.f_body = f

    def viscosity_constant(self, nu):
        self.nu = nu

    def _F_linear(self, u, p, v, q):
        FF = ( fd.inner(2.0 * self.nu * _D(u), _D(v)) \
               - p * fd.div(v) - q * fd.div(u) \
               - fd.inner(self.f_body, v) ) * fd.dx  # FIXME degree?
        return FF

    def solve(self, F=None, par=None, appctx=None):
        '''Define weak form and solve the Stokes problem.'''
        assert self.Z != None
        assert self.up != None
        assert self.f_body != None
        assert len(self.bcs) > 0          # requires some Dirichlet boundary
        u, p = fd.split(self.up)          # get UFL objects
        v, q = fd.TestFunctions(self.Z)
        if F == None:
            assert self.nu != None
            self.F = self._F_linear(u, p, v, q)
        else:
            self.F = F
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
