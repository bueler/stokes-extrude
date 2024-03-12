'''Class for solving the Stokes problem on an extruded mesh.'''

import numpy as np
import firedrake as fd

# Newton solve Stokes
par_defaults = { \
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

    def setupTaylorHood(self, kp=1):
        # set up Taylor-Hood mixed method
        self.V = fd.VectorFunctionSpace(self.mesh, 'Lagrange', kp+1)
        self.W = fd.FunctionSpace(self.mesh, 'Lagrange', kp)
        self.Z = self.V * self.W
        self.up = fd.Function(self.Z)
        return self.V.dim(), self.W.dim()

    def dirichlet(self, ind, val):
        self.bcs = self.bcs + [ fd.DirichletBC(self.Z.sub(0), val, ind) ]

    def solve(self, par=None):
        '''Solve the Stokes problem.'''
        # body force FIXME
        if self.dim == 2:
            fbody = fd.Constant((0.0, -1.0))
        elif self.dim == 3:
            fbody = fd.Constant((0.0, 0.0, -1.0))
        else:
            assert ValueError
        # Stokes weak form
        nu = 1.0  # FIXME
        u, p = fd.split(self.up)
        v, q = fd.TestFunctions(self.Z)
        self.F = ( fd.inner(2.0 * nu * _D(u), _D(v)) \
                 - p * fd.div(v) - q * fd.div(u) \
                 - fd.inner(fbody, v) ) * fd.dx
        # solve
        fd.solve(self.F == 0, self.up, bcs=self.bcs,
                 options_prefix='s', solver_parameters=par)
        return u, p

    def savesolution(self, name=None):
        ''' Save u, p solution into .pvd file.'''
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        u.rename('velocity (m s-1)')
        p.rename('pressure (Pa)')
        print('saving u,p to %s' % name)
        fd.File(name).write(u,p)
