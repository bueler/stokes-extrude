'''Class for solving the Stokes problem on an extruded mesh.'''

import numpy as np
import firedrake as fd

def extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = fd.FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)

class StokesExtruded:
    '''Use Firedrake to solve a Stokes problem on an extruded mesh, exploiting a vertical mesh hierarchy for geometric multigrid in the vertical, and using algebraic multigrid over the coarse mesh.  See the documentation on extruded meshes at https://www.firedrakeproject.org/extruded-meshes.html'''

    def __init__(self, mesh):
        self.mesh = mesh

    def savesolution(self, name=None):
        ''' Save state and diagnostics into .pvd file.'''
        u.rename('velocity (m as-1)')
        p.rename('pressure (Pa)')
        print('saving u,p to %s' % name)
        fd.File(name).write(u,p)

    def TaylorHood(self, kp=kp, printsizes=False):
        # set up Taylor-Hood mixed method
        V = fd.VectorFunctionSpace(self.mesh, 'Lagrange', kp+1)
        W = fd.FunctionSpace(self.mesh, 'Lagrange', kp)
        if printsizes:
            print('StokesExtruded: dimensions n_u = %d, n_p = %d' % (V.dim(), W.dim()))
        Z = V * W
        up = fd.Function(Z)
        u, p = fd.split(up)       # scaled velocity, unscaled pressure
        v, q = fd.TestFunctions(Z)

    def solve(self):
        '''Solve the Stokes problem.'''

        # Stokes weak form
        fbody = fd.Constant((0.0, -1.0)) # FIXME
        F = ( fd.inner(2.0 * nu * D(u), D(v)) \
              - p * fd.div(v) - q * fd.div(u) \
              - fd.inner(fbody, v) ) * fd.dx

        # zero Dirichlet on base (and stress-free on top and cliffs)
        bcs = [ fd.DirichletBC(Z.sub(0), fd.Constant((0.0, 0.0)), 'bottom')]  # FIXME

        # Newton-LU solve Stokes   FIXME GMG in vert, AMG in horizontal
        par = {'snes_linesearch_type': 'bt',
               'snes_max_it': 200,
               'snes_rtol': 1.0e-8,
               'snes_atol': 1.0e-12,
               'snes_stol': 0.0,
               'ksp_type': 'preonly',
               'pc_type': 'lu',
               'pc_factor_shift_type': 'inblocks'}
        fd.solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

        # return everything needed post-solve
        u, p = up.split()
        return u, p
