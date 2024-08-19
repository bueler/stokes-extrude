'''Class for solving the Stokes problems on extruded meshes.'''

import numpy as np
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print

def _D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)

# a "pinch column" is one with zero mesh height (layer thickness)

class _PinchColumnPressure(fd.DirichletBC):
    def __init__(self, V, bR, tR, htol=1.0):
        self.bR = bR
        self.tR = tR
        self.htol = htol
        super().__init__(V, fd.Constant(0.0), None)

    @fd.utils.cached_property
    def nodes(self):
        # return P1 nodes in columns with surface elevation less than 1.0 meter
        h = fd.Function(self.function_space()).interpolate(self.tR - self.bR)
        return np.where(h.dat.data_ro < self.htol)[0]

class _PinchColumnVelocity(fd.DirichletBC):
    def __init__(self, V, bR, tR, htol=1.0, dim=2):
        self.bR = bR
        self.tR = tR
        self.htol = htol
        self.dim = dim
        zerovec = fd.as_vector([0.0, 0.0]) if dim == 2 else fd.as_vector([0.0, 0.0, 0.0])
        super().__init__(V, zerovec, None)

    @fd.utils.cached_property
    def nodes(self):
        # return vector P2 nodes in columns with height (thickness) less than htol
        # warning: assumes velocity space is P2
        P2scalar = fd.FunctionSpace(self.function_space().mesh(), 'CG', 2)
        h = fd.Function(P2scalar).interpolate(self.tR - self.bR)
        if self.dim == 2:
            hh = fd.Function(self.function_space()).interpolate(fd.as_vector([h, h]))
            return np.where(hh.dat.data_ro < self.htol)[0]
        else:
            hhh = fd.Function(self.function_space()).interpolate(fd.as_vector([h, h, h]))
            return np.where(hhh.dat.data_ro < self.htol)[0]

class StokesExtrude:
    '''Use Firedrake to solve a Stokes problem on an extruded mesh.
    A standard linear weak form is available, or the user can set that.
    Solvers can exploit a vertical mesh hierarchy for geometric multigrid.
    (Algebraic multigrid can be used over the coarse mesh.)

    Geometry functionality includes the ability to set the upper and lower
    elevation from functions on the base mesh, or from scalar constants.
    Zero-height columns are allowed.  (To do this call trivializepinchcolumns()
    after setting elevations and the mixed space.)

    One can use classical Taylor-Hood (P2 x P1), higher-order Taylor-Hood,
    or P2 x DG0.  (However, only the first-option is well-tested.)

    One can set a variety of Dirichlet and Neumann boundary conditions.
    The user is responsible for choosing a well-posed problem; e.g. at
    least some Dirichlet conditions should be set.

    See the documentation on extruded meshes at
        https://www.firedrakeproject.org/extruded-meshes.html'''

    def __init__(self, mesh):
        self.mesh = mesh
        self.bdim = mesh.cell_dimension()[0]
        self.dim = sum(mesh.cell_dimension())
        self.P1R = fd.FunctionSpace(self.mesh, 'P', 1, vfamily='R', vdegree=0)
        self.tR = fd.Constant(1.0)
        self.bR = fd.Constant(0.0)
        self.bcs = []
        self.F_neumann = []
        self.Z = None
        self.up = None
        self.nu = None
        self.f_body = None

    def set_elevations(self, bottom, top):
        # warning: assumes base mesh is P1
        # warning: assumes original z is 0 for bottom and 1 for top
        # warning: assumes bottom < top
        if np.isscalar(bottom):
            self.bR = fd.Constant(bottom)
        else:
            self.bR = fd.Function(self.P1R)
            self.bR.dat.data[:] = bottom.dat.data_ro
        if np.isscalar(top):
            self.tR = fd.Constant(top)
        else:
            self.tR = fd.Function(self.P1R)
            self.tR.dat.data[:] = top.dat.data_ro
        Vcoord = self.mesh.coordinates.function_space()
        x = fd.SpatialCoordinate(self.mesh)
        newz = self.bR + (self.tR - self.bR) * x[self.bdim]
        if self.bdim == 1:
            newcoord = fd.Function(Vcoord).interpolate(fd.as_vector([x[0], newz]))
        else:
            newcoord = fd.Function(Vcoord).interpolate(fd.as_vector([x[0], x[1], newz]))
        self.mesh.coordinates.assign(newcoord)

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

    def _addcondition(self, obj):
        self.bcs += [ obj ]  # append to list

    def trivializepinchcolumns(self, htol=1.0):
        # warning: call after set_elevations()
        # warning: call after setting mixed space
        self.pinchU = _PinchColumnVelocity(self.Z.sub(0), self.bR, self.tR, htol=htol, dim=self.dim)
        self._addcondition(self.pinchU)
        self.pinchP = _PinchColumnPressure(self.Z.sub(1), self.bR, self.tR, htol=htol)
        self._addcondition(self.pinchP)

    def dirichlet(self, ind, val):
        self._addcondition(fd.DirichletBC(self.Z.sub(0), val, ind))

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
            appctx = {'stokesextrude_nu': self.nu}
        else:
            appctx.update({'stokesextrude_nu': self.nu})
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
