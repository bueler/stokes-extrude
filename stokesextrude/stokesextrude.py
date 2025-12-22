"""Class for solving Stokes problems on extruded meshes.  Documented
by the README.md.  See also the documentation on extruded meshes at
https://www.firedrakeproject.org/extruded-meshes.html"""

import numpy as np
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print

# a "pinch column" is one with zero mesh height (layer thickness)


class _PinchColumn(fd.DirichletBC):
    def __init__(self, V, g, sub_domain):
        self.ready = False
        super().__init__(V, g, sub_domain)

    def set_pinch_data(self, hier, bR, tR, htol=1.0):
        assert isinstance(bR, list)
        assert isinstance(tR, list)
        self.hier = hier
        self.levs = len(self.hier)
        self.bR = bR
        self.tR = tR
        self.htol = htol
        self.ready = True

    @fd.utils.cached_property
    def function_arg(self, g):
        # override this from the base class so as to avoid check which breaks
        #   when using Schur solvers
        self._function_arg = g


class _PinchColumnPressure(_PinchColumn):
    def __init__(self, V, g, sub_domain):
        super().__init__(V, fd.Constant(0.0), None)

    @fd.utils.cached_property
    def nodes(self):
        assert self.ready
        # find the right mesh in the hierarchy
        for j in range(self.levs):
            if self.function_space().mesh() == self.hier[j]:
                break
        # return P1 nodes in columns with surface elevation less than 1.0 meter
        h = fd.Function(self.function_space()).interpolate(self.tR[j] - self.bR[j])
        return np.where(h.dat.data_ro_with_halos < self.htol)[0]


class _PinchColumnVelocity(_PinchColumn):

    def __init__(self, V, g, sub_domain, dim=2):
        assert dim in [2, 3]
        self.dim = dim
        zerovec = (
            fd.as_vector([0.0, 0.0]) if dim == 2 else fd.as_vector([0.0, 0.0, 0.0])
        )
        super().__init__(V, zerovec, None)

    @fd.utils.cached_property
    def nodes(self):
        assert self.ready
        # find the right mesh in the hierarchy
        for j in range(self.levs):
            if self.function_space().mesh() == self.hier[j]:
                break
        # return vector P2 nodes in columns with height (thickness) less than htol
        # warning: assumes velocity space is P2
        P2scalar = fd.FunctionSpace(self.function_space().mesh(), "CG", 2)
        h = fd.Function(P2scalar).interpolate(self.tR[j] - self.bR[j])
        if self.dim == 2:
            hh = fd.Function(self.function_space()).interpolate(fd.as_vector([h, h]))
            return np.where(hh.dat.data_ro_with_halos < self.htol)[0]
        else:
            hhh = fd.Function(self.function_space()).interpolate(
                fd.as_vector([h, h, h])
            )
            return np.where(hhh.dat.data_ro_with_halos < self.htol)[0]


class StokesExtrude:
    def __init__(self, basemesh, mz=4, levs=1, htol=1.0):
        # save initialization arguments
        self._cmz = mz  # number of layers in (coarsest) extruded mesh
        self.levs = levs
        self.pinchhtol = htol
        # extruded mesh dimension
        bdim = basemesh.cell_dimension()
        assert np.isscalar(bdim)
        self.dim = bdim + 1
        # construct extruded mesh hierarchy
        #   if levs == 1 then hierarchy is list of length one
        #   also: copy original coordinates on each level
        if self.levs == 1:
            self.basehier = [
                basemesh,
            ]
            self.mesh = fd.ExtrudedMesh(
                basemesh, self._cmz, layer_height=1.0 / self._cmz
            )
            self.hier = [
                self.mesh,
            ]
            self.xorig = [
                self.mesh.coordinates.copy(deepcopy=True),
            ]
            self.P1R = [
                fd.FunctionSpace(self.mesh, "P", 1, vfamily="R", vdegree=0),
            ]
        else:
            assert np.isscalar(self.levs) and self.levs > 1
            # note basemesh is now the coarsest base mesh
            # generally StokesExtrude ignors self.basehier in methods, but
            #   it is useful to users needing basemesh coordinates
            self.basehier = fd.MeshHierarchy(basemesh, self.levs - 1)
            self.hier = fd.ExtrudedMeshHierarchy(
                self.basehier, 1.0, base_layer=self._cmz, refinement_ratio=2
            )
            self.mesh = self.hier[-1]
            self.xorig = [mesh.coordinates.copy(deepcopy=True) for mesh in self.hier]
            self.P1R = [
                fd.FunctionSpace(mesh, "P", 1, vfamily="R", vdegree=0)
                for mesh in self.hier
            ]
        # populate self.bR, self.tR with elevations compatible with "original coordinates"
        #   on each level
        self.reset_elevations(0.0, 1.0)
        # empty data on mixed space, viscosity model, and boundary conditions
        self.Z = None
        self.up = None
        self.nu = None
        self.dirbcs = []
        self.F_neumann = []

    def _validate_elevation_order(self):
        for j in range(self.levs):
            delta = fd.Function(self.P1R[j]).interpolate(self.tR[j] - self.bR[j])
            assert np.min(delta.dat.data) >= 0.0
        return None

    def reset_elevations(self, bottom, top):
        # warning: assumes bottom < top
        # first put bottom and top into lists of the right form
        if np.isscalar(bottom):
            self.bR = [fd.Constant(bottom) for j in range(self.levs)]
        elif isinstance(bottom, fd.Constant):
            self.bR = [bottom for j in range(self.levs)]
        elif isinstance(bottom, fd.Function) and self.levs == 1:
            self.bR = [
                fd.Function(self.P1R[0]),
            ]
            self.bR[0].dat.data_with_halos[:] = bottom.dat.data_ro_with_halos
        elif isinstance(bottom, list):
            assert len(bottom) == self.levs
            self.bR = [fd.Function(VR) for VR in self.P1R]
            for j in range(self.levs):
                self.bR[j].dat.data_with_halos[:] = bottom[j].dat.data_ro_with_halos
        else:
            raise NotImplementedError("bottom must be scalar, Constant, or list")
        if np.isscalar(top):
            self.tR = [fd.Constant(top) for j in range(self.levs)]
        elif isinstance(top, fd.Constant):
            self.tR = [top for j in range(self.levs)]
        elif isinstance(top, fd.Function) and self.levs == 1:
            self.tR = [
                fd.Function(self.P1R[0]),
            ]
            self.tR[0].dat.data_with_halos[:] = top.dat.data_ro_with_halos
        elif isinstance(top, list):
            assert len(top) == self.levs
            self.tR = [fd.Function(VR) for VR in self.P1R]
            for j in range(self.levs):
                self.tR[j].dat.data_with_halos[:] = top[j].dat.data_ro_with_halos
        else:
            raise NotImplementedError("top must be scalar, Constant, or list")
        # second, re-generate coordinates on each level
        for j in range(self.levs):
            xyzo = self.xorig[j]  # no copy; just a rename
            newz = self.bR[j] + (self.tR[j] - self.bR[j]) * xyzo[self.dim - 1]  # UFL
            Vcoord = self.hier[j].coordinates.function_space()
            if self.dim == 2:
                newcoord = fd.Function(Vcoord).interpolate(
                    fd.as_vector([xyzo[0], newz])
                )
            else:
                newcoord = fd.Function(Vcoord).interpolate(
                    fd.as_vector([xyzo[0], xyzo[1], newz])
                )
            self.hier[j].coordinates.assign(newcoord)
        # third, validate
        self._validate_elevation_order()

    def mixed_TaylorHood(self, k=1):
        """Set-up Taylor-Hood mixed elements P_{k+1} x P_k."""
        self.V = fd.VectorFunctionSpace(self.mesh, "Lagrange", k + 1)
        self.W = fd.FunctionSpace(self.mesh, "Lagrange", k)
        self.Z = self.V * self.W
        self.up = fd.Function(self.Z)
        return self.V.dim(), self.W.dim()

    def mixed_PkDG(self, ku=2, kp=1):
        """Set-up mixed elements P_ku x DG_kp.  Note DG = DQ on prisms etc."""
        self.V = fd.VectorFunctionSpace(self.mesh, "Lagrange", ku)
        self.W = fd.FunctionSpace(self.mesh, "DQ", kp)
        self.Z = self.V * self.W
        self.up = fd.Function(self.Z)
        return self.V.dim(), self.W.dim()

    def dirichlet(self, ind, val):
        self.dirbcs += [fd.DirichletBC(self.Z.sub(0), val, ind)]

    def neumann(self, ind, val):
        self.F_neumann += [(val, ind)]  # append to list

    def D(self, w):
        return 0.5 * (fd.grad(w) + fd.grad(w).T)

    def viscosity_constant(self, nu):
        self.nu = nu

    def solve(self, F=None, par=None, appctx=None, pinch=True):
        """Define weak form and solve the Stokes problem."""
        # check that we are ready
        assert self.Z is not None
        assert self.up is not None
        assert F is not None
        assert len(self.dirbcs) > 0  # require some Dirichlet boundary
        # set up solver variables, weak form, and Neumann boundary conditions
        u, p = fd.split(self.up)  # get UFL objects
        v, q = fd.TestFunctions(self.Z)
        if appctx == None:
            appctx = {"stokesextrude_nu": self.nu}
        else:
            appctx.update({"stokesextrude_nu": self.nu})
        if len(self.F_neumann) > 0:
            # non-homogeneous Neumann conditions for side facets
            for ff in self.F_neumann:  # ff = (val, ind)
                F -= fd.inner(ff[0], v) * fd.ds_v(ff[1])
        if pinch:
            # FIXME this is still not functional on a hierarchy
            pinchU = _PinchColumnVelocity(self.Z.sub(0), None, None, dim=self.dim)
            pinchU.set_pinch_data(self.hier, self.bR, self.tR, htol=self.pinchhtol)
            pinchP = _PinchColumnPressure(self.Z.sub(1), None, None)
            pinchP.set_pinch_data(self.hier, self.bR, self.tR, htol=self.pinchhtol)
            bclist = self.dirbcs + [pinchU, pinchP]
        else:
            bclist = self.dirbcs
        # problem and solver
        prob = fd.NonlinearVariationalProblem(F, self.up, bcs=bclist)
        self.solver = fd.NonlinearVariationalSolver(
            prob, options_prefix="stokes", solver_parameters=par, appctx=appctx
        )
        # actually solve
        self.solver.solve()
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        return u, p

    def save_solution(self, name=None):
        """Save u, p solution into .pvd file."""
        u, p = self.up.subfunctions[0], self.up.subfunctions[1]
        u.rename("velocity (m s-1)")
        p.rename("pressure (Pa)")
        if self.mesh.comm.size > 1:
            printpar("saving u,p,rank to %s" % name)
            rank = fd.Function(fd.FunctionSpace(self.mesh, "DG", 0))
            rank.dat.data[:] = self.mesh.comm.rank
            rank.rename("rank")
            VTKFile(name).write(u, p, rank)
        else:
            print("saving u,p to %s" % name)
            VTKFile(name).write(u, p)
