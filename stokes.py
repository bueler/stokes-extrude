'''Class for solving the Stokes problem on an extruded mesh.'''

import numpy as np
import firedrake as fd
#from problem import secpera, g, rhoi, nglen, B3

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

    def __init__(self, args):
        self.args = args
        self.Dtyp = 1.0 / secpera        # s-1
        self.sc = 1.0e-7                 # velocity scale for symmetric scaling
        # we store the basemesh info
        self.basemesh = None
        self.mx = None

    def _kinematical(self, mesh, u):
        ''' Evaluate kinematic part of residual from given velocity u, namely
        as a field defined on the whole extruded mesh:
            kres = u ds/dx - w.
        Note n_s = <-s_x, 1> so this is <u,w> . n_s.'''
        _, z = fd.SpatialCoordinate(mesh)
        kres_ufl = u[0] * z.dx(0) - u[1]
        Q1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        return fd.Function(Q1).interpolate(kres_ufl)

    def _regDu2(self, u):
        reg = self.args.visceps * self.Dtyp**2
        return 0.5 * fd.inner(D(u), D(u)) + reg

    def _stresses(self, mesh, u):
        ''' Generate effective viscosity and tensor-valued deviatoric stress
        from the velocity solution.'''
        Q1 = fd.FunctionSpace(mesh,'Q',1)
        Du2 = self._regDu2(u)
        r = 1.0 / nglen - 1.0
        assert nglen == 3.0
        nu = fd.Function(Q1).interpolate(0.5 * B3 * Du2**(r/2.0))
        nu.rename('effective viscosity (Pa s)')
        TQ1 = fd.TensorFunctionSpace(mesh, 'Q', 1)
        tau = fd.Function(TQ1).interpolate(2.0 * nu * D(u))
        tau /= 1.0e5
        tau.rename('tau (bar)')
        return nu, tau

    def createbasemesh(self, mx=None, xmax=None):
        '''Create and save the Firedrake base mesh (of intervals).'''
        self.mx = mx
        self.basemesh = fd.IntervalMesh(mx, length_or_left=0.0, right=xmax)

    def extracttop(self, mesh, field):
        '''On an extruded mesh with some ice-free (i.e. empty) columns, loop
        over the base mesh finding top cells where ice is present, then top
        nodes, and evaluate the field there.  Only works for Q1 fields.
        (Thanks Lawrence Mitchell.)'''
        assert self.basemesh is not None
        Q1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        # get the cells from basemesh and mesh
        bmP1 = fd.FunctionSpace(self.basemesh, 'Lagrange', 1)
        bmcnm = bmP1.cell_node_map().values
        cnm = Q1.cell_node_map().values
        coff = Q1.cell_node_map().offset  # node offset in column
        # get the cell-wise indexing scheme
        section, iset, facets = Q1.cell_boundary_masks
        # facets ordered with sides first, then bottom, then top
        off = section.getOffset(facets[-1])
        dof = section.getDof(facets[-1])
        topind = iset[off:off+dof]  # nodes on top of a cell
        assert len(topind) == 2
        # loop over base mesh cells computing top-node field value
        f = np.zeros(self.mx + 1)
        for cell in range(self.basemesh.cell_set.size):
            start, extent = mesh.cell_set.layers_array[cell]
            ncell = extent - start - 1
            if ncell == 0:
                continue  # leave r unchanged for these base mesh nodes
            topcellnodes = cnm[cell, ...] + coff * ncell - 1
            f_all = field.dat.data_ro[topcellnodes] # at ALL nodes in top cell
            f[bmcnm[cell,...]] = f_all[topind]
        return f

    def extrudetogeometry(self, s, b, report=False):
        '''Generate extruded mesh over self.basemesh, to height s.  The icy
        columns get their height from s, with minimum height args.Hmin.  By
        default the extruded mesh has empty (0-element) columns if ice-free
        according to s.  Optional reporting of mesh stats.'''
        assert self.basemesh is not None
        # extrude to temporary total height 1.0
        mz = self.args.mz
        layermap = np.zeros((self.mx, 2), dtype=int)  # [[0,0], ..., [0,0]]
        thk = s - b
        thkelement = ( (thk[:-1]) + (thk[1:]) ) / 2.0
        icyelement = (thkelement > self.args.Hmin + 1.0e-3)
        layermap[:,1] = mz * np.array(icyelement, dtype=int)
        # FIXME: in parallel we must provide local, haloed layermap
        mesh = fd.ExtrudedMesh(self.basemesh, layers=layermap,
                               layer_height=1.0 / mz)
        if report:
            icycount = sum(icyelement)
            print('mesh: base has %d intervals; %d are ice-free' \
                  % (self.mx, self.mx - icycount))
            print('      extruded has %d x %d icy quad elements' \
                  % (icycount, mz))
        # put s(x) into a Firedrake function on the base mesh
        P1base = fd.FunctionSpace(self.basemesh, 'Lagrange', 1)
        sbase = fd.Function(P1base)
        sbase.dat.data[:] = np.maximum(s, self.args.Hmin)
        # change mesh height to s(x)
        x, z = fd.SpatialCoordinate(mesh)
        # FIXME next line needs modification if b!=0
        xxzz = fd.as_vector([x, extend(mesh, sbase) * z])
        coords = fd.Function(mesh.coordinates.function_space())
        mesh.coordinates.assign(coords.interpolate(xxzz))
        return mesh

    def savestate(self, mesh, u, p, kres, savename=None):
        ''' Save state and diagnostics into .pvd file.'''
        nu, tau = self._stresses(mesh, u)
        u *= secpera
        u.rename('velocity (m a-1)')
        p /= 1.0e5
        p.rename('pressure (bar)')
        kres.rename('kinematic residual (a=0)')
        print('saving u,p,nu,tau,kres to %s' % savename)
        fd.File(savename).write(u,p,nu,tau,kres)

    def solve(self, mesh, printsizes=False):
        '''Solve the Glen-Stokes problem on the input extruded mesh.
        Returns the separate velocity and pressure solutions.'''

        # set up mixed method for Stokes dynamics problem
        V = fd.VectorFunctionSpace(mesh, 'Lagrange', 2)
        W = fd.FunctionSpace(mesh, 'Lagrange', 1)
        if printsizes:
            print('      dimensions n_u = %d, n_p = %d' % (V.dim(), W.dim()))
        Z = V * W
        up = fd.Function(Z)
        scu, p = fd.split(up)       # scaled velocity, unscaled pressure
        v, q = fd.TestFunctions(Z)

        # symmetrically-scaled Glen-Stokes weak form
        fbody = fd.Constant((0.0, - rhoi * g))
        sc = self.sc
        Du2 = self._regDu2(scu * sc)
        assert nglen == 3.0
        nu = 0.5 * B3 * Du2**((1.0 / nglen - 1.0)/2.0)
        F = ( sc*sc * fd.inner(2.0 * nu * D(scu), D(v)) \
              - sc * p * fd.div(v) - sc * q * fd.div(scu) \
              - sc * fd.inner(fbody, v) ) * fd.dx

        # zero Dirichlet on base (and stress-free on top and cliffs)
        bcs = [ fd.DirichletBC(Z.sub(0), fd.Constant((0.0, 0.0)), 'bottom')]

        # Newton-LU solve Stokes, split, descale, and return
        par = {'snes_linesearch_type': 'bt',
               'snes_max_it': 200,
               'snes_rtol': 1.0e-4,    # not as tight as default 1.0e-8
               'snes_stol': 0.0,       # expect CONVERGED_FNORM_RELATIVE
               'ksp_type': 'preonly',
               'pc_type': 'lu',
               'pc_factor_shift_type': 'inblocks'}
        fd.solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

        # return everything needed post-solve
        u, p = up.split()
        u *= sc
        return u, p, self._kinematical(mesh, u)

    def viewperturb(self, s, b, klist, eps=1.0, savename=None):
        '''For given s(x), compute solution perturbations from s[k] + eps,
        i.e. lifting surface by eps, at an each icy interior node in klist.
        Saves du,dp,dres to file self.savename, a .pvd file.'''
        assert self.basemesh is not None
        # solve the Glen-Stokes problem on the unperturbed extruded mesh
        meshs = self.extrudetogeometry(s, b)
        us, ps, kress = self.solve(meshs)
        # solve on the PERTURBED extruded mesh
        sP = s.copy()
        for k in klist:
            if k < 1 or k > len(s)-2:
                print('WARNING viewperturb(): skipping non-interior node k=%d' \
                      % k)
            elif s[k] > b[k] + 0.001:
                sP[k] += eps
            else:
                print('WARNING viewperturb(): skipping bare-ground node k=%d' \
                      % k)
        meshP = self.extrudetogeometry(sP, b)
        uP, pP, kresP = self.solve(meshP)
        # compute difference as a function on the unperturbed mesh
        V = fd.VectorFunctionSpace(meshs, 'Lagrange', 2)
        W = fd.FunctionSpace(meshs, 'Lagrange', 1)
        du = fd.Function(V)
        du.dat.data[:] = uP.dat.data_ro[:] - us.dat.data_ro[:]
        du *= secpera
        du.rename('du (m a-1)')
        dp = fd.Function(W)
        dp.dat.data[:] = pP.dat.data_ro[:] - ps.dat.data_ro[:]
        dp /= 1.0e5
        dp.rename('dp (bar)')
        # dres is difference of full residual cause a(x) cancels
        dres = fd.Function(W)
        dres.dat.data[:] = kresP.dat.data_ro[:] - kress.dat.data_ro[:]
        dres.rename('dres')
        print('saving perturbations du,dp,dres to %s' % savename)
        fd.File(savename).write(du,dp,dres)
