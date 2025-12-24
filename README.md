# stokes-extrude

This repository provides a Python package named `stokesextrude` for Stokes-like fluid problems, including glacier-relevant tools, on extruded meshes.  The core technology is all from the [Firedrake](https://www.firedrakeproject.org/) finite element library.

The implementation is in 3 source files in directory `stokesextrude/`:

  * `stokesextrude.py`: Provides `StokesExtrude` class which solves the Stokes equations over an extruded mesh.
  * `solverparams.py`: Defines a dictionary `SolverParams` which contains dictionaries of PETSc solver parameters.
  * `traceextend.py`: Tools for extending fields from the base mesh to the extruded mesh, and for computing top or bottom traces of fields on the extruded mesh.

See `tests/` and `examples/` for examples.

## installation

Install with pip: `pip install -e .`

## basic example

A minimal example, which shows some basic functionality, might look like

```python
from firedrake import *
from stokesextrude import *
basemesh = UnitIntervalMesh(20)
se = StokesExtrude(basemesh, mz=10)
se.mixed_TaylorHood()
se.viscosity_constant(1.0)
u, p = split(se.up)
v, q = TestFunctions(se.Z)
f_body = Constant((1.0, -1.0))
F = ( inner(2.0 * se.nu * se.D(u), se.D(v)) - p * div(v) - q * div(u) \
      - inner(f_body, v) ) * dx
se.dirichlet(('bottom',), Constant((0.0, 0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
params['snes_converged_reason'] = None
u, p = se.solve(F=F, par=params)
se.save_solution('result.pvd')
```

It creates a 20 x 10 2D mesh of quadrilaterals, with P2 x P1 stable elements, over a unit square.  The Stokes problem is linear, with constant viscosity one.  The base has zero Dirichlet (u=0) conditions but otherwise the sides are stress free.  The body force pushes rightward and downward.  One might regard this as a model of a linearly-viscous block glued to a 45 degree slope.  The solver is direct.

## first run

Save the above code to `basic.py`.  Remember to activate the Firedrake venv before running.  View the output result with [Paraview](https://www.paraview.org/).

```bash
$ source firedrake/bin/activate
$ python3 basic.py
  Nonlinear stokesextrude_ solve converged due to CONVERGED_FNORM_ABS iterations 1
saving u,p to result.pvd
$ paraview result.pvd
```

## capabilities

In more detail, we use [Firedrake](https://www.firedrakeproject.org) to solve a Stokes problem on an extruded mesh, in 2D or 3D.  The user provides the base mesh, which will be 1D or 2D, respectively.  Elements are products of the base mesh element and an interval.

The Firedrake documentation on [extruded meshes](https://www.firedrakeproject.org/extruded-meshes.html) is a good place to start if you want to understand how these meshes work.

Here are some capabilities of the `StokesExtrude` class:
  1. The most important functionality is geometric.  One can set the upper and lower surface elevation/location from given fields (scalar `Function`) on the base mesh, or from scalar constants.
  1. Tools are provided to go back and forth between fields defined over the base mesh versus trace values at the top and bottom of the extruded mesh.  These can be in the `R` space of the extruded mesh.
  1. Zero-height columns are allowed.  The classes `PinchColumnPressure` and `PinchColumnVelocity` are defined for this purpose.  They adds conditions similar to Dirichlet conditions for all degrees of freedom, e.g. velocities and pressures, which are in zero-height columns.
  1. At initialization, the class can create a mesh hierarchy (from a base mesh hierarchy) for geometric multigrid.
  1. One can set a variety of Dirichlet and Neumann boundary conditions.  The user is responsible for choosing a well-posed problem; e.g. at least some Dirichlet conditions should be set.
  1. One can set classical Taylor-Hood (P2 x P1), higher-order Taylor-Hood, or P2 x DG0.
  1. Tests and examples are provide with linear and power-law viscosity; the latter is for glaciers.

Note that the user provides the weak form itself.

## known limitations

The combination of zero-height columns and mesh hierarchy is under development.

## pytest

```bash
$ pytest .
```

For coverage report:

```bash
$ pytest --cov-report=html --cov=stokesextrude tests/
$ firefox htmlcov/index.html
```

This requires the `pytest-cov` pip package.
