# stokes-extrude

This repository provides a Python package named `stokesextrude` for Stokes problems, including glacier cases, on extruded meshes.  The core technology is all from the [Firedrake](https://www.firedrakeproject.org/) finite element library.

The implementation is in 3 source files in directory `stokesextrude/`:

  * `stokesextrude.py`: Provides `StokesExtrude` class which solves the Stokes equations over an extruded mesh.
  * `solverparams.py`: Defines a dictionary `SolverParams` which contains dictionaries of PETSc solver parameters.
  * `traceextend.py`: Tools for extending fields from the base mesh to the extruded mesh, and for computing top or bottom traces of fields on the extruded mesh.

See `tests/` and `examples/` for examples.

## installation

Install with pip: `pip install -e .`

## basic example

A minimal example, which shows the basic functionality, might look like

```python
from firedrake import *
from stokesextrude import *
basemesh = UnitIntervalMesh(10)
se = StokesExtrude(basemesh, mz=4)
se.mixed_TaylorHood()
se.viscosity_constant(1.0)
se.body_force(Constant((1.0, -1.0)))
se.dirichlet(('bottom',), Constant((0.0,0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
u, p = se.solve(par=params)
se.savesolution('result.pvd')
```

It creates a 10 x 4 2D mesh of quadrilaterals, with P2 x P1 stable elements, over a unit square.  The Stokes problem is linear, with constant viscosity one.  The base has zero Dirichlet (u=0) conditions but otherwise the sides are stress free.  The body force pushes rightward and downward.  One might regard this as a model of a viscous block glued to a 45 degree slope.

## first run

Save the above code to `basic.py`.  Remember to activate the Firedrake venv before running.  View the output result with [Paraview](https://www.paraview.org/).

```bash
$ source firedrake/bin/activate
$ python3 basic.py
saving u,p to result.pvd
$ paraview result.pvd
```

## capabilities

In more detail, we use [Firedrake](https://www.firedrakeproject.org) to solve a Stokes problem on an extruded mesh.  The [Firedrake documentation on extruded meshes[(https://www.firedrakeproject.org/extruded-meshes.html) is a good place to start.

Here are some capabilities:
  1. A standard linear weak form, with a user-configurable viscosity constant, is available.  Alternatively, the user can provide the weak form.
  2. One can set a variety of Dirichlet and Neumann boundary conditions.  The user is responsible for choosing a well-posed problem; e.g. at least some Dirichlet conditions should be set.
  3. Geometry functionality includes the ability to set the upper and lower elevation from functions on the base mesh, or from scalar constants.
  4. Zero-height columns are allowed; to do this call `StokesExtrude.trivializepinchcolumns()` after setting elevations and the mixed space.
  5. One can use classical Taylor-Hood (P2 x P1), higher-order Taylor-Hood, or P2 x DG0.  (However, only the first-option is well-tested.)
  6. Solvers can exploit a vertical mesh hierarchy for geometric multigrid.  Algebraic multigrid can be used over the coarse mesh.
  7. Tests and examples are provide with linear viscosity, and with power-law viscosity suitable for glaciers.


## pytest

```bash
$ pytest tests/
```

For coverage report:

```bash
$ pytest --cov-report=html --cov=stokesextrude tests/
$ firefox htmlcov/index.html
```

This requires the `pytest-cov` pip package.
