# stokes-extrude

A Python package for Stokes problems, including glacier cases, on extruded meshes in [Firedrake](https://www.firedrakeproject.org/).

The package is named `stokesextruded`.  The implementation is two source files:

  * `stokesextruded/stokesextruded.py`: Provides `StokesExtruded` class which solves the Stokes equations over an extruded mesh.  See `tests/` for examples.
  * `stokesextruded/solverparams.py`: Provides a dictionary of dictionaries `SolverParams` with PETSc solver parameters.  See `tests/` for examples.

## installation

Install with pip: `pip install -e .`

## basic example

A minimal example, which shows the basic functionality, might look like

    from firedrake import *
    from stokesextruded import *
    basemesh = UnitIntervalMesh(10)
    mesh = ExtrudedMesh(basemesh, 4, layer_height=1.0 / 4)
    se = StokesExtruded(mesh)
    se.mixed_TaylorHood()
    se.viscosity_constant(1.0)
    se.body_force(Constant((1.0, -1.0)))
    se.dirichlet(('bottom',), Constant((0.0,0.0)))
    params = SolverParams['newton']
    params.update(SolverParams['mumps'])
    u, p = se.solve(par=params)
    se.savesolution('result.pvd')

This creates a 10 x 4 2D mesh of quadrilaterals, with P2 x P1 stable elements, over a unit square.  The Stokes problem is linear, with constant viscosity one.  The base has zero Dirichlet (u=0) conditions but otherwise the sides are stress free.  The body force pushes rightward and downward.  One might regard this as a model of a viscous block glued to a 45 degree slope.

## first run

The above code is in `examples/basic.py`.  Remember to activate the Firedrake venv before running.  View the output result with [Paraview](https://www.paraview.org/).

    $ source firedrake/bin/activate
    $ cd examples/
    $ python3 basic.py
    saving u,p to result.pvd
    $ paraview result.pvd

## pytest

    $ pytest tests/

For coverage report:

    pytest --cov-report html --cov=stokesextruded tests/
    firefox htmlcov/index.html

This requires the `pytest-cov` pip package.