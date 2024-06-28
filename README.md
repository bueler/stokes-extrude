# stokes-extrude

A Python class for Stokes problems, including glacier cases, on extruded meshes in Firedrake.

The implementation is two source files:

  * `stokesextruded/stokesextruded.py`: Provides `StokesExtruded` class which solves the Stokes equations over an extruded mesh.  See `tests/` for examples.
  * `stokesextruded/solverparams.py`: Provides a dictionary of dictionaries `SolverParams` with PETSc solver parameters.  See `tests/` for examples.

# installation

Install with pip: `pip install -e .`

## pytest

        pytest tests/

For coverage report (requires the `pytest-cov` package):

        pytest --cov-report html --cov=fascd tests/
        firefox htmlcov/index.html
