# stokes-extrude

A Python class for Stokes problems on extruded meshes in Firedrake.

## pytest

        pytest tests/

For coverage report (requires the `pytest-cov` package):

        pytest --cov-report html --cov=fascd tests/
        firefox htmlcov/index.html
