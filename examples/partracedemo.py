# Demonstrate correctness of trace_bottom() in parallel:
#   $ mpiexec -n P python3 tracedemo.py
# (Note that it is not clear how to get this parallel
# functionality from pytest.)  Example study:
#   $ for P in 1 2 3 5 6 10 11 20; do mpiexec -n $P python3 partracedemo.py; done

from firedrake import *
from stokesextrude import *

basemesh = UnitSquareMesh(5, 5)
mesh = ExtrudedMesh(basemesh, 3)
x, y = SpatialCoordinate(basemesh)
P1base = FunctionSpace(basemesh, 'CG', 1)
fbase = Function(P1base).interpolate((x - y) * x + y * y)
fext = extend_p1_from_basemesh(mesh, fbase)
fbase2 = trace_scalar_to_p1(basemesh, mesh, fext, surface='bottom')
assert errornorm(fbase, fbase2) < 1.0e-14
