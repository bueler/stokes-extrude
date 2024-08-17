from firedrake import *
#from stokesextrude import *
#basemesh = UnitIntervalMesh(10)

mesh = Mesh("Rdomain.msh")

H = FunctionSpace(mesh,'CG',1)
u = Function(H, name='u(x,y)')
v = TestFunction(H)

fsrc = Function(H).interpolate(Constant(1.0))
fsrc.rename('f(x,y)')

F = ( dot(grad(u), grad(v)) - fsrc * v ) * dx

BCs = DirichletBC(H, Constant(0.0), 3)

solve(F == 0, u, bcs=[BCs],
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

File("result.pvd").write(fsrc, u)
