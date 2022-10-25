#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:52:04 2021

@author: daniel
"""

from fenics import*
from mshr import*
import numpy as np

# Flow parameters (km, yr, MPa)
ni = 3.0 # Glen's flow law exponent
A = 100.0 # Glen's flow law parameter
rho = 9.037e-13 # density
g = 9.760e+12 # gravitational acceleration
# Horizontal extent
Lx = 1 # (km)
# Number of nodes
Nx = 101 # Nodes along x
Nz = 11 # Nodes along z

# Define the EISMINT domain
def EISMINT():
    x = np.linspace(0, Lx, Nx)
    tops = 0.1*np.ones(len(x))
    bots = np.zeros(len(x))
    return x, bots, tops

xvec, zbot, ztop = EISMINT()

# Map the geometry enclosed by bottomSurf and topSurf to structured mesh
def createStructuredMesh(xvec, bottomsurf, topsurf, Nx, Nz):
    hl = Nx-1 # Number of horizontal layers
    vl = Nz-1 # Number of vertical layers
    # generate mesh on unitsquare
    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), hl, vl)
    
    # Extract x and y coordinates from mesh
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]
    x0 = min(xvec)
    x1 = max(xvec)
    
    # Map coordinates on unit square to the computational domain
    xnew = x0 + x*(x1-x0)
    zs = np.interp(xnew, xvec, topsurf)
    zb = np.interp(xnew, xvec, bottomsurf)
    ynew = zb + y*(zs-zb)
    
    xynewcoor = np.array([xnew, ynew]).transpose()
    mesh.coordinates()[:] = xynewcoor
    
    return mesh

# Class describing the bottom boundary of the domain
class bottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and near(x[1], 0.0) )
    
# Class describing the left boundary of the domain
class leftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and near(x[0], 0.0) )
    
# Class describing the right boundary of the domain 
class rightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and near(x[0], Lx) )
    
# Function extracting values along surface ztop
def boundaryValues(u, xvec, ztop):
    # Define projection space
    V = FunctionSpace(u.function_space().mesh(), "Lagrange", 2)
    
    # Project velocity onto V
    ux = project(u[0], V)
    uz = project(u[1], V)
    
    # Allow extrapolation, in case coordinates are slightly outside the domain
    ux.set_allow_extrapolation(True)
    uz.set_allow_extrapolation(True)
    
    # Preallocate array for velocity components
    uxb = np.zeros(len(xvec))
    uzb = np.zeros(len(xvec))
    
    # Extract velocity along ztop
    for i in range(len(xvec)):
        uxb[i] = ux(xvec[i], ztop[i])
        uzb[i] = uz(xvec[i], ztop[i])
        
    return uxb, uzb

# Solve Stoke's equation on doman described by mesh
def solveStokes(mesh):
    # Define Taylor-Hood element
    V = VectorElement("Lagrange", triangle, 2)
    Q = FiniteElement("Lagrange", triangle, 1)
    TH = V*Q
    W = FunctionSpace(mesh, TH)
    
    # Test function
    y = TestFunction(W)
    (v, q) = split(y)
    
    # Trial function
    w = TrialFunction(W)
    (u, p) = split(w)
    
    # Initial guess for nonlinear solver
    w0 = Function(W)
    (u0, p0) = split(w0)
    
    # Create boundary objects
    bottom = bottomBoundary()
    left = leftBoundary()
    right = rightBoundary()
    
    bcbot = DirichletBC(W.sub(0), ((0.0, 0.0)), bottom)
    bcleft = DirichletBC(W.sub(0), ((0.0, 0.0)), left)
    bcright = DirichletBC(W.sub(0), ((0.0, 0.0)), right)
    bcs = [bcbot, bcleft, bcright]
    
    # Glen's flow law
    def viscosity(u):
        strainrate = sym(grad(u))
        effstrainrate = 0.5*tr(strainrate*strainrate)
        return 0.5*A**(-1.0/ni)*(effstrainrate + 1e-10)**((1.0-ni)/(2.0*ni))
    
    # Force term
    f = Constant((0.0,-rho*g))
    
    # Define bilinear form
    a = ( 2*viscosity(u0)*inner(sym(grad(u)), grad(v))-div(v)*p-div(u)*q )*dx
    
    # Define linear form
    L = inner(f, v)*dx
    
    # Define function for holding the solution
    w = Function(W)
    (u, p) = split(w)
    (ux, uz) = split(u)
    
    # Set up fixed point iteration algorithm for solving nonlinear problem
    i = 0
    maxiter = 100
    tol = 1e-6
    for i in range(maxiter):
        # Define linearized variational problem of nonlinear PDE
        pde = LinearVariationalProblem(a, L, w, bcs)
        solver = LinearVariationalSolver(pde)
        
        # Solve variational problem
        solver.solve()
        
        # Estimate the error
        diff = w.vector().get_local()-w0.vector().get_local()
        if (np.linalg.norm(diff, ord=np.Inf)<tol):
            break
            
        # Update the initial guess
        w0.assign(w)
        
    if (i+1>= maxiter):
        print("Warning: maximum number of iterations reached")
    else:
        print("Nonlinear solver finished in", i+1, "iterations")
        
    return(w)
# Solve surface equation
def solveSurface(xvec, ztop, u, ddx, dt, eps):
    # Define 1D mesh
    interval = IntervalMesh(Nx-1, 0, Lx)
    
    # Define 1D function space
    V = FunctionSpace(interval, "Lagrange", 1)
    
    # Extract velocity on boundary
    uxb, uzb = boundaryValues(u, xvec, ztop)
    ux = Function(V)
    uz = Function(V)
    ux.vector().set_local(uxb)
    uz.vector().set_local(uzb)
    
    # Define trial function on V
    v = TrialFunction(V)
    
    # Define test function on V
    s = TestFunction(V)
    
    # Function for holding the old solution
    s0 = Function(V)
    s0.vector().set_local(np.copy(ztop))
    
    # Accumulation function (km/yr)
    a_sexpr = "fmax(0, fmin(0.0005, s*( R-fabs(x[0]-0.5*L) ) ))"
    a_s = Expression(a_sexpr, s=1e-5, R=200, L=Lx, degree=1)
    
    # Define bilinear and linear forms
    a = Constant(1.0/dt)*s*v*dx + ux*s.dx(0)*v*dx + Constant(eps)*s.dx(0)*v.dx(0)*dx
    L = Constant(1.0/dt)*s0*v*dx + (a_s + uz)*v*dx
    
    # Function for holding the new solution
    h = Function(V)
    
    # Define linear variational problem
    pde = LinearVariationalProblem(a, L, h, [])
    solver = LinearVariationalSolver(pde)
    
    # Solve variational problem
    solver.solve()
    
    return h.vector().get_local()

t = 0 # initial time (yr)
T = 50 # simulation time (yr)
dt = 25 # time stepping (yrs)
eps = 0.2 # artificial viscosity
ddx = Lx/(Nx-1) # horizontal grid resolution (km)

while(t<T):
    # Generate mesh on domain bounded by curves zbot and ztop
    mesh = createStructuredMesh(xvec, zbot, ztop, Nx, Nz)
    
    # Solve stokes
    w = solveStokes(mesh)
    w.set_allow_extrapolation(True)
    (u, p) = w.split()
    
    # Solve surface equation
    ztop = solveSurface(xvec, ztop, u, ddx, dt, eps)
   
    # Update time
    t += float(dt)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# %%
