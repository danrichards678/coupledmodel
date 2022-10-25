#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:19:08 2021

@author: daniel
"""
from fenics import*
from ufl import i, j, k, l
import numpy as np
import FenicsTransforms as ft
import newtonsolver
from ufl import i,j,k,l

def effstrainrate(u):
    D = sym(grad(u))
    return tr(D*D)/2 + DOLFIN_EPS


class iterative:
    def __init__(self,domain,f=Constant((0.0,0.0)),p_in=0.0,A=1.0,n=1.0,tol=1e-4):
        self.nD = domain.nD
        self.W = domain.W
        self.bcs = domain.bcs

        self.T2 = domain.T2
        self.T4 = domain.T4

        self.A=A

        self.tol=tol
        self.mpicomm = domain.mesh.mpi_comm()

        # Force term
        #self.f = Expression(("0.0+1.0*(x[0]<1.0)","0.0"),degree=1)
        self.f = f
        self.ni = n
        

        #Pressure term
        self.n = FacetNormal(domain.mesh)
        self.ds = Measure('ds', domain=domain.mesh, subdomain_data=domain.bounds)
        self.p_in = p_in

        ## Define some test and trial functions
        self.y = TestFunction(self.W)
        (self.v,self.q) = split(self.y)

        self.w = TrialFunction(self.W)
        (self.u,self.p) = split(self.w)

        self.w0=0
        # Form for use in constructing preconditioner matrix
        self.b = Constant(pow(self.A,-1./self.ni))*inner(grad(self.u), grad(self.v))*dx + self.p*self.q*dx
        #self.b = Constant(pow(self.A,-1./self.ni))*I[i,k]*I[j,l]*(Dx(self.u[k], l)+Dx(self.u[l], k))*Dx(self.v[i], j)*dx + self.p*self.q*dx


        # Define linear form
        self.L = inner(self.f, self.v)*dx - inner(self.p_in*self.n,self.v)*self.ds(1)

        # Assemble preconditioner system
        self.P, self.btmp = assemble_system(self.b, self.L, self.bcs)

        # Create Krylov solver and AMG preconditioner
        self.solver = KrylovSolver('tfqmr', "hypre_amg")
        prm = self.solver.parameters
        prm['nonzero_initial_guess'] = True
        prm['absolute_tolerance'] = tol
        prm['relative_tolerance'] = tol
        prm['maximum_iterations'] = 20000
        if n>1.1: # i.e. if we are using this solver for initial guesses into newton
            prm['error_on_nonconvergence'] = False
        
        

    def Anisotropic(self,mu,bcs):
        #This is a wrapper

        if self.ni == 1.0:
            w = self.LinearAnisotropic(mu,bcs)
        else:
            w = self.NewtonAnisotropic(mu,bcs)

        return w

    def Isotropic(self):
        # This is a wrapper

        if self.ni == 1.0:
            w = self.LinearIsotropic()
        else:
            w = self.NonLinearIsotropic()

        return w




        
    def LinearAnisotropic(self,mu,bcs):

        # Define function for holding the solution
        w = Function(self.W)
    
        # Define variational problem
        a = (0.5*Constant(1/self.A)*mu[i,j,k,l]*(Dx(self.u[k], l)+Dx(self.u[l], k))*Dx(self.v[i], j) \
                - self.p*Dx(self.v[i], i) + self.q*Dx(self.u[i], i))*dx

        #Assemble
        A,bb = assemble_system(a, self.L, bcs)

        # Associate operator (A) and preconditioner matrix (P)
        self.solver.set_operators(A, self.P)

        #Solve
        self.solver.solve(w.vector(), bb)

      
        
        return w

    def PicardAnisotropic(self,mu,bcs):

        w = Function(self.W)
        n = self.ni

        # Initial guess for nonlinear solver
        #if self.w0==0:
        w0 = self.LinearAnisotropic(mu,bcs)
        # else:
        #     w0 = self.w0
        (u0,p0) = split(w0)
    
        # Define variational problem
        a = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u0)**((1-n)/(2*n))*mu[i,j,k,l]*(Dx(self.u[k], l)+Dx(self.u[l], k))*Dx(self.v[i], j) \
                - self.p*Dx(self.v[i], i) + self.q*Dx(self.u[i], i))*dx


        #Assemble
        A,bb = assemble_system(a, self.L, bcs)

        # Associate operator (A) and preconditioner matrix (P)
        self.solver.set_operators(A, self.P)
    
 
        #Set up fixed point iteration alogrithm
        maxiter =10
        for it in range(maxiter):

            # Define variational problem
            a = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u0)**((1-n)/(2*n))*mu[i,j,k,l]*(Dx(self.u[k], l)+Dx(self.u[l], k))*Dx(self.v[i], j) \
                    - self.p*Dx(self.v[i], i) + self.q*Dx(self.u[i], i))*dx


            #Assemble
            A,bb = assemble_system(a, self.L, bcs)

            # Associate operator (A) and preconditioner matrix (P)
            self.solver.set_operators(A, self.P)
            
            self.solver.solve(w.vector(), bb)
            
            #Estimate error
            diff = w.vector().get_local()-w0.vector().get_local()
            # if (np.linalg.norm(diff, ord=np.Inf)<self.tol):
            #     break

            #Update initial guess
            assign(w0,w)
            

        # if (it+1>= maxiter):
        #     print("Warning: maximum number of iterations reached, diff = ", diff)
        # else:
        #     print("Nonlinear solver finished in", it+1, "iterations")

        
        return w

    def NewtonAnisotropic(self,mu,bcs):

        w_tr = TrialFunction(self.W)
        (u,p) = split(w_tr)
        n = self.ni

        # Initial guess for nonlinear solver using picard iteration
        
        w = self.PicardAnisotropic(mu,bcs)
        
    
        # Define variational problem
        F = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u)**((1-n)/(2*n))*mu[i,j,k,l]*(Dx(u[k], l)+Dx(u[l], k))*Dx(self.v[i], j) \
                - p*Dx(self.v[i], i) + self.q*Dx(u[i], i))*dx - self.L

        F = action(F, w)
        J = derivative(F, w, w_tr)

        problem = Problem(J,F,bcs)
        self.custom_solver = CustomSolver(self.mpicomm)
        prm = self.custom_solver.parameters
        prm["maximum_iterations"] = 2000
        prm["absolute_tolerance"] = 1e-4
        prm["relative_tolerance"] = 1e-4
        prm["relaxation_parameter"] = 0.7
        prm['krylov_solver']['maximum_iterations'] = 20000
        self.custom_solver.solve(problem, w.vector())

        # problem = NonlinearVariationalProblem(F,w,bcs,J)
        # solver = NonlinearVariationalSolver(problem)
        # prm = solver.parameters
        # prm['newton_solver']['linear_solver'] = 'tfqmr'
        # prm['newton_solver']['preconditioner'] = 'hypre_amg'
        # prm["newton_solver"]["maximum_iterations"] = 200
        # prm["newton_solver"]["relative_tolerance"] = 1E-5
        # prm["newton_solver"]["absolute_tolerance"] = 1E-5
        # prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-8
        # prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-8
        # prm['newton_solver']['krylov_solver']['maximum_iterations'] = 10000
        # #prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
        # prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
        # prm['newton_solver']['error_on_nonconvergence'] = False
        # prm['newton_solver']['relaxation_parameter'] = 1.0


        # solver.solve()
        self.w0 =w

        return w



    def LinearIsotropic(self,mu=Constant(1.0)):
    

        # Define function for holding the solution
        w = Function(self.W)
    
        # Define variational problem
        a = (0.5*Constant(1/self.A)*mu*inner(sym(grad(self.u)), grad(self.v))\
                +div(self.v)*self.p+div(self.u)*self.q )*dx

        #Assemble
        A,bb = assemble_system(a, self.L, self.bcs)

        # Associate operator (A) and preconditioner matrix (P)
        self.solver.set_operators(A, self.P)

        #Solve
        self.solver.solve(w.vector(), bb)

      
        return w


    def NonLinearIsotropic(self):

        w = Function(self.W)
        n = self.ni

        # Initial guess for nonlinear solver
        
        self.w0 = self.LinearIsotropic()
        (u0,p0) = split(self.w0)
    
        # Define variational problem
        a = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u0)**((1-n)/(2*n))*inner(sym(grad(self.u)), grad(self.v))\
                +div(self.v)*self.p + div(self.u)*self.q )*dx


        #Assemble
        A,bb = assemble_system(a, self.L, self.bcs)

        # Associate operator (A) and preconditioner matrix (P)
        self.solver.set_operators(A, self.P)
    
        
        #Set up fixed point iteration alogrithm
        maxiter =100
        for it in range(maxiter):
            

            self.solver.solve(w.vector(), bb)
            #Estimate error
            diff = w.vector().get_local()-self.w0.vector().get_local()
            if (np.linalg.norm(diff, ord=np.Inf)<self.tol):
                break

            #Update initial guess
            self.w0.assign(w)
        


        return w

class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class CustomSolver(NewtonSolver):
    def __init__(self,mpicomm):
        NewtonSolver.__init__(self, mpicomm,
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "tfqmr")
        #PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("pc_hypre_type", "boomeramg")
        PETScOptions.set("ksp_rtol", "1.0e-7")
        PETScOptions.set("ksp_atol", "1.0e-6")

        self.linear_solver().set_from_options()



class direct:
    def __init__(self,domain,f=Constant((0.0,0.0)),p_in=0.0,A=1.0,n=1.0,tol=1e-6,pit=3):
        self.nD = domain.nD
        self.W = domain.W
        self.bcs = domain.bcs

        self.T2 = domain.T2
        self.T4 = domain.T4

        self.A=A
        self.ni = n

        # Force term
        self.f = f

        if MPI.size(MPI.comm_world)==1:
            self.linearsolver = 'superlu'
        else:
            self.linearsolver = 'mumps'
        self.preconditioner = 'hypre_amg'

        self.tol=tol
        self.picarditerations = pit
        
        #Pressure term
        self.n = FacetNormal(domain.mesh)
        self.ds = Measure('ds', domain=domain.mesh, subdomain_data=domain.bounds)
        self.p_in = p_in

        ## Define some test and trial functions
        self.y = TestFunction(self.W)
        (self.v,self.q) = split(self.y)

        self.w = TrialFunction(self.W)
        (self.u,self.p) = split(self.w)


        # Define linear form
        self.L = inner(self.f, self.v)*dx - inner(self.p_in*self.n,self.v)*self.ds(1)

    def Anisotropic(self,mu,bcs):
        #This is a wrapper

        if self.ni == 1.0:
            w = self.LinearAnisotropic(mu,bcs)
        else:
            w = self.NewtonAnisotropic(mu,bcs)

        return w

    def Isotropic(self):
        # This is a wrapper

        if self.ni == 1.0:
            w = self.LinearIsotropic()
        else:
            w = self.NonLinearIsotropic()

        return w




        
    def LinearAnisotropic(self,mu,bcs):


        # Define function for holding the solution
        w = Function(self.W)
    
        # Define variational problem
        a = (0.5*Constant(1/self.A)*mu[i,j,k,l]*(Dx(self.u[k], l)+Dx(self.u[l], k))*Dx(self.v[i], j) \
                + self.p*Dx(self.v[i], i) + self.q*Dx(self.u[i], i))*dx

        pde = LinearVariationalProblem(a,self.L,w,bcs)
        solver = LinearVariationalSolver(pde)
        solver.parameters['linear_solver'] = self.linearsolver
        solver.parameters['preconditioner'] = self.preconditioner
        solver.solve()
      
        
        return w

    def PicardAnisotropic(self,mu,bcs):

        w = Function(self.W)
        n = self.ni

        # Initial guess for nonlinear solver
        
        self.w0 = self.LinearAnisotropic(mu,bcs)
        (u0,p0) = split(self.w0)
    
        # Define variational problem
        a = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u0)**((1-n)/(2*n))*mu[i,j,k,l]*(Dx(self.u[k], l)+Dx(self.u[l], k))*Dx(self.v[i], j) \
                + self.p*Dx(self.v[i], i) + self.q*Dx(self.u[i], i))*dx
    
        pde = LinearVariationalProblem(a,self.L,w,bcs)
        solver = LinearVariationalSolver(pde)
        solver.parameters['linear_solver'] = self.linearsolver
        solver.parameters['preconditioner'] = self.preconditioner
        
        

        #Set up fixed point iteration alogrithm
        
        for it in range(self.picarditerations):
            
            solver.solve()
            
            #Estimate error
            diff = w.vector().get_local()-self.w0.vector().get_local()
            # if (np.linalg.norm(diff, ord=np.Inf)<tol):
            #     break

            #Update initial guess
            self.w0.assign(w)

            pde = LinearVariationalProblem(a,self.L,w,bcs)
            solver = LinearVariationalSolver(pde)
            solver.parameters['linear_solver'] = self.linearsolver
            solver.parameters['preconditioner'] = self.preconditioner

        # if (it+1>= maxiter):
        #     pass
        #     #print("Warning: maximum number of iterations reached, diff = ", diff)
        # else:
        #     print("Nonlinear solver finished in", it+1, "iterations")

        
        return w

    def NewtonAnisotropic(self,mu,bcs):

        w_tr = TrialFunction(self.W)
        (u,p) = split(w_tr)
        n = self.ni

        # Initial guess for nonlinear solver using picard iteration
        
        w = self.PicardAnisotropic(mu,bcs)
        
    
        # Define variational problem
        F = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u)**((1-n)/(2*n))*mu[i,j,k,l]*(Dx(u[k], l)+Dx(u[l], k))*Dx(self.v[i], j) \
                - p*Dx(self.v[i], i) + self.q*Dx(u[i], i))*dx - self.L

        F = action(F, w)
        J = derivative(F, w, w_tr)

        # problem = Problem(J,F,self.bcs)
        # self.custom_solver = PETScSNESSolver('newtonls')
        # prm = self.custom_solver.parameters


        # # self.custom_solver = CustomDirectSolver(self.mpicomm)
        # # prm = self.custom_solver.parameters
        # prm['linear_solver'] = self.linearsolver
        # prm["maximum_iterations"] = 2000
        # prm["absolute_tolerance"] = self.tol
        # prm["relative_tolerance"] = self.tol
        # #prm["relaxation_parameter"] = 0.7
        # # #prm['krylov_solver']['maximum_iterations'] = 20000
        # self.custom_solver.solve(problem, w.vector())

        problem = NonlinearVariationalProblem(F,w,bcs,J)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        # prm['nonlinear_solver'] = 'snes'
        # prm['snes_solver']['linear_solver'] = self.linearsolver
        # prm["snes_solver"]["maximum_iterations"] = 2000
        # prm["snes_solver"]["relative_tolerance"] = self.tol
        # prm["snes_solver"]["absolute_tolerance"] = self.tol
        # prm['snes_solver']['relaxation_parameter'] = 0.7

        prm['newton_solver']['linear_solver'] = self.linearsolver
        #prm['newton_solver']['preconditioner'] = self.preconditioner
        prm["newton_solver"]["maximum_iterations"] = 2000
        prm["newton_solver"]["relative_tolerance"] = self.tol
        prm["newton_solver"]["absolute_tolerance"] = self.tol
        prm['newton_solver']['relaxation_parameter'] = 0.6
        prm['newton_solver']['error_on_nonconvergence'] = False
        # # prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-8
        # # prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-8
        # # prm['newton_solver']['krylov_solver']['maximum_iterations'] = 10000
        # # #prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
        # # prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
        # # prm['newton_solver']['error_on_nonconvergence'] = False
        
        solver.solve()

        self.w0 =w

        return w


    def LinearIsotropic(self,mu=Constant(1.0)):
    

        # Define function for holding the solution
        w = Function(self.W)
    
        # Define variational problem
        a = (0.5*Constant(1/self.A)*mu*inner(sym(grad(self.u)), grad(self.v))\
                +div(self.v)*self.p+div(self.u)*self.q )*dx

        pde = LinearVariationalProblem(a,self.L,w,self.bcs)
        solver = LinearVariationalSolver(pde)
        solver.parameters['linear_solver'] = self.linearsolver
        solver.parameters['preconditioner'] = self.preconditioner
        solver.solve()
        return w


    def NonLinearIsotropic(self):

        w = Function(self.W)
        n = self.ni

        # Initial guess for nonlinear solver
        
        self.w0 = self.LinearIsotropic()
        (u0,p0) = split(self.w0)
    
        # Define variational problem
        a = (0.5*Constant(pow(self.A,-1./n))*effstrainrate(u0)**((1-n)/(2*n))*inner(sym(grad(self.u)), grad(self.v))\
                +div(self.v)*self.p + div(self.u)*self.q )*dx

        pde = LinearVariationalProblem(a,self.L,w,self.bcs)
        solver = LinearVariationalSolver(pde)
        solver.parameters['linear_solver'] = self.linearsolver
        solver.parameters['preconditioner'] = self.preconditioner

        #Set up fixed point iteration alogrithm
        maxiter =100
        tol = 1e-8
        for it in range(maxiter):
            

            solver.solve()
            #Estimate error
            diff = w.vector().get_local()-self.w0.vector().get_local()
            if (np.linalg.norm(diff, ord=np.Inf)<tol):
                break

            #Update initial guess
            self.w0.assign(w)
        


        return w

    
#     def NewtonIsotropic(self,w,n=3.0):
#         u,p = split(w)

#         # Define variational problem
#         F = (0.5*Constant(pow(self.A,-1/n))*effstrainrate(u)**((1-n)/(2*n))*inner(sym(grad(u)), grad(self.v))\
#                 -div(self.v)*p-div(u)*self.q )*dx - self.L

#         J = derivative(F,w)

#         problem = Problem(J,F,self.bcs)
#         self.cs.solve(problem, w.vector())
        
#         return w



        
# class Problem(NonlinearProblem):
#     def __init__(self, J, F, bcs):
#         self.bilinear_form = J
#         self.linear_form = F
#         self.bcs = bcs
#         NonlinearProblem.__init__(self)

#     def F(self, b, x):
#         assemble(self.linear_form, tensor=b)
#         for bc in self.bcs:
#             bc.apply(b, x)

#     def J(self, A, x):
#         assemble(self.bilinear_form, tensor=A)
#         for bc in self.bcs:
#             bc.apply(A)


# class CustomSolver(NewtonSolver):
#     def __init__(self,mpicomm):
#         NewtonSolver.__init__(self, mpicomm,
#                               PETScKrylovSolver(), PETScFactory.instance())

#     def solver_setup(self, A, P, problem, iteration):
#         self.linear_solver().set_operator(A)

#         PETScOptions.set("ksp_type", "gmres")
#         PETScOptions.set("ksp_monitor")
#         PETScOptions.set("pc_type", "ilu")

#         self.linear_solver().set_from_options()






