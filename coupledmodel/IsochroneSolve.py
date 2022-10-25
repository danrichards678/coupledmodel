from fenics import*
import numpy as np
from numerics import advection, diffusion, source, backward_euler

class dgmethod:
    def __init__(self,domain,psibc):
        
        self.mesh=domain.mesh
        self.ndim=domain.mesh.geometric_dimension()

        self.F =domain.F
        
        self.psibc = psibc
    

        ## Initilise list of functions for spherical harmonic projection
        self.psi=Function(self.F)
        self.psi.assign(Expression("1.0 - x[1]",degree=1))

        #Define function for previous timestep
        self.psi0 = Function(self.F)

        #Define solver
        #self.solver = KrylovSolver('tfqmr', "hypre_amg")

        


        
    def iterate(self,u,dt,mesh):
        
        
        
        
        # Test function
        g = TestFunction(self.F)
        
        # Trial function
        f = TrialFunction(self.F)

        #Diffusivity
        kappa = Constant(0.01)

        # Penalty term
        alpha = Constant(3.)

        # Mesh-related functions
        n = FacetNormal ( mesh )
        h = CellDiameter ( mesh )

        # Define discrete time derivative operator
        Dt =  lambda f:    backward_euler(f, self.psi0, dt)
        a_A = lambda f, g: advection(f, g, u, n)
        a_D = lambda f, g: diffusion(f, g, kappa, alpha, n, h)
        a_S = lambda g: source(g, Constant(1.0))

        F = Dt(f)*g*dx + a_A(f, g) + a_D(f, g) + a_S(g)
        
        a = lhs(F)
        L = rhs(F)

        solve(a == L, self.psi, self.psibc, solver_parameters={"linear_solver": "mumps"})


        self.psi0.assign(self.psi)



        