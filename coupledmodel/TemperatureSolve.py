from fenics import*
import numpy as np
from numerics import advection, diffusion, source, backward_euler
from ufl import i, j, k, l


def effstrainrate(u):
    D = sym(grad(u))
    return tr(D*D)/2 + DOLFIN_EPS


class solv:
    def __init__(self,domain,T0=-30.,A=1.,n=1.,rho=1.):
        
        self.mesh=domain.mesh
        self.ndim=domain.mesh.geometric_dimension()

        self.F =domain.F
        
        
        # Parameters for heat equation
        self.K = 66.2695
        self.c = 2.0006e18
        self.rho=rho
        self.Qg = 1.578
        self.A = A
        self.n = n
    
         #Define function for previous timestep
        self.T0 = Function(self.F)
        self.T0.assign(Constant(T0))

        

        
        #Define solver
        #self.solver = KrylovSolver('tfqmr', "hypre_amg")
        

        


        
    def iterate(self,u,mu,dt,T0):
        
        
        
        def weakform(self,Qd):
             # Test function
            g = TestFunction(self.F)
            
            # Trial function
            f = TrialFunction(self.F)

            #Diffusivity
            kappa = Constant(self.K/(self.rho*self.c))

            # Penalty term
            alpha = Constant(0.)

            # Mesh-related functions
            n = FacetNormal ( self.mesh )
            h = CellDiameter ( self.mesh )

            # Define discrete time derivative operator
            Dt =  lambda f:    backward_euler(f, self.T0, dt)
            a_A = lambda f, g: advection(f, g, u, n)
            a_D = lambda f, g: diffusion(f, g, kappa, alpha, n, h)
            a_S = lambda g: source(g, Qd/(self.rho*self.c))

            F = Dt(f)*g*dx + a_D(f, g) + a_S(g)
           
            a = lhs(F)
            L = rhs(F)

            #A,bb = assemble_system(a,L)

            return a,L

        
        T = Function(self.F)

        # Get previous timestep
        self.T0 = T0

        # Dissipation
        D = sym(grad(u))
        Sij = mu[i,j,k,l]*D[k,l]
        S = pow(self.A,-1/self.n)*pow(effstrainrate(u),(1-self.n)/(2*self.n))\
            *as_tensor(Sij, (i,j))
        
        Qd = tr(dot(S,D))
        

        # Solve
        a,L = weakform(self,Qd)
        solve(a == L, T, solver_parameters={"linear_solver": "superlu_dist"})

        

        return T

