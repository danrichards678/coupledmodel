from fenics import *
import numpy as np
import leopart as lp
from scipy.interpolate import interp1d


class surf:
    def __init__(self,domain):
        # 
        self.H = domain.H
        self.L = domain.L

        # Collect 2D function space from domain
        self.F2 = domain.F2

        # Get boundary mesh and coordinates of top
        self.bm = domain.bm
        self.topcoords = domain.topcoords
        self.topinds = domain.topinds

        self.xp = self.topcoords[:,0]

        # Create particle on surface
        self.p = lp.particles(domain.topcoords,[np.zeros_like(self.xp)], domain.mesh)

        # Create advection and add-delete classes
        u = Function(domain.V)
        self.ap = lp.advect_rk4(self.p,self.F2,u,'open')

        

    def zb(self,x):
        return -self.H + 0.5*self.H*np.sin(2*np.pi*x/self.L)




    def UpdateMesh(self,u,mesh,dt):
        # Get old postions
        oldp = np.copy(self.p.positions())
        print(oldp.shape)
        zsa = interp1d(oldp[:,0],oldp[:,1],fill_value='extrapolate')



        # Update particles
        self.ap.do_step(dt)
        newp = self.p.positions()
        #Get new particle positions
        zsb = interp1d(newp[:,0],newp[:,1],fill_value='extrapolate')
        ynew = zsb(oldp[:,0])
        self.newxy = np.array([oldp[:,0],ynew]).transpose()
        print(self.newxy.shape)
        # Update mesh coordinates
        xa = mesh.coordinates()[:,0]
        ya = mesh.coordinates()[:,1]

        xb = xa
        yb = self.zb(xb) + (zsb(xb) - self.zb(xb))*(ya - self.zb(xb))/(zsa(xb)-self.zb(xb))

        xynewcoor = np.array([xb,yb]).transpose()
        mesh.coordinates()[:] = xynewcoor
        mesh.bounding_box_tree().build(mesh)

        # Recreate particles on new mesh
        self.p = lp.particles(self.newxy,[np.zeros_like(self.newxy[:,0])],mesh)
        self.ap = lp.advect_rk4(self.p,self.F2,u,'open')





        



        
    # Function extracting values along surface ztop
    def boundaryValues(self, u, z0):
        # Define projection space
        V = FunctionSpace(u.function_space().mesh(), "Lagrange", 2)
        
        # Project velocity onto V
        ux = project(u[0], V2)
        uz = project(u[1], V2)
        
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
        

    def solveSurf(self,u,dt):



        # Extract velocity on boundary
        z0 = self.s0.vector().get_local()
        uxb, uzb = self.boundaryValues(u, self.xvec, z0)
        
        self.ux.vector().set_local(uxb)
        self.uz.vector().set_local(uzb)
        
         # Define trial function on V
        v = TrialFunction(self.V)
        
        # Define test function on V
        s = TestFunction(self.V)
    
        
       
        # Define bilinear and linear forms
        a = (1.0/dt)*s*v*dx + self.ux*s.dx(0)*v*dx \
            + Constant(self.eps)*s.dx(0)*v.dx(0)*dx
        L = (1.0/dt)*self.s0*v*dx + (self.a_s + self.uz)*v*dx
        
        
        
        # Define linear variational problem
        pde = LinearVariationalProblem(a, L, self.h, [])
        solver = LinearVariationalSolver(pde)
        
        # Solve variational problem
        solver.solve()

        # Assign solution to s0 for next timestep
        self.s0.assign(self.h)
       
        
        return self.h.vector().get_local()


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)