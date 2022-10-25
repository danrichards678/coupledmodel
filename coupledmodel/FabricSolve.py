#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:22:28 2021

@author: daniel
"""

from fenics import*
from ufl import i, j, k, l
import numpy as np
import speccaf.solver as solver
import speccaf.spherical as spherical
import FenicsTransforms as ft
from numerics import advection, diffusion, source, backward_euler
import leopart as lp



# fbc = np.array([ 1.        ,  0.25094783,  0.        ,  0.30734707, -0.05210749,\
#         0.        , -0.05492612,  0.        , -0.07266042, -0.07893723,\
#         0.        , -0.08088659,  0.        , -0.08860682,  0.        ,\
#        -0.1199742 ])

# fbc = np.array([ 1.00000000e+00+0.j,  1.20610934e+00+0.j,  1.27259449e-01+0.j,
#        -3.11609094e-02+0.j,  5.98273149e-01+0.j,  1.27088235e-01+0.j,
#        -1.20776437e-01+0.j,  9.44386665e-04+0.j,  5.51422997e-03+0.j,
#         1.90922662e-01+0.j,  7.72129262e-02+0.j, -7.44876212e-02+0.j,
#        -2.02664639e-02+0.j,  9.43551550e-03+0.j,  1.20575127e-03+0.j,
#        -8.33452724e-05+0.j])

fbc=np.array([ 1.        +0.j, -0.63232535+0.j,  0.        +0.j, -0.84203795+0.j,
        0.20264813+0.j,  0.        +0.j,  0.22189721+0.j,  0.        +0.j,
        0.31829083+0.j, -0.04063185+0.j,  0.        +0.j, -0.04103348+0.j,
        0.        +0.j, -0.04163456+0.j,  0.        +0.j, -0.01989168+0.j])


class dgmethod:
    def __init__(self,domain,constT=-20.):
        
        self.mesh=domain.mesh
        self.ndim=domain.mesh.geometric_dimension()

        self.F =domain.F
        self.F2 = domain.F2
        

        # Initial Fabric
        self.sh = spherical.spherical(4)

        ## Initilise list of functions for spherical harmonic projection
        self.f_df=[]
        for ilm in range(self.sh.nlm):
            self.f_df.append(Function(self.F2))
        
        #self.f_df[0].assign(Expression(("1.","0"),degree=1))

        for ilm in range(self.sh.nlm):
            self.f_df[ilm].assign(Expression((str(fbc[ilm].real),str(fbc[ilm].imag)),degree=1))
        #Define function for previous timestep
        self.f0 = Function(self.F)

        #Define function for fabric evolution
        self.Fstar0 = Function(self.F)

        # Number of points in function
        self.npoints = self.f0.vector().get_local().size

        # Temperature
        self.Temp = constT*np.ones(self.npoints)

        # Numpy fabric
        self.fnp = self.sh.spec_array()
        #self.fnp[0]=1.
        self.fnp = fbc
        self.fnp=np.tile(self.fnp,(self.npoints,1))
        
        ##Initial orientation tensors
        self.a2 = a2calc(self.f_df)
        self.a4 = a4calc(self.f_df)

        self.gradunp = np.zeros((self.npoints,3,3))

        #Define solver
        #self.solver = KrylovSolver('tfqmr', "hypre_amg")

        


        
    def iterate(self,u,dt):
        
        
        
        def weakform(self,Fstar0,f0):
             # Test function
            g = TestFunction(self.F)
            
            # Trial function
            f = TrialFunction(self.F)

            #Diffusivity
            kappa = Constant(0.01)

            # Penalty term
            alpha = Constant(3.)

            # Mesh-related functions
            n = FacetNormal ( self.mesh )
            h = CellDiameter ( self.mesh )

            # Define discrete time derivative operator
            Dt =  lambda f:    backward_euler(f, f0, dt)
            a_A = lambda f, g: advection(f, g, u, n)
            a_D = lambda f, g: diffusion(f, g, kappa, alpha, n, h)
            a_S = lambda g: source(g, Fstar0)

            F = Dt(f)*g*dx + a_A(f, g) + a_D(f, g) + a_S(g)
           
            a = lhs(F)
            L = rhs(F)

            #A,bb = assemble_system(a,L)

            return a,L

        
        f = Function(self.F)

        # Get velocity gradient
        (ux,uy) = split(u)
        self.gradunp[:,0,0] = ft.Fenics2Numpy(project(ux.dx(0),self.F))
        self.gradunp[:,0,1] = ft.Fenics2Numpy(project(ux.dx(1),self.F))
        self.gradunp[:,1,0] = ft.Fenics2Numpy(project(uy.dx(0),self.F))
        self.gradunp[:,1,1] = ft.Fenics2Numpy(project(uy.dx(1),self.F))

        # Orientational evolution
        sc = solver.couplediterate(self.Temp,self.gradunp,self.sh)
        Fstarnp = sc.interiorstep(self.fnp,dt)
       
        for ilm in range(self.sh.nlm):

            # Real part
            self.Fstar0 = ft.Numpy2Fenics(self.Fstar0,Fstarnp[:,ilm].real)

            a,L = weakform(self,self.Fstar0,self.f_df[ilm].sub(0))
            solve(a == L, f, solver_parameters={"linear_solver": "gmres"})


            ftempr = ft.Fenics2Numpy(f) 
            assign(self.f_df[ilm].sub(0),f)
            

            # Imaginary part
            self.Fstar0 = ft.Numpy2Fenics(self.Fstar0,Fstarnp[:,ilm].imag)

            a,L = weakform(self,self.Fstar0,self.f_df[ilm].sub(1))
            solve(a == L, f, solver_parameters={"linear_solver": "gmres"})
            
            ftempi = ft.Fenics2Numpy(f)
            assign(self.f_df[ilm].sub(1),f)

            # Update fnp
            self.fnp[:,ilm] = ftempr + 1j*ftempi
        
        # Correction
        self.fnp[:,0] = 1.0
        
        # Update orientation tensors
        self.a2 = a2calc(self.f_df)
        self.a4 = a4calc(self.f_df)


        
class leosolver:
    def __init__(self,domain,T,bc=False,pbc=False):

        self.ndim=domain.mesh.geometric_dimension()

        self.bc = bc
        self.pbc = pbc
        
        # Number of particles
        self.npart = 5
        self.npmin = 4
        self.npmax = 12

        #Function Spaces
        self.mesh = domain.mesh
        self.F2 = domain.F2
        self.F = domain.F
        self.V = domain.V

        #Initilise particle locations
        self.x = lp.RandomCell(self.mesh).generate(self.npart)

        #Initilise spherical class
        self.sh = spherical.spherical(6)

        ##Initial fabric condtion - isotropic
        self.fnp = np.zeros((self.x.shape[0],self.sh.nlm),dtype='complex128')
        self.fnp[:,0] = 1.

        ## Initilise list of functions for spherical harmonic projection
        self.f_df=[]
        for ilm in range(self.sh.nlm):
            self.f_df.append(Function(self.F2))

        self.f_df[0].assign(Expression(("1.","0"),degree=1))

        ## Temperature field
        if isinstance(T,float):
            self.T_df = Function(self.F)
            self.T_df.assign(Constant(T))
        else:
            self.T_df = T
        
        self.T=np.zeros(self.x.shape[0])


        ##Initilise particle
        self.gradu=np.zeros((self.x.shape[0],3,3))
        self.psi = 1.0 - self.x[:,1]
        self.p = self.odf2particle()

        ##Initial orientation tensors
        self.a2 = a2calc(self.f_df)
        self.a4 = a4calc(self.f_df)

        self.f0idx =  6

        #Isochrone function
        self.psi_df = Function(self.F)
        self.psi_df.assign(Expression("1.0-x[1]",degree=1))
        

        # For editing
        dudx_df = Function(self.F)
        dudy_df = Function(self.F)
        dvdx_df = Function(self.F)
        self.u = Function(domain.V)
        if self.pbc:
            self.lims = domain.lims
            self.ap = lp.advect_rk3(self.p, self.V, self.u, "periodic", self.lims.flatten())
        else:
            self.ap = lp.advect_rk3(self.p, self.V, self.u, "open")
        
        self.AD = lp.AddDelete(self.p,self.npmin,self.npmax,[dudx_df,dudy_df,dvdx_df,self.T_df,self.psi_df]+self.f_df)
        self.all_cells = [c.index() for c in cells(self.mesh)]



    def particle2odf(self):
        numlocalparticles  = len(self.p.get_property(1)) # for mpi
        fnp = np.zeros((numlocalparticles,self.sh.nlm),dtype='complex128')
        for ilm in range(self.sh.nlm):
            fnp[:,ilm] = np.array(self.p.get_property(ilm+self.f0idx)[::2]) + 1j*np.array(self.p.get_property(ilm+self.f0idx)[1::2])
        return fnp

    def odf2particle(self):
        f=[]
        for ilm in range(self.sh.nlm):
            f.append(np.stack((self.fnp[:,ilm].real,self.fnp[:,ilm].imag),axis=1))
        
        return lp.particles(self.x,[self.gradu[:,0,0],self.gradu[:,0,1],self.gradu[:,1,0],self.T,self.psi] +f, self.mesh)

    def edit_properties(self,particles, candidate_cells,fnp):

        num_properties = particles.num_properties()
        # Transform fnp into 2 column vector of real and imaginary parts
        ind =0
        for c in candidate_cells:
            for pi in range(particles.num_cell_particles(c)):
                particle_props = list(particles.property(c, pi, prop_num)
                                    for prop_num in range(num_properties))


                x = particle_props[0].x()
                # Edit properties

                particles.set_property(c, pi, 5, Point(self.psi[ind]))

                for ilm in range(self.sh.nlm):
                    particles.set_property(c, pi, ilm+self.f0idx, Point(np.array(\
                                    [fnp[ind,ilm].real, fnp[ind,ilm].imag])))

                    # Hacky boundary condition
                    if self.bc:
                    
                        l = 0.05
                        if x<l:
                            particles.set_property(c,pi,ilm+self.f0idx, Point(np.array(\
                                        [fbc[ilm].real, fbc[ilm].imag])))


                            
        
                ind = ind +1
        
        
    def iterate(self,w,dt,xvec=None,ztop=None,T=False):
        
        # Get new velocity gradients
        (ux,uy)=split(w.sub(0))
        dudx_df = project(ux.dx(0),self.F)
        dudy_df = project(ux.dx(1),self.F)
        dvdx_df = project(uy.dx(0),self.F)

        



        #Advect particles
        #ap = lp.advect_particles(self.p, self.F2, u, "open")
        assign(self.u,w.sub(0))
        self.ap.do_step(dt)

        ##Add delete
        #AD = lp.AddDelete(self.p,self.npmin,self.npmax,[dudx_df,dudy_df,dvdx_df,self.T_df]+self.f_df)
        if T:
            assign(self.T_df,T)
        
        self.AD.do_sweep()

        #Get new positions
        self.x = self.p.positions()

        #Interpolate velocity gradient onto new particle positions
        self.p.interpolate(dudx_df,1)
        self.p.interpolate(dudy_df,2)
        self.p.interpolate(dvdx_df,3)

        #Interpolate new temperature onto new particle positions
        self.p.interpolate(self.T_df,4)


        ## Build numpy matrices
        self.gradu=np.zeros((self.x.shape[0],3,3))
        self.gradu[:,0,0] = self.p.get_property(1)
        self.gradu[:,0,1] = self.p.get_property(2)
        self.gradu[:,1,0] = self.p.get_property(3)
        self.gradu[:,1,1] = -self.gradu[:,0,0]
        self.T = np.array(self.p.get_property(4))

        ## Fnp at new position
        self.fnp = self.particle2odf()

        # Isochrone evolution
        self.psi = np.array(self.p.get_property(5))
        self.psi = self.psi + dt


        


        # Set inlet to isotropic
        # tol=1e-2
        # self.fnp[self.x[:,0]<tol,:]=0.0
        # self.fnp[self.x[:,0]<tol,0]=1.0

        # Evolve fabric
        sc = solver.couplediterate(self.T,self.gradu,self.sh)
        self.fnp = sc.interiorstep(self.fnp,dt)*dt + self.fnp

        ##Apply corrector
        self.fnp[:,0] = 1.0
        # self.fnp = FabricCorrector.simplecorrector(self.fnp)

        #Boundary condition: isochrone =0 and fabric is isotropic
        if isinstance(xvec,np.ndarray):
            tol = 1.0*dt
            h = np.interp(self.x[:,0],xvec,ztop)
            dy = np.abs(h - self.x[:,1])

            self.dy=dy
            self.tol=tol

            self.psi[dy<tol] = 0
            self.fnp[dy<tol,1:] = 0.0
        #Update particle
        #self.p = self.odf2particle()
        self.edit_properties(self.p,self.all_cells,self.fnp)
        
        #Update fenics projection
        for ilm in range(self.sh.nlm):
            lstsq = lp.l2projection(self.p,self.F2,ilm+self.f0idx)
            lstsq.project(self.f_df[ilm])

        #Update temperature
        lstsq = lp.l2projection(self.p,self.F,4)
        lstsq.project(self.T_df)

        #Update isochrone
        lstsq = lp.l2projection(self.p,self.F,5)
        lstsq.project(self.psi_df)

        #Update orientation tensors
        self.a2 = a2calc(self.f_df)
        self.a4 = a4calc(self.f_df)



def idx(l,m):
    return ((l*l)//4 + m)

def a2calc(f):

    a00 = 0.333333333333333333*f[idx(0,0)].sub(0) -0.14907119849998597976*f[idx(2,0)].sub(0) +2*0.18257418583505537115*f[idx(2,2)].sub(0)
    a01 = -2*0.18257418583505537115*f[idx(2,2)].sub(1)
    a10 = a01
    a02=2*((-0.18257418583505537115))*f[idx(2,1)].sub(0)
    a20=a02
    a11=((0.33333333333333333333))*f[idx(0,0)].sub(0)+ ((-0.14907119849998597976))*f[idx(2,0)].sub(0)+ 2*((-0.18257418583505537115))*f[idx(2,2)].sub(0)
    a12=-2*((-0.18257418583505537115))*f[idx(2,1)].sub(1)
    a21=a12
    a22=((0.33333333333333333333))*f[idx(0,0)].sub(0)+ ((0.29814239699997195952))*f[idx(2,0)].sub(0)

    a = as_matrix([[a00, a01],[a10, a11]])

    a3d = as_matrix([[a00, a01, a02],[a10, a11, a12],[a20, a21, a22]])


    return a3d


def a4calc(f):

    a4list = []
    for i in range(3):
        a4list += [[],]
        for j in range(3):
            a4list[i] += [[],]
            for k in range(3):
                a4list[i][j] += [[],]
                for l in range(3):
                    a4list[i][j][k] += [[],]

    a4list[0][0][0][0] =((0.2))*f[idx(0,0)].sub(0)\
                        + ((-0.12777531299998798265))*f[idx(2,0)].sub(0)\
                        + 2*((0.15649215928719031813))*f[idx(2,2)].sub(0)\
                        + ((0.028571428571428571429))*f[idx(4,0)].sub(0)\
                        + 2*((-0.030116930096841707924))*f[idx(4,2)].sub(0)\
                        + 2*((0.039840953644479787999))*f[idx(4,4)].sub(0)
        
    a4list[1][0][0][0]=-2*((0.078246079643595159065))*f[idx(2,2)].sub(1)\
                        - 2*((-0.015058465048420853962))*f[idx(4,2)].sub(1)\
                        - 2*((0.039840953644479787999))*f[idx(4,4)].sub(1)
    a4list[0][1][0][0]=a4list[1][0][0][0]
    a4list[0][0][1][0]=a4list[1][0][0][0]
    a4list[0][0][0][1]=a4list[1][0][0][0]

    a4list[1][1][0][0]=((0.066666666666666666667))*f[idx(0,0)].sub(0)\
                        + ((-0.042591770999995994217))*f[idx(2,0)].sub(0)\
                        + ((0.0095238095238095238095))*f[idx(4,0)].sub(0)\
                        + 2*((-0.039840953644479787999))*f[idx(4,4)].sub(0)
    a4list[1][0][1][0]=a4list[1][1][0][0]
    a4list[1][0][0][1]=a4list[1][1][0][0]
    a4list[0][1][1][0]=a4list[1][1][0][0]
    a4list[0][1][0][1]=a4list[1][1][0][0]
    a4list[0][0][1][1]=a4list[1][1][0][0]

    a4list[1][1][1][0]=-2*((0.078246079643595159065))*f[idx(2,2)].sub(1)\
                        - 2*((-0.015058465048420853962))*f[idx(4,2)].sub(1)\
                        - 2*((-0.039840953644479787999))*f[idx(4,4)].sub(1)
    a4list[1][1][0][1]=a4list[1][1][1][0]
    a4list[1][0][1][1]=a4list[1][1][1][0]
    a4list[0][1][1][1]=a4list[1][1][1][0]

    a4list[1][1][1][1]=((0.2))*f[idx(0,0)].sub(0)\
                            + ((-0.12777531299998798265))*f[idx(2,0)].sub(0)\
                            + 2*((-0.15649215928719031813))*f[idx(2,2)].sub(0)\
                            + ((0.028571428571428571429))*f[idx(4,0)].sub(0)\
                            + 2*((0.030116930096841707924))*f[idx(4,2)].sub(0)\
                            + 2*((0.039840953644479787999))*f[idx(4,4)].sub(0)


    
    a4list[2][0][0][0]=2*((-0.078246079643595159065))*f[idx(2,1)].sub(0)\
                + 2*((0.031943828249996995663))*f[idx(4,1)].sub(0)\
                + 2*((-0.028171808490950552584))*f[idx(4,3)].sub(0)
    a4list[0][2][0][0]=a4list[2][0][0][0]
    a4list[0][0][2][0]=a4list[2][0][0][0]
    a4list[0][0][0][2]=a4list[2][0][0][0]

    a4list[2][1][0][0]=-2*((-0.026082026547865053022))*f[idx(2,1)].sub(1)\
                    - 2*((0.010647942749998998554))*f[idx(4,1)].sub(1)\
                    - 2*((-0.028171808490950552584))*f[idx(4,3)].sub(1)
    a4list[2][0][1][0]=a4list[2][1][0][0]
    a4list[2][0][0][1]=a4list[2][1][0][0]
    a4list[1][2][0][0]=a4list[2][1][0][0]
    a4list[1][0][2][0]=a4list[2][1][0][0]
    a4list[1][0][0][2]=a4list[2][1][0][0]
    a4list[0][2][1][0]=a4list[2][1][0][0]
    a4list[0][2][0][1]=a4list[2][1][0][0]
    a4list[0][1][2][0]=a4list[2][1][0][0]
    a4list[0][1][0][2]=a4list[2][1][0][0]
    a4list[0][0][2][1]=a4list[2][1][0][0]
    a4list[0][0][1][2]=a4list[2][1][0][0]

    a4list[2][2][0][0]=((0.066666666666666666667))*f[idx(0,0)].sub(0)\
                        + ((0.021295885499997997109))*f[idx(2,0)].sub(0)\
                        + 2*((0.026082026547865053022))*f[idx(2,2)].sub(0)\
                        + ((-0.038095238095238095238))*f[idx(4,0)].sub(0)\
                        + 2*((0.030116930096841707924))*f[idx(4,2)].sub(0)
    a4list[2][0][2][0]=a4list[2][2][0][0]
    a4list[2][0][0][2]=a4list[2][2][0][0]
    a4list[0][2][2][0]=a4list[2][2][0][0]
    a4list[0][2][0][2]=a4list[2][2][0][0]
    a4list[0][0][2][2]=a4list[2][2][0][0]

    a4list[2][1][1][0]=2*((-0.026082026547865053022))*f[idx(2,1)].sub(0)\
                        + 2*((0.010647942749998998554))*f[idx(4,1)].sub(0)\
                        + 2*((0.028171808490950552584))*f[idx(4,3)].sub(0)
    a4list[2][1][0][1]=a4list[2][1][1][0]
    a4list[2][0][1][1]=a4list[2][1][1][0]
    a4list[1][2][1][0]=a4list[2][1][1][0]
    a4list[1][2][0][1]=a4list[2][1][1][0]
    a4list[1][1][2][0]=a4list[2][1][1][0]
    a4list[1][1][0][2]=a4list[2][1][1][0]
    a4list[1][0][2][1]=a4list[2][1][1][0]
    a4list[1][0][1][2]=a4list[2][1][1][0]
    a4list[0][2][1][1]=a4list[2][1][1][0]
    a4list[0][1][2][1]=a4list[2][1][1][0]
    a4list[0][1][1][2]=a4list[2][1][1][0]                    

    a4list[2][2][1][0]=-2*((0)+(0.026082026547865053022))*f[idx(2,2)].sub(1)\
                    - 2*((0)+(0.030116930096841707924))*f[idx(4,2)].sub(1)
    a4list[2][2][0][1]=a4list[2][2][1][0]
    a4list[2][1][2][0]=a4list[2][2][1][0]
    a4list[2][1][0][2]=a4list[2][2][1][0]
    a4list[2][0][2][1]=a4list[2][2][1][0]
    a4list[2][0][1][2]=a4list[2][2][1][0]
    a4list[1][2][2][0]=a4list[2][2][1][0]
    a4list[1][2][0][2]=a4list[2][2][1][0]
    a4list[1][0][2][2]=a4list[2][2][1][0]
    a4list[0][2][2][1]=a4list[2][2][1][0]
    a4list[0][2][1][2]=a4list[2][2][1][0]
    a4list[0][1][2][2]=a4list[2][2][1][0]


    a4list[2][2][2][0]=2*((-0.078246079643595159065))*f[idx(2,1)].sub(0)\
                        + 2*((-0.042591770999995994217))*f[idx(4,1)].sub(0)
    a4list[2][2][0][2]=a4list[2][2][2][0]
    a4list[2][0][2][2]=a4list[2][2][2][0]
    a4list[0][2][2][2]=a4list[2][2][2][0]


    
    a4list[2][1][1][1]=-2*((-0.078246079643595159065))*f[idx(2,1)].sub(1)\
                        - 2*((0.031943828249996995663))*f[idx(4,1)].sub(1)\
                        - 2*((0.028171808490950552584))*f[idx(4,3)].sub(1)
    a4list[1][2][1][1]=a4list[2][1][1][1]
    a4list[1][1][2][1]=a4list[2][1][1][1]
    a4list[1][1][1][2]=a4list[2][1][1][1]

    a4list[2][2][1][1]=((0.066666666666666666667))*f[idx(0,0)].sub(0)\
                        + ((0.021295885499997997109))*f[idx(2,0)].sub(0)\
                        + 2*((-0.026082026547865053022))*f[idx(2,2)].sub(0)\
                        + ((-0.038095238095238095238))*f[idx(4,0)].sub(0)\
                        + 2*((-0.030116930096841707924))*f[idx(4,2)].sub(0)
    a4list[2][1][2][1]=a4list[2][2][1][1]
    a4list[2][1][1][2]=a4list[2][2][1][1]
    a4list[1][2][2][1]=a4list[2][2][1][1]
    a4list[1][2][1][2]=a4list[2][2][1][1]
    a4list[1][1][2][2]=a4list[2][2][1][1]                    

    a4list[2][2][2][1]=-2*((-0.078246079643595159065))*f[idx(2,1)].sub(1)\
                        - 2*((-0.042591770999995994217))*f[idx(4,1)].sub(1)
    a4list[2][2][1][2]=a4list[2][2][2][1]
    a4list[2][1][2][2]=a4list[2][2][2][1]
    a4list[1][2][2][2]=a4list[2][2][2][1]

    a4list[2][2][2][2]=((0.2))*f[idx(0,0)].sub(0)\
                        + ((0.2555506259999759653))*f[idx(2,0)].sub(0)\
                        + ((0.076190476190476190476))*f[idx(4,0)].sub(0)

    

    return as_tensor(a4list)
        
        


    
    