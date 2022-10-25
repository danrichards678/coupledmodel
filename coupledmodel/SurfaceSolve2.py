from fenics import *
import numpy as np
from numpy.lib.function_base import interp
from scipy.interpolate import interp1d, interp2d
import leopart as lp
from mpi4py import MPI


class surf1d:
    def __init__(self,domain,acc):
         # Define 1D mesh
        self.Nx = domain.Nx
        self.Ny = domain.Ny
        self.interval = IntervalMesh(domain.Nx-1, 0, domain.L)
        self.acc = acc

        



        class PeriodicBoundary(SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[0] = x[0] - domain.L

        if domain.pbc:
            self.pbc = PeriodicBoundary()
        else:
            self.pbc = None

      

        # Define Bubble-enriched Function space

        Qe = FiniteElement('CG',self.interval.ufl_cell(), 1)
        Be = FiniteElement('B',self.interval.ufl_cell(), 2)
        #self.M = FunctionSpace(self.interval,Qe+Be, constrained_domain=self.pbc)
        self.Q = FunctionSpace(self.interval, Qe, constrained_domain=self.pbc)
        self.M = FunctionSpace(self.interval, 'CG', 2, constrained_domain=self.pbc)

        self.xvec = np.squeeze(self.M.tabulate_dof_coordinates())
        # self.ztop = np.zeros_like(self.xvec)
        # self.zbot = -domain.H + 0.5*domain.H*np.sin(2*np.pi*self.xvec/domain.L)
        self.ztop = domain.H*np.ones_like(self.xvec)
        self.zbot = np.zeros_like(self.xvec)

        self.ux = Function(self.M)
        self.uz = Function(self.M)

         # Define trial function on V
        self.v = TrialFunction(self.M)
        
        # Define test function on V
        self.s = TestFunction(self.M)
        
        # Function for holding the old solution
        self.s0 = Function(self.M)

        # Function for holding the new solution
        self.h = Function(self.M)


    def boundaryValues(self,u):

        ux = u.sub(0)
        uz = u.sub(1)

        # Allow extrapolation, in case coordinates are slightly outside the domain

        ux.set_allow_extrapolation(True)
        uz.set_allow_extrapolation(True)
        
        # Preallocate array for velocity components
        uxb = np.zeros(len(self.xvec))
        uzb = np.zeros(len(self.xvec))
        
        # Extract velocity along ztop
        for i in range(len(self.xvec)):
            uxb[i] = ux(self.xvec[i], self.ztop[i])
            uzb[i] = uz(self.xvec[i], self.ztop[i])
            
        return uxb, uzb


    def iterate(self,u,mesh,dt):

        self.solve(u,dt)

        self.UpdateMesh(mesh)

        



    def solve(self,u,dt,eps=0.3):
       

        #Extract velocity on boundary
        uxb, uzb = self.boundaryValues(u)

        self.us = np.sqrt(uxb**2 + uzb**2)
        
        self.ux.vector()[:]=uxb
        self.uz.vector()[:]=uzb
       
        self.s0.vector().set_local(np.copy(self.ztop))

        # Accumulation
        a_s = Constant(self.acc)

        # Define bilinear and linear forms
        a = Constant(1.0/dt)*self.s*self.v*dx - self.ux*self.s*self.v.dx(0)*dx\
             + Constant(eps)*self.s.dx(0)*self.v.dx(0)*dx
        L = Constant(1.0/dt)*self.s0*self.v*dx + (a_s + self.uz)*self.v*dx

        
        
        # Define linear variational problem
        pde = LinearVariationalProblem(a, L, self.h, [])
        solver = LinearVariationalSolver(pde)
        
        # Solve variational problem
        solver.solve()

        # Save previous step 
        self.z0 = np.copy(self.ztop)

        #Update ztop
        self.ztop = project(self.h,self.M).vector().get_local()

        #return self.ztop


    def UpdateMesh(self,mesh):

        # Extract x and y coordinates from mesh
        x = mesh.coordinates()[:, 0]
        y = mesh.coordinates()[:, 1]
        
        # Map coordinates on unit square to the computational domain
        #xnew = x0 + x*(x1-x0)
        zb = interp1d(self.xvec,self.zbot,fill_value='extrapolate')
        zsa = interp1d(self.xvec,self.z0,fill_value='extrapolate')
        zsb = interp1d(self.xvec,self.ztop,fill_value='extrapolate')

        ynew = zb(x) + (zsb(x) - zb(x))*(y - zb(x))/(zsa(x)-zb(x))
    
        xynewcoor = np.array([x, ynew]).transpose()
        mesh.coordinates()[:] = xynewcoor
        mesh.bounding_box_tree().build(mesh)
        
        return mesh

    def updatebcs(self,domain,n=1):


        if n==1:
            h = self.h(domain.L)
            c = domain.L/((1/2)*h**2 - (1/6)*h**3)
            vx  = Expression('c*(x[1]-0.5*x[1]*x[1])',c=c,degree=1)

        
        bc_bottom = DirichletBC(domain.W.sub(0), Constant((0.0,0.0)),   domain.bottom)
        bc_left = DirichletBC(domain.W.sub(0).sub(0), Constant(0.0), domain.left)
        bc_right = DirichletBC(domain.W.sub(0).sub(0), vx,domain.right)

        bcs = [bc_bottom, bc_left, bc_right]

        return bcs

class leoline:
    def __init__(self,domain,acc,n=1,h0=False):
        self.Nx = domain.Nx
        self.Ny = domain.Ny
        self.acc = acc
        self.mesh = domain.mesh
        self.V = domain.V
        self.u = Function(domain.V)
        self.n=n

        #
        self.tol=0.001
        self.L = domain.L
        self.np = 150

        if h0:
            h = np.load('height.npy')
            self.xvec = np.load('heightx.npy')
            self.ztop = h[-1,:]
        else:
            self.xvec = np.linspace(0,self.L,self.np)
            self.ztop = domain.topf(self.xvec)

        self.zbot = domain.bottomf(self.xvec)
        self.z0 = np.copy(self.ztop)

        #Initial height interpolant
        self.h = interp1d(self.xvec,self.ztop,fill_value="extrapolate")

    def iterate(self,u,mesh,dt):

        self.solve(u,mesh,dt)

        self.UpdateMesh(mesh)

    def solve(self,u,mesh,dt):

        # Recreate particles
        self.xp = np.stack((self.xvec,self.ztop-self.tol), axis=-1)
        self.p = lp.particles(self.xp,[],mesh)
        self.ap = lp.advect_rk3(self.p, u.function_space(), u, "open")

        #Advect
        self.ap.do_step(dt)

        ##Add delete
        #self.AD.do_sweep()

        #Get new positions
        self.xp = self.p.positions()


        # Gather positions
        comm = MPI.COMM_WORLD
        if comm.Get_size()>1:
           
            rank = comm.Get_rank()
            root = 0

            sendbufx = np.copy(self.xp[:,0])
            sendbufz = np.copy(self.xp[:,1])

            # Collect local array sizes using the high-level mpi4py gather
            sendcounts = np.array(comm.gather(len(sendbufx), root))

            if rank == root:
                #print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
                recvbufx = np.empty(sum(sendcounts))
                recvbufz = np.empty(sum(sendcounts))
            else:
                recvbufx = None
                recvbufz = None

            comm.Gatherv(sendbuf=sendbufx, recvbuf=(recvbufx, sendcounts), root=root)
            comm.Gatherv(sendbuf=sendbufz, recvbuf=(recvbufz, sendcounts), root=root)
            #if rank == root:
                #print("Gathered array: {}".format(recvbuf))

            recvbufx = comm.bcast(recvbufx,root=root)
            recvbufz = comm.bcast(recvbufz,root=root)

            #print(recvbufz[50])

            self.xpg = np.stack((recvbufx, recvbufz), axis=-1)
        else:
            self.xpg = self.xp




        # Create height interpolant including accumulation
     
        self.h = interp1d(self.xpg[:,0],self.xpg[:,1]+self.acc*dt+self.tol,fill_value="extrapolate")
        self.z0 = np.copy(self.ztop)
        self.ztop = self.h(self.xvec)


       

    def UpdateMesh(self,mesh):

        # Extract x and y coordinates from mesh
        x = mesh.coordinates()[:, 0]
        y = mesh.coordinates()[:, 1]


        zb = interp1d(self.xvec,self.zbot,fill_value='extrapolate')
        zsa = interp1d(self.xvec,self.z0,fill_value='extrapolate')
        zsb = interp1d(self.xvec,self.ztop,fill_value='extrapolate')

        ynew = zb(x) + (zsb(x) - zb(x))*(y - zb(x))/(zsa(x)-zb(x))
    
        xynewcoor = np.array([x, ynew]).transpose()
        mesh.coordinates()[:] = xynewcoor
        mesh.bounding_box_tree().build(mesh)
        
        return mesh
    
    def updatebcs(self,domain):

       
        hL = self.h(domain.L)
        n = int(self.n)

        us = ((n+2)/(n+1))*self.acc*domain.L/hL
            
        #c = domain.L/((1/2)*h**2 - (1/6)*h**3)
        #vx  = Expression('c*(x[1]-0.5*x[1]*x[1])',c=c,degree=1)
        
        vx = Expression('c*(1.0-pow((1.0-x[1]*B),n+1))',c=us,B=1.0/hL,n=n,degree=n+1)
        #u = Expression('c*(1-pow((h-x[1])/h,n+1))',c=us,h=hL,n=n,degree=n+1)

        dHdx = (self.ztop[-1]-self.ztop[-2])/(self.xvec[-1]-self.xvec[-2])

        vy = Expression('-a*(1.0 - (1.0-x[1]*B)*((n+2)/(n+1) + pow(1.0-x[1]*B,n+1)/(n+1)))\
                        +c*(1.0-(1.0-x[1]*B))*d',a=self.acc,B=1.0/hL,n=n,c=us,d=dHdx,degree=n+1)
        
        v= Expression(('c*(1.0-pow((1.0-x[1]*B),n+1))','-a*(1.0 - (1.0-x[1]*B)*((n+2)/(n+1)\
                                                             + pow(1.0-x[1]*B,n+1)/(n+1)))\
                        +c*(1.0-(1.0-x[1]*B))*d'),a=self.acc,B=1.0/hL,n=n,c=us,d=dHdx,degree=n+1)
        
        bc_bottom = DirichletBC(domain.W.sub(0), Constant((0.0,0.0)),   domain.bottom)
        bc_left = DirichletBC(domain.W.sub(0).sub(0), Constant(0.0), domain.left)
        bc_right = DirichletBC(domain.W.sub(0).sub(0), vx,domain.right)

        bcs = [bc_bottom, bc_left, bc_right]
        domain.bcs = bcs

        return bcs




        






class leosurf:
    def __init__(self,domain,acc):
         # Define 1D mesh
        self.Nx = domain.Nx
        self.Ny = domain.Ny
        self.interval = RectangleMesh(Point(0.0,0.0), Point(domain.L,1), self.Nx-1,1)
        self.acc = acc

        self.V = domain.V
        self.xy = project(Expression(("x[0]","x[1]"),degree=1),domain.V)

        class PeriodicBoundary(SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[0] = x[0] - domain.L

        if domain.pbc:
            self.pbc = PeriodicBoundary()
        else:
            self.pbc = None

        # Number of particles
        self.npart = 5
        self.npmin = 4
        self.npmax = 12


      

        # Define Function spaces


        self.M = FunctionSpace(self.interval, 'DG', 0, constrained_domain=self.pbc)
        self.V = VectorFunctionSpace(self.interval,'DG', 0, constrained_domain=self.pbc)

        self.xvec = np.squeeze(self.M.tabulate_dof_coordinates())
        self.xvec = self.xvec[:,0]
        # self.ztop = np.zeros_like(self.xvec)
        # self.zbot = -domain.H + 0.5*domain.H*np.sin(2*np.pi*self.xvec/domain.L)
        self.ztop = domain.H*np.ones_like(self.xvec)
        self.zbot = np.zeros_like(self.xvec)

        #Initilise particle locations
        self.x = lp.RandomCell(self.interval).generate(self.npart)

        #Initial surface
        self.hp = domain.H*np.ones(self.x.shape[0])

        # Function for velocities
        self.ux = Function(self.M)
        self.uz = Function(self.M)
        self.u = Function(self.V)

        # Function for holding the old solution
        self.s0 = Function(self.M)

        # Function for holding the new solution
        self.h = Function(self.M)
        self.h.assign(Constant(domain.H))

        #Initilise particle
        self.p = lp.particles(self.x,[self.hp],self.interval)

        if self.pbc:
            #to do
            self.lims = domain.lims
        else:
            self.ap = lp.advect_rk3(self.p,self.V,self.u,"open")
        
        self.AD = lp.AddDelete(self.p,self.npmin,self.npmax,[self.h])
        self.all_cells = [c.index() for c in cells(self.interval)]




    def boundaryValuesMPI(self,u,x):
        u = project(u,self.V)

        ux = u.sub(0)
        uz = u.sub(1)

        uxarray = ux.vector().get_local()
        uzarray = uz.vector().get_local()

        xarray = self.xy.sub(0).vector().get_local()
        zarray = self.xy.sub(1).vector().get_local()

        mpi_comm = MPI.comm_world
        MPI.barrier(mpi_comm)

        

        ux_g = mpi_comm.gather(uxarray,root=0)
        uz_g = mpi_comm.gather(uzarray,root=0)

        xg = mpi_comm.gather(xarray,root=0)
        zg = mpi_comm.gather(zarray,root=0)

        x = mpi_comm.gather(x,root=0)

        if mpi_comm.Get_rank() == 0:

            ux_g = np.concatenate(ux_g,axis=0)
            uz_g = np.concatenate(uz_g,axis=0)
            xg = np.concatenate(xg,axis=0)
            zg = np.concatenate(zg,axis=0)

            print(ux_g.shape)
            print(xg.shape)
            
            uxi = interp2d(xg,zg,ux_g)
            uzi = interp2d(xg,zg,uz_g)
            
            
            # Preallocate array for velocity components
            uxb = np.zeros(len(x))
            uzb = np.zeros(len(x))
            
            # Extract velocity along ztop
            for i in range(len(x)):
                uxb[i] = uxi(x[i], self.h(x[i],0))
                uzb[i] = uzi(x[i], self.h(x[i],0))

        else:
            uxb = None
            uzb = None
        
        mpi_comm.Bcast(uxb)
        mpi_comm.Bcast(uzb)
            
        return uxb, uzb


    def boundaryValues(self,u,x):


        ux = u.sub(0)
        uz = u.sub(1)

        
        # Allow extrapolation, in case coordinates are slightly outside the domain

        ux.set_allow_extrapolation(True)
        uz.set_allow_extrapolation(True)
        
        # Preallocate array for velocity components
        uxb = np.zeros(len(x))
        uzb = np.zeros(len(x))
        
        # Extract velocity along ztop
        for i in range(len(x)):
            uxb[i] = ux(x[i], self.h(x[i],0))
            uzb[i] = uz(x[i], self.h(x[i],0))
            
        return uxb, uzb

    def boundaryValuesMPI2(self,u,x,mesh):
        ux = u.sub(0)
        uz = u.sub(1)

        
        # Allow extrapolation, in case coordinates are slightly outside the domain

        ux.set_allow_extrapolation(True)
        uz.set_allow_extrapolation(True)
        
        # Preallocate array for velocity components
        uxb = np.zeros(len(x))
        uzb = np.zeros(len(x))
        
        # Extract velocity along ztop
        for i in range(len(x)):
            point = Point(x[i], self.h(x[i],0))
            if mesh.bounding_box_tree().compute_first_entity_collision(point) < mesh.num_cells():

                uxb[i] = ux(x[i], self.h(x[i],0))
            uzb[i] = uz(x[i], self.h(x[i],0))
            
        return uxb, uzb



    def iterate(self,u,mesh,dt):


        # mpi_comm = MPI.comm_world
        # if MPI.size(mpi_comm)>1:


        

        self.solve(u,dt)

        self.UpdateMesh(mesh)

        

    def edit_properties(self,particles, candidate_cells,hpnew):

        num_properties = particles.num_properties()
        # Transform fnp into 2 column vector of real and imaginary parts
        ind =0
        for c in candidate_cells:
            for pi in range(particles.num_cell_particles(c)):
                particle_props = list(particles.property(c, pi, prop_num)
                                    for prop_num in range(num_properties))


                x = particle_props[0].x()
                # Edit properties
                #particles.set_property(c, pi, 1, Point(hpnew))
                particles.set_property(c,pi, 1, Point(hpnew[ind]))           
        
                ind = ind +1

    def solve(self,u,dt):

        #Extract velocity on boundary
        if MPI.size(MPI.comm_world)==1:
            uxb, uzb = self.boundaryValues(u,self.xvec)
        else:
            uxb, uzb = self.boundaryValuesMPI(u,self.xvec)


        self.us = np.sqrt(uxb**2 + uzb**2)
        
        self.ux.vector()[:]=uxb
        self.uz.vector()[:]=uzb



        # Update velocity vield
        assign(self.u.sub(0),self.ux)
       

        #Advect particles
        self.ap.do_step(dt)

        #Add delete
        self.AD.do_sweep()

        #Get new positions
        self.x = self.p.positions()
        


        #Update height
        uxp,uzp = self.boundaryValues(u,self.x[:,0])
        hp = np.array(self.p.get_property(1))
        self.hpnew = hp + dt*(self.acc + uzp)
        self.edit_properties(self.p,self.all_cells,self.hpnew)


        # Update fenics projection
        lstsq = lp.l2projection(self.p,self.M,1)
        lstsq.project(self.h)

        # Save previous step 
        self.z0 = np.copy(self.ztop)

        #Update ztop
        self.ztop = project(self.h,self.M).vector().get_local()

        #Gather z0 and ztop
        # if MPI.size(MPI.comm_world)>1:
        #     mpi_comm = MPI.comm_world
        #     MPI.barrier(mpi_comm)

            # self.z0 = mpi_comm.gather(self.z0,root=0)
            # self.ztop = mpi_comm.gather(self.ztop,root=0)

            # mpi_comm.Bcast(self.z0,root=0)
            # mpi_comm.Bcast(self.ztop,root=0)




        #return self.ztop


    def UpdateMesh(self,mesh):

        # Extract x and y coordinates from mesh
        x = mesh.coordinates()[:, 0]
        y = mesh.coordinates()[:, 1]
        
        # Map coordinates on unit square to the computational domain
        #xnew = x0 + x*(x1-x0)

        if MPI.size(MPI.comm_world)>1:
            mpi_comm = MPI.comm_world
            MPI.barrier(MPI.comm_world)

            self.z0 = mpi_comm.gather(self.z0,root=0)
            self.ztop = mpi_comm.gather(self.ztop,root=0)
            self.xvec = mpi_comm.gather(self.xvec,root=0)


            mpi_comm.Bcast(self.z0,root=0)
            mpi_comm.Bcast(self.ztop,root=0)
            mpi_comm.Bcast(self.xvec,root=0)


        zb = interp1d(self.xvec,self.zbot,fill_value='extrapolate')
        zsa = interp1d(self.xvec,self.z0,fill_value='extrapolate')
        zsb = interp1d(self.xvec,self.ztop,fill_value='extrapolate')

        ynew = zb(x) + (zsb(x) - zb(x))*(y - zb(x))/(zsa(x)-zb(x))
    
        xynewcoor = np.array([x, ynew]).transpose()
        mesh.coordinates()[:] = xynewcoor
        mesh.bounding_box_tree().build(mesh)
        
        return mesh

    def updatebcs(self,domain,n=1):

        h = self.h(domain.L,0.0)
        
            
        c = domain.L/((1/2)*h**2 - (1/6)*h**3)
        vx  = Expression('c*(x[1]-0.5*x[1]*x[1])',c=c,degree=1)
        


        
        bc_bottom = DirichletBC(domain.W.sub(0), Constant((0.0,0.0)),   domain.bottom)
        bc_left = DirichletBC(domain.W.sub(0).sub(0), Constant(0.0), domain.left)
        bc_right = DirichletBC(domain.W.sub(0).sub(0), vx,domain.right)

        bcs = [bc_bottom, bc_left, bc_right]

        return bcs



