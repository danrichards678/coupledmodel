from fenics import *
import numpy as np
import FenicsTransforms as ft
import os

class numpysol:
    def __init__(self,nt,domain,fabric):
        
        self.sh = fabric.sh
        self.npoints = fabric.npoints
        # To store rhostar
        self.f = np.zeros((nt,self.npoints,self.sh.nlm),dtype='complex128')
        self.f[0,...] = fabric.fnp


        # Orientation tensors
        self.a2 = np.zeros((nt,self.npoints,3,3))
        self.a4 = np.zeros((nt,self.npoints,3,3,3,3))
        self.a2[0,...] = np.moveaxis(self.sh.a2(fabric.fnp.T),-1,0)
        self.a4[0,...] = np.moveaxis(self.sh.a4(fabric.fnp.T),-1,0)

        # Velocity and pressure
        self.u = np.zeros((nt,self.npoints,2))
        self.p = np.zeros((nt,self.npoints))
        self.gradu = np.zeros((nt,self.npoints,3,3))

        # Coordinates
        self.x=fabric.mesh.coordinates()[:,0]
        self.y=fabric.mesh.coordinates()[:,1]
        if fabric.mesh.geometry().dim()==3: #3D
            self.z = fabric.mesh.coordinates()[:,2]
        self.t=fabric.mesh.cells()

        self.x_dg = fabric.F.tabulate_dof_coordinates()[:,0]
        self.y_dg = fabric.F.tabulate_dof_coordinates()[:,1]

       

    def save(self,fabric,w,it):
        (u,p) = split(w)
        #(ux,uy) = split(u)

        self.f[it,...] = fabric.fnp

        self.a2[it,...] = np.moveaxis(self.sh.a2(fabric.fnp.T),-1,0)
        self.a4[it,...] = np.moveaxis(self.sh.a4(fabric.fnp.T),-1,0)

        # self.u[it,:,0] = ft.Fenics2Numpy(ux)
        # self.u[it,:,1] = ft.Fenics2Numpy(uy)
        

        self.gradu[it,...] = fabric.gradunp

class fenicssol:
    def __init__(self,nt,domain,yslice=0.):
        # Function spaces for saving
        self.T33 = domain.T3D
        self.V = domain.V
        self.u = []
        self.p = []

        self.a2 = []

        self.ufile = File("results/u.pvd")
        self.a2file = File("results/a2.pvd")

        tol = 0.001  # avoid hitting points outside the domain
        npoints=101
        Lmax = domain.mesh.coordinates()[:,0].max()
        self.xp = np.linspace(0 + tol, Lmax - tol, npoints)
        self.points = [(x_, yslice) for x_ in self.xp]  # 2D points
        
        self.u_line = np.zeros((nt,npoints,2))
        self.a200_line = np.zeros((nt,npoints))

        

    def save(self,fabric,w,it):
        u,p = split(w)
        ux,uy = split(u)
        self.u.append(project(w.sub(0),self.V))
        if isinstance(fabric.a2,np.ndarray):
            a2 = ft.Numpy2Fenics(Function(self.T33),fabric.a2[:,:2,:2])
        else:
            a2 = project(fabric.a2,self.T33)

        self.a2.append(a2)

        # self.u_line[it,:,0] = np.array([ux(point) for point in self.points])
        # self.u_line[it,:,1] = np.array([uy(point) for point in self.points])
        # self.a200_line[it,:] = np.array([a2[0,0](point) for point in self.points])

        up = project(u,self.V)
        a2p = project(fabric.a2,self.T33)

        up.rename('u','velocity')
        a2p.rename('a2','2nd order orientation tensor')

        
        self.ufile << up
        self.a2file << a2p


class arcsol:
    def __init__(self,nt,domain,filename,visc,n,T,hx=False):
        # Function spaces for saving
        self.T33 = domain.T3D
        self.V = domain.V
        self.Q = domain.Q
        self.nt=nt
        
        self.time = np.zeros(nt)

        if n<1.01:
            type = 'linear'
        else:
            type = 'nonlinear'


        folder = filename + type + visc + 'nt' + str(nt) + 'temp' + str(T)
        dirname = '../results/' +folder
        self.dirname = dirname
        mpi_comm = MPI.comm_world
        if MPI.rank(mpi_comm) == 0:
            if os.path.exists(dirname):
                pass
            else:
                os.mkdir(dirname)

        self.upfile = File(dirname +"/u.pvd")
        self.a2pfile = File(dirname + "/a2.pvd")
        self.Tpfile = File(dirname + "/T.pvd")
        self.psipfile = File(dirname + "/psi.pvd")
        self.Wfile = File(dirname + '/W.pvd')
       
        self.uxfile = XDMFFile(dirname +"/u.xdmf")
        self.a2xfile = XDMFFile(dirname + "/a2.xdmf")
        self.Wxfile = XDMFFile(dirname + "/W.xdmf")


        if hx.any():
            self.xvec = hx
            self.ztop = np.zeros((nt,self.xvec.size))


        #Save text file
        header = 'Visc = ' + visc + 'nt = ' + str(nt) + \
            ', title = ' + filename + '\n'
        

        self.file = open(dirname + "/time.txt", "w+")
        self.file.write(header)


    def save(self,fabric,w,it,dt,T=False,isochrone=False,h=False):
        u,p = split(w)

        if it>0:
            self.time[it]=self.time[it-1]+dt

        up = project(u,self.V)
        a2p = project(fabric.a2,self.T33)

        D = sym(grad(u))
        W = skew(grad(u))

        vortnum = project(sqrt(inner(W,W)/inner(D,D)),self.Q)

        up.rename('u','velocity')
        a2p.rename('a2','2nd order orientation tensor')
        vortnum.rename('W','Vorticity Number')
        self.upfile << (up, self.time[it])
        self.a2pfile << (a2p, self.time[it])
        self.Wfile << (vortnum, self.time[it])

        if T:
            T.rename('T','Temperature (C)')
            self.Tpfile << (T, self.time[it])

        if isochrone:
            #psi = fabric.psi_df
            psi = isochrone
            psi.rename('psi','isochrones')

            self.psipfile << (psi,self.time[it])
   
        self.uxfile.write_checkpoint(up,"u",it, XDMFFile.Encoding.HDF5, True)
        self.a2xfile.write_checkpoint(a2p, "a2", it, XDMFFile.Encoding.HDF5, True )
        self.Wxfile.write_checkpoint(vortnum, "W", it, XDMFFile.Encoding.HDF5, True )

        #Write to text file
        text = "{:.0f},  {:.5e}\n".format(it,self.time[it])
        self.file.write(text)

        #
        if h.any():
            self.ztop[it,:] = h
        if it==self.nt-1:
                #Save
                if h.any():
                    np.save(self.dirname +'/height.npy', self.ztop)
                    np.save(self.dirname +'/heightx.npy',self.xvec)
                np.save(self.dirname +'/time.npy',self.time)


