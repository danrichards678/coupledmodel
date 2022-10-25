from typing import NewType
from fenics import *
from scipy.interpolate import interp1d
import numpy as np



def expB(x,H,L):
    return -H + 0.5*H*np.sin(2*np.pi*x/L)

def expF(x,H,L):
    return -H + 0.1*H*np.exp((-(x-L/2)**2)/((10*H)**2))

def sharpF(x,H,L):
    return -H + 0.2*H*np.exp((-(x-L/2)**2)/((0.2*H)**2))



class ISMIP:
    def __init__(self,L,H,Nx,Nz,experiment='B'):
        x = np.linspace(0,L,Nx)

        
        self.H=H
        self.L=L

        

        self.type = experiment

        self.ztop = self.topf(x)
        self.zbot = self.bottomf(x)

        self.Nx=Nx
        self.Ny = Nz
        self.xvec=x

        self.mesh = createStructuredMesh(x,self.zbot,self.ztop,Nx,Nz)

        

        # Class describing the bottom boundary of the domain
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) )
            

        class PeriodicBoundary(SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[0] = x[0] - L
                y[1] = x[1]

        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # r = -x[1] -H + 0.5*H*sin(2*np.pi*x[0]/L)
                if experiment == 'F':
                    r = -x[1] + expF(x[0],H,L)
                elif experiment == 'Sharp':
                    r = -x[1] + sharpF(x[0],H,L)
                else:
                    r = -x[1] + expB(x[0],H,L)
                return (on_boundary and near(r, 0.0, 0.01*H) )
        
       
        self.pbc = PeriodicBoundary()
        self.top = topBoundary()
        self.bottom = bottomBoundary()

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)

    #Define functions for top and bottom
    def bottomf(self,x):
        if self.type == 'F':
            z = -self.H + 0.1*self.H*np.exp((-(x-self.L/2)**2)/((10*self.H)**2))
        elif self.type=='Sharp':
            z = -self.H + 0.2*self.H*np.exp((-(x-self.L/2)**2)/((0.2*self.H)**2))
        else:
            z = -self.H + 0.5*self.H*np.sin(2*np.pi*x/self.L)
        return z

    def topf(self,x):
        return np.zeros_like(x)


        

    def NoSlip(self,W):
        bc_bottom = DirichletBC(W.sub(0), Constant((0,0)),   self.bottom)
        bc_top = DirichletBC(W.sub(0).sub(1), Constant(0.0), self.top)

        self.bcs = [bc_bottom]
        return self.bcs


class General:
    def __init__(self,x,zbot,ztop,Nz):

        
        H=ztop[0]
        self.H=H
        L = x[-1]
        self.L=L


        topf = interp1d(x,ztop)
        self.topf = topf
        bottomf = interp1d(x,zbot)
        self.bottomf = bottomf

        self.ztop = ztop    
        self.zbot = zbot

        self.Nx=x.shape[0]
        self.Ny = Nz
        self.xvec=x

        self.mesh = self.createStructuredMesh()

        

        # Class describing the bottom boundary of the domain
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                r = -x[1] + topf(x[0])
                return (on_boundary and near(r, 0.0, 0.01*H) )
            

        class PeriodicBoundary(SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[0] = x[0] - L
                y[1] = x[1]

        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                # r = -x[1] -H + 0.5*H*sin(2*np.pi*x[0]/L)
                r = -x[1] + bottomf(x[0])
                return (on_boundary and near(r, 0.0, 0.01*H) )

        class leftBoundary(SubDomain):
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )

        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], L))

        
        
       
        self.pbc = None#PeriodicBoundary()
        self.left = leftBoundary()
        self.bottom = bottomBoundary()
        self.right = rightBoundary()
        self.top = topBoundary()



        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        



    def NoSlip(self,W):
        bc_bottom = DirichletBC(W.sub(0), Constant((0.0,0.0)),   self.bottom)
        bc_left = DirichletBC(W.sub(0).sub(0), Constant(0.0), self.left)

        self.bcs = [bc_bottom,bc_left]
        return self.bcs

    def ExpressionRight(self,W,expression):
        bc_bottom = DirichletBC(W.sub(0), Constant((0.0,0.0)),   self.bottom)
        bc_left = DirichletBC(W.sub(0).sub(0), Constant(0.0), self.left)
        bc_right = DirichletBC(W.sub(0).sub(0), Expression((expression),degree=1),self.right)

        self.bcs = [bc_bottom, bc_left, bc_right]
        return self.bcs

    #Define functions for top and bottom
    def createStructuredMesh(self):
        hl = self.Nx-1 # Number of horizontal layers
        vl = self.Ny-1  # Number of vertical layers
        # generate mesh on unitsquare
        mesh = UnitSquareMesh(hl, vl)
        
        # Extract x and y coordinates from mesh
        x = mesh.coordinates()[:, 0]
        y = mesh.coordinates()[:, 1]
        x0 = min(self.xvec)
        x1 = max(self.xvec)
    
        # Map coordinates on unit square to the computational domain
        xnew = x0 + x*(x1-x0)
        ynew = self.bottomf(xnew) + y*(self.topf(xnew)-self.bottomf(xnew))
        
        xynewcoor = np.array([xnew, ynew]).transpose()
        mesh.coordinates()[:] = xynewcoor
        
        return mesh

        







class Box:
    def __init__(self,L,H,Nx,Nz):
        
        self.mesh = RectangleMesh(Point(0.0,0.0), Point(L,H), Nx-1,Nz-1)

        self.mesh = meshrefinex(self.mesh,L,n=1.3)
        self.Nx = Nx
        self.Ny = Nz
        self.L = L
        self.H = H
        self.xvec = np.linspace(0,L,Nx)


        

        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) )
            

        class leftBoundary(SubDomain):
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )

        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], L))

        class PeriodicBoundary(SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[0] = x[0] - L
                y[1] = x[1]

        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>DOLFIN_EPS and x[0]<L-DOLFIN_EPS\
                    and x[1]>0.5*H)

        self.pbc = None#PeriodicBoundary()
        self.left = leftBoundary()
        self.bottom = bottomBoundary()
        self.right = rightBoundary()
        self.top = topBoundary()

        self.lims = np.array(
            [
                [0., 0., 0., H],
                [L, L, 0., H],
                [0., L, 0., 0.],
                [0., L, H, H],
            ]
            )

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)


    def topf(self,x):
        return np.ones_like(x)

    def bottomf(self,x):
        return np.zeros_like(x)


        



    def NoSlip(self,W):
        bc_bottom = DirichletBC(W.sub(0), Constant((0.0,0.0)),   self.bottom)
        bc_left = DirichletBC(W.sub(0).sub(0), Constant(0.0), self.left)

        self.bcs = [bc_bottom,bc_left]
        return self.bcs

    def ExpressionRight(self,W,expression):
        bc_bottom = DirichletBC(W.sub(0), Constant((0.0,0.0)),   self.bottom)
        bc_left = DirichletBC(W.sub(0).sub(0), Constant(0.0), self.left)
        bc_right = DirichletBC(W.sub(0).sub(0), Expression((expression),degree=1),self.right)

        self.bcs = [bc_bottom, bc_left, bc_right]
        return self.bcs

    def IsochroneBC(self,F):
        bc_top = DirichletBC(F, Constant(0.0), self.top)
        self.psibc = [bc_top]
        return self.psibc




# Map the geometry enclosed by bottomSurf and topSurf to structured mesh
def createStructuredMesh(xvec, bottomsurf, topsurf, Nx, Nz):
    hl = Nx-1 # Number of horizontal layers
    vl = Nz-1 # Number of vertical layers
    # generate mesh on unitsquare
    mesh = UnitSquareMesh(hl, vl)
    
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

def meshrefinex(mesh,L,n=2):
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]

    

    xnew = L*pow(x/L,n)

    xynewcoor = np.array([xnew,y]).transpose()
    mesh.coordinates()[:] = xynewcoor

    return mesh


def meshrefinemid(mesh,L):
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]

    

    xnew = 0.5*L*(pow(2*x/L - 1,3) +1)

    xynewcoor = np.array([xnew,y]).transpose()
    mesh.coordinates()[:] = xynewcoor

    return mesh






