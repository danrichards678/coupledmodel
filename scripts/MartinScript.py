#%%
import sys
sys.path.insert(0,'../coupledmodel/')
from fenics import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import StokesSolvers
import IceMeshes
import FabricSolve
import Viscosities
import Save
import FunctionSpaces
import SurfaceSolve2
import IsochroneSolve
set_log_level(40)
nt = 1 # number of time steps
dt = 1e-3
g=9.7692e+15; rho=9.1380e-19; A=100; H=500; acc=0.5
n=3.0
T = -10.0
L=20
visctype = 'Glen'

time = np.zeros(nt)

# Create mesh and boundary 
domain =  IceMeshes.Box(L,1,60,15)


FunctionSpaces.Init(domain,element_pair=0,fabricdegree=0)

#Define Boundary Conditions
#bcs = domain.ExpressionInlet(domain.W,"1.0")#-100.*pow(x[1]-0.5,2)+1.0")
#bcs = domain.ExpressionRight(domain.W,"5*1.0*20*pow(x[1],4)")
#bcs = domain.ExpressionRight(domain.W,"3*1.0*20*(x[1]-0.5*pow(x[1],2))")

# Init surface
surf = SurfaceSolve2.leoline(domain,1.0,n=n,h0=True)
surf.UpdateMesh(domain.mesh)
#bcs = surf.updatebcs(domain)
bcs = domain.NoSlip(domain.W)

## Initilise stokes solver
stokes = StokesSolvers.direct(domain,\
    f=Constant((0.0,-H*pow(H/acc,1/n)*rho*g)),A=A,n=n)
w=Function(domain.W)

## Initilise viscosity
visc = Viscosities.leoanisotropic(domain)

#Initial Fabric
fabric = FabricSolve.leosolver(domain,T)


#Isochrones
psibc = domain.IsochroneBC(domain.F)
isochrone = IsochroneSolve.dgmethod(domain,psibc)

# Class for solution
sol = Save.arcsol(nt,domain,'Martin',visctype,n,T,hx=surf.xvec)

#Calculate inital viscosity
mu = visc.Wrapper(visctype,fabric)

#Calculate inital velocity
w = stokes.Anisotropic(mu,bcs)

#Save inital field
sol.save(fabric,w,0,dt,isochrone=fabric.psi_df,h=surf.ztop)


for it in tqdm(range(1,nt)):

    # Find new timestep
    # maxu = w.sub(0).vector().norm('linf')
    # hmin = domain.mesh.hmin()

    # mpi_comm = MPI.comm_world
    # MPI.barrier(mpi_comm)

    # dt = CFL*MPI.max(mpi_comm,hmin)/MPI.max(mpi_comm,maxu)
    time[it] = time[it-1]*dt

    #Update fabric
    fabric.iterate(w,dt)#,surf.xvec,surf.ztop)

    #Update isoschrones
    #isochrone.iterate(w.sub(0),dt,domain.mesh)

    #Update surface
    surf.iterate(w.sub(0),domain.mesh,dt)
    #bcs = surf.updatebcs(domain)


    
    #Calculate velocity from fabric
    mu = visc.Wrapper(visctype,fabric)

    #Calculate velocity field
    w = stokes.Anisotropic(mu,bcs)
    
    # Save here
    sol.save(fabric,w,it,dt,isochrone=fabric.psi_df,h=surf.ztop)



    

plot(w.sub(0))
plt.figure()
plt.plot(surf.xvec,surf.ztop)
#plt.scatter(surf.xvec,surf.ztop)
plt.figure()
#plt.scatter(surf.xvec,surf.us)

# %%
