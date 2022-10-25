#%%

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
nt = 100 # number of time steps
CFL=1.0

g=9.7692e+15
rho=9.1380e-19
secPerYr=31556926.
A=100#/secPerYr#3.169e-24
alpha=0.5
n=1.0
T=-20

time = np.zeros(nt); dt=0

# Create mesh and boundary 
domain =  IceMeshes.Box(10,1,30,10)
#domain.pbc = None

FunctionSpaces.Init(domain,element_pair=0,fabricdegree=0)

#Define Boundary Conditions
#bcs = domain.ExpressionInlet(domain.W,"1.0")#-100.*pow(x[1]-0.5,2)+1.0")
bcs = domain.NoSlip(domain.W)

## Initilise stokes solver
stokes = StokesSolvers.iterative(domain,\
    f=Constant((rho*g*np.sin(np.deg2rad(alpha)),-rho*g*np.cos(np.deg2rad(alpha)))),\
        A=100,n=n)
#stokes = StokesSolvers.ssolver(domain,f=Constant((1.0,0.0)))

## Initilise viscosity
visc = Viscosities.leoanisotropic(domain)

#Initial Fabric
fabric = FabricSolve.leosolver(domain,constT=T,pbc=True)

# Init surface
surf = SurfaceSolve2.surf1d(domain)

# Class for solution
sol = Save.arcsol(nt,domain,CFL,'Box','Taylor',n,T)

# u_0 = Expression(('x[1]','0.0'),degree=1)
# u = project(u_0,stokes.V)
set_log_level(30)
for it in tqdm(range(nt)):

    
    #Update viscosities from fabric
    mu = visc.Taylor(fabric)

    #Update velocity field
    w = stokes.Anisotropic(mu)
    (u,p) = split(w)
    ux,uy = split(u)

    # Save here, so we have first save with inital fabric and velocity field
    sol.save(fabric,w,it,dt)

    # Find timestep for next step
    umag = project(sqrt(ux**2 +uy**2), domain.Q)
    maxu = umag.vector().norm('linf')
    hmin = domain.mesh.hmin()

    mpi_comm = MPI.comm_world
    MPI.barrier(mpi_comm)

    dt = CFL*MPI.max(mpi_comm,hmin)/MPI.max(mpi_comm,maxu)
    
    

    # Update fabric for next step
    fabric.iterate(w,dt)

    # Update surface
    surf.iterate(w.sub(0),domain.mesh,dt)

    


# %%
