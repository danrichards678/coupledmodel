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
nt = 1 # number of time steps
CFL=0.0001

g=9.7692e+15
rho=9.1380e-19
secPerYr=31556926.
A=100#/secPerYr#3.169e-24
alpha=0.5
n=3.0
T = -20.0

time = np.zeros(nt)

# Create mesh and boundary 
domain =  IceMeshes.ISMIP(10e3,1e3,60,10)

FunctionSpaces.Init(domain,element_pair=0,fabricdegree=0)

#Define Boundary Conditions
#bcs = domain.ExpressionInlet(domain.W,"1.0")#-100.*pow(x[1]-0.5,2)+1.0")
bcs = domain.NoSlip(domain.W)

## Initilise stokes solver
stokes = StokesSolvers.direct(domain,\
    f=Constant((rho*g*np.sin(np.deg2rad(alpha)),-rho*g*np.cos(np.deg2rad(alpha)))),\
        A=A,n=n)
#stokes = StokesSolvers.ssolver(domain,f=Constant((1.0,0.0)))

## Initilise viscosity
visc = Viscosities.leoanisotropic(domain)

#Initial Fabric
fabric = FabricSolve.leosolver(domain,T)

# Init surface
surf = SurfaceSolve2.leoline(domain,0,n=n,zbot=-1000)

# Class for solution
sol = Save.fenicssol(nt,domain,yslice=0.25)


# u_0 = Expression(('x[1]','0.0'),degree=1)
# u = project(u_0,stokes.V)
set_log_level(10)
for it in tqdm(range(nt)):

    
    #Update viscosities from fabric
    mu = visc.Glen(fabric)

    #Update velocity field
    w = stokes.Anisotropic(mu,bcs)
    
    # Save here
    #sol.save(0,w,it)

    # Find timestep for next step
    maxu = w.sub(0).vector().norm('linf')
    hmin = domain.mesh.hmin()

    mpi_comm = MPI.comm_world
    MPI.barrier(mpi_comm)

    dt = CFL*MPI.max(mpi_comm,hmin)/MPI.max(mpi_comm,maxu)
    # Update fabric for next step
    #fabric.iterate(w.sub(0),dt)

    # Update mesh
    # disp = project(w.sub(0)*dt,domain.V)
    # ALE.move(domain.mesh,disp)
    surf.iterate(w.sub(0),domain.mesh,dt)

    

plot(w.sub(0))
#plt.figure()
#plt.scatter(surf.xvec,surf.ztop)
plt.figure()


ux = w.sub(0).sub(0)
uz = w.sub(0).sub(1)


# Allow extrapolation, in case coordinates are slightly outside the domain

ux.set_allow_extrapolation(True)
uz.set_allow_extrapolation(True)

# Preallocate array for velocity components
uxb = np.zeros(len(surf.xvec))
uzb = np.zeros(len(surf.xvec))

# Extract velocity along ztop
for i in range(len(surf.xvec)):
    uxb[i] = ux(surf.xvec[i], surf.ztop[i])
    uzb[i] = uz(surf.xvec[i], surf.ztop[i])

us = np.sqrt(uxb**2 + uzb**2)

plt.plot(surf.xvec,us)

            
plt.figure()
W = skew(grad(w.sub(0)))
D = sym(grad(w.sub(0)))
CG = FunctionSpace(domain.mesh,'CG',4)
vortnum = project(sqrt(inner(W,W)/inner(D,D) + 1e-7),CG)
vortnum.rename('W','Vorticity Number')
Wfile = File('icebenchmarkW.pvd')
Wfile << vortnum
# %%
