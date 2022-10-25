#%%

from fenics import *
import numpy as np
import StokesSolvers
import BoundaryConditions
import FabricSolve
import Viscosities
import Save
import FunctionSpaces
import TemperatureSolve
from tqdm import tqdm
parameters['ghost_mode'] = 'shared_facet' 
parameters["krylov_solver"]["nonzero_initial_guess"] = True


nt = 50 # number of time stepss
CFL=0.1
n=1.0
Tin = -20.0
g=9.7692e+15
rho=9.1380e-19
A=100


time = np.zeros(nt); dt=0 

# Load mesh define boundaries
filename = "RoundChannel30.xml"
domain =  BoundaryConditions.RoundChannel(filename)

FunctionSpaces.Init(domain,element_pair=0,fabricdegree=0)

#Define Boundary Conditions
#bcs = domain.ExpressionInlet(domain.W,"-100.*pow(x[1]-0.5,2)+1.0")
Tbcs = domain.TemperatureBC(domain.F,Tin=Tin)
bcs = domain.Free(domain.W)

## Initilise stokes solver
stokes = StokesSolvers.iterative(domain,n=n,A=A,p_in=1e1,tol=1e-4)#,f=Constant((np.sin(np.deg2rad(alpha)),np.cos(np.deg2rad(alpha)))))

## Initilise viscosity
visc = Viscosities.leoanisotropic(domain)

## Initilise temperature
temp = TemperatureSolve.solv(domain,T0=Tin,n=n,rho=rho)
T=temp.T0

#Initial Fabric
fabric = FabricSolve.leosolver(domain,T)

# Class for solution
sol = Save.arcsol(nt,domain,CFL,filename,'Taylor',n,Tin)

# u_0 = Expression(('1.0','0.0','0.0'),degree=1)
# w = project(u_0,domain.W)
set_log_level(30)
for it in tqdm(range(nt)):
    
    
    #Update viscosities from fabric
    mu = visc.Taylor(fabric)

    #Update velocity field
    w = stokes.Anisotropic(mu)
    (u,p) = split(w)
    ux,uy = split(u)

    # Save here, so we have first save with inital fabric and velocity field
    sol.save(fabric,w,it,dt,T)

    # Find timestep for next step
    umag = project(sqrt(ux**2 +uy**2), domain.Q)
    maxu = umag.vector().norm('linf')
    hmin = domain.mesh.hmin()

    mpi_comm = MPI.comm_world
    MPI.barrier(mpi_comm)

    dt = CFL*MPI.max(mpi_comm,hmin)/MPI.max(mpi_comm,maxu)
    
    
    # Update temperature for next step
    T = temp.iterate(u,mu,dt,fabric.T_df)

    # Update fabric for next step
    fabric.iterate(w,T,dt)

sol.file.write('done!')
sol.file.close()

list_timings(TimingClear.clear, [TimingType.wall])

# %%
