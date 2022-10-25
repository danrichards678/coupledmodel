#%%

from fenics import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import StokesSolvers
import BoundaryConditions
import FabricSolve
import Viscosities
import Save
import FunctionSpaces
nt = 2 # number of time stepss
CFL=1.
time = np.zeros(nt)

# Create mesh and boundary 
domain =  BoundaryConditions.RoundChannel("RoundChannel30.xml")

FunctionSpaces.Init(domain,fabricdegree=1)

#Define Boundary Conditions
bcs = domain.ExpressionInlet(domain.W,"1.0")#-100.*pow(x[1]-0.5,2)+1.0")
#bcs = domain.Free(domain.W)

## Initilise stokes solver
alpha =0.5
stokes = StokesSolvers.ssolver(domain)#,f=Constant((np.sin(np.deg2rad(alpha)),np.cos(np.deg2rad(alpha)))))
w = Function(domain.W)

## Initilise viscosity
visc = Viscosities.leoanisotropic(domain)

#Initial Fabric
fabric = FabricSolve.leosolver(domain)

# Class for solution
sol = Save.fenicssol(nt,domain,yslice=0.25)


# u_0 = Expression(('x[1]','0.0'),degree=1)
# u = project(u_0,stokes.V)
for it in tqdm(range(nt)):
    
    if it>0:
        time[it]=time[it-1]+dt
    
    #Update viscosities from fabric
    mu = visc.Glen(fabric)

    #Update velocity field
    w = stokes.LinearAnisotropic(mu)
    (u,p) = split(w)
    ux,uy = split(u)

    # Save here, so we have first save with inital fabric and velocity field
    sol.save(fabric,w,it)

    # Find timestep for next step
    dt = CFL*domain.mesh.hmin()/np.max(np.abs(w.sub(0).vector().get_local()))
    
    # Update fabric for next step
    fabric.iterate(w.sub(0),dt)


c=plot(sol.u[-1].sub(0))
plt.colorbar(c)

plt.figure()
c=plot(sol.u[-1].sub(0)-sol.u[0].sub(0))
plt.colorbar(c)

plt.figure()
c=plot(sol.a2[-1][0,0])
plt.colorbar(c)

plt.figure()
D = sym(grad(u))
W = skew(grad(u))
from ufl import i,j
vortnumber = sqrt((W[i,j]*W[i,j])/(D[i,j]*D[i,j]))
c=plot(vortnumber)
plt.colorbar(c)

# plt.figure()
# plt.plot(sol.xp,sol.u_line[-1,:,0])
# plt.plot(sol.xp,sol.a200_line[-1,:])
# %%
