#%%

from fenics import *
from mshr import *
from tqdm import tqdm
from ufl import i, j, k, l
import SpecCAF.spherical as spherical
import numpy as np
import matplotlib.pyplot as plt
import StokesSolvers
import Boundaries
import FabricSolve
import Viscosities
import Save
import closuresvect
# Create mesh and boundary 

domain =  Boundaries.Hshape(60)
mesh = domain.mesh

## Initilise stokes solver
stokes = StokesSolvers.ssolver(mesh)

## Initilise viscosity
visc = Viscosities.anisotropic(mesh,element='DG',degree=0)

#Define Boundary Conditions
bcs = domain.PressureDrop(stokes.W)

#Set fabric field
class fabric:
    pass
xy = visc.P.tabulate_dof_coordinates()
fabric.npoints = visc.npoints
# fabric.a2 = np.zeros((fabric.npoints,3,3))
# fabric.a2[:,1,1] = xy[:,1]
# fabric.a2[:,2,2] = 1-xy[:,1]

a2 = np.array([[0.0, -0.0, 0.],\
              [0, 0.5, 0.],\
              [0., 0., 0.5]])
fabric.a2= np.stack([a2]*fabric.npoints)

fabric.a4 = closuresvect.compute_closure(fabric.a2)

# Class for solution
#sol = Save.solution(nt,fabric)


    
    
#Find viscosities from fabric
mu = visc.Static(fabric)
#mu = visc.Glen()

#Solve for velocity field
w = stokes.LinearAnisotropic(bcs, mu)
#w = stokes.Caffe(bcs,fabric,n=3.0)
(u,p) = split(w)


# Save

plt.figure()
(ux,uy) = split(u)
umag = sqrt(ux**2+uy**2)
c=plot(umag)
plt.colorbar(c)
# %%
