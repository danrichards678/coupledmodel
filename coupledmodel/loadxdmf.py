#%%
from dolfin import *
import numpy as np
import IceMeshes
import FunctionSpaces
from matplotlib import pyplot as plt


# Create mesh and boundary 
L=20
domain =  IceMeshes.Box(L,1,60,15)


FunctionSpaces.Init(domain,element_pair=0,fabricdegree=0)

W = Function(domain.Q)


folder = 'MartinnonlinearGlennt2temp-10.0'
dirname = '../results/' +folder

with XDMFFile(dirname +  "/W.xdmf") as infile:
    infile.read_checkpoint(W,"W",0)
    

mesh = W.function_space().mesh()
v = W.compute_vertex_values(mesh)
x    = mesh.coordinates()[:,0]
y    = mesh.coordinates()[:,1]
t    = mesh.cells()

vmin = 0
vmax = 1.
numLvls = 21
cmap = 'viridis'
v[v < vmin] = vmin + 1e-12
v[v > vmax] = vmax - 1e-12
from matplotlib.ticker import ScalarFormatter
levels    = np.linspace(vmin, vmax, numLvls)
formatter = ScalarFormatter()
norm      = None

fig,ax = plt.subplots()

c = ax.tricontourf(x, y, t, v,levels=levels,norm=norm, cmap = plt.get_cmap(cmap))
plt.colorbar(c)

ax.set_xlim([0,10])

# %%
