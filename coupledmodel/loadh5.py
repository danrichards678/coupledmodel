#%%
from dolfin import *
import numpy as np
import BoundaryConditions
import FunctionSpaces


# Load mesh define boundaries
filename = "RoundChannel80.xml"
domain =  BoundaryConditions.RoundChannel(filename)

FunctionSpaces.Init(domain,fabricdegree=1)

up = Function(domain.V)

nt = 7
CFL = 15.
visc = 'Taylor'

folder = 'linear' + visc + 'nt' + str(nt) + 'CFL' + str(CFL) + 'mesh' + filename[:-4]
dirname = '../results/' +folder

ufile = HDF5File(domain.mesh.mpi_comm(), dirname +  "/u.h5", "r")


ufile.read(up)