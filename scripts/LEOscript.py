#%%
from fenics import*
from mshr import *
import leopart as lp
from tqdm import tqdm
import numpy as np
import SpecCAF.GeneralSolver
import SpecCAF.spherical
import matplotlib.pyplot as plt
import Boundaries
import StokesSolvers
import Viscosities
import itertools
import FunctionSpaces
nt=15
dt=0.01

npart = 15
npmin=12
npmax=20

domain =  Boundaries.TwoCirclesSym(30,r=0.35)

FunctionSpaces.Init(domain,fabricdegree=0)

#Define Boundary Conditions
bcs = domain.ExpressionInlet(domain.W,"1.0")#-100.*pow(x[1]-0.5,2)+1.0")
#bcs = domain.Free(domain.W)
## Initilise stokes solver
stokes = StokesSolvers.ssolver(domain)

## Initilise viscosity
visc = Viscosities.anisotropic(domain)


mu = visc.Glen()

#Solve for velocity field
w = stokes.LinearAnisotropic( mu)
(u,p) = split(w)

V = domain.F2
P = domain.F

##Velocity and initial condition
# V = VectorFunctionSpace(mesh, "CG", 1)
# P = FunctionSpace(mesh,"CG",1)
# u = Function(V)
# u.assign(Constant((1.0,0.0)))
(ux,uy)=split(u)

# Initialize particles
x = lp.RandomCell(domain.mesh).generate(npart)
sh = SpecCAF.spherical.spherical(6)


##Initial fabric condtion - isotropic
fnp = np.zeros((x.shape[0],sh.nlm),dtype='complex128')
fnp[:,0] = 1.

## Initilise 
dudx_df = Function(P)
dudy_df = Function(P)
dvdx_df = Function(P)
T_df = Function(P)
T_df.assign(Constant(-20.))

## Initilise list of functions for spherical harmonic projection
f_df=[]
for ilm in range(sh.nlm):
    f_df.append(Function(V))

f_df[0].assign(Expression(("1.","0"),degree=1))




def particle2odf(p,step=5):
    fnp = np.zeros((p.number_of_particles(),sh.nlm),dtype='complex128')
    for ilm in range(sh.nlm):
        fnp[:,ilm] = np.array(p.get_property(ilm+step)[::2]) + 1j*np.array(p.get_property(ilm+step)[1::2])
    return fnp

def odf2particle(fnp,gradu,T,xnew):
    f=[]
    for ilm in range(sh.nlm):
        f.append(np.stack((fnp[:,ilm].real,fnp[:,ilm].imag),axis=1))
    
    return lp.particles(xnew,[gradu[:,0,0],gradu[:,0,1],gradu[:,1,0],T] +f, domain.mesh)
    

def edit_properties(particles, candidate_cells,fnp,step=5):

    num_properties = particles.num_properties()
    # Transform fnp into 2 column vector of real and imaginary parts
    ind =0
    for c in candidate_cells:
        for pi in range(particles.num_cell_particles(c)):
            particle_props = list(particles.property(c, pi, prop_num)
                                for prop_num in range(num_properties))



            # Edit properties
            for ilm in range(sh.nlm):
                particles.set_property(c, pi, ilm+step, Point(np.array(\
                                [fnp[ind,ilm].real, fnp[ind,ilm].imag])))
    
            ind = ind +1




##Initilise particle
gradu=np.zeros((x.shape[0],3,3))
T=np.zeros(x.shape[0])
p = odf2particle(fnp,gradu,T,x)
all_cells = [c.index() for c in cells(domain.mesh)]
ap = lp.advect_rk3(p, domain.V, w.sub(0), "open")
AD = lp.AddDelete(p,npmin,npmax,[dudx_df,dudy_df,dvdx_df,T_df]+f_df)
    

for it in tqdm(range(nt)):

    # Get new velocity gradients
    (ux,uy)=split(u)
    dudx_df = project(ux.dx(0),P)
    dudy_df = project(ux.dx(1),P)
    dvdx_df = project(uy.dx(0),P)
    

    ## Advect
    
    ap.do_step(dt)
    
    

    ##Add delete
    #
    AD.do_sweep()

    x = p.positions()

    ## Interpolate velocity gradient onto particles
    p.interpolate(dudx_df,1)
    p.interpolate(dudy_df,2)
    p.interpolate(dvdx_df,3)
    p.interpolate(T_df,4)

    ## Build numpy matrices
    gradu=np.zeros((x.shape[0],3,3))
    gradu[:,0,0] = p.get_property(1)
    gradu[:,0,1] = p.get_property(2)
    gradu[:,1,0] = p.get_property(3)
    gradu[:,1,1] = -gradu[:,0,0]
    T = np.array(p.get_property(4))

    ## Fnp at position
    fnp = particle2odf(p)

    ## Fabric evolution
    speccaf = SpecCAF.GeneralSolver.couplediterate(T,gradu,sh)
    fnp = speccaf.rk3(fnp,dt)


    ## Assign back to particles
    #p = odf2particle(fnp,gradu,T,x)
    edit_properties(p,all_cells,fnp)

    

    #Project harmonic terms onto fenics
    for ilm in range(sh.nlm):
        lstsq = lp.l2projection(p,V,ilm+5)
        lstsq.project(f_df[ilm])




    

    ## Update viscosity
    ## Update velocity field



plot(f_df[0])    

plt.figure()
fr,fi=split(f_df[1])
c=plot(fr)
plt.colorbar(c)
# %%
