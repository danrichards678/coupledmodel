#%%
from fenics import*
from mshr import *
import leopart as lp
import numpy as np
from matplotlib import pyplot as plt


nt=30
dt = 0.01


npart = 15
npmin=12
npmax=20


mesh = UnitSquareMesh(20,20)

#Velocity 
V = VectorFunctionSpace(mesh, "DG", 0)
VCG = VectorFunctionSpace(mesh,"CG",2)
P = FunctionSpace(mesh,"DG",1)
u = Function(VCG)
u.assign(Expression(("0.0","0.0"),degree=1))
(ux,uy)=split(u)

udg = project(u,V)

# Initialize particles
x = lp.RandomCell(mesh).generate(npart)

def numpy2particle(xnew,q,r):
    return lp.particles(xnew,[q, r], mesh)


def particle2numpy(p):
    x = p.positions()
    a = np.array(p.get_property(1))
    b = np.array(p.get_property(2))
    return x,a,b

# Initial particle
a = np.zeros(x.shape[0])
b = np.zeros(x.shape[0])
a[x[:,0]<0.5]=1.0
p = numpy2particle(x,a,b)

# Functions for interpolating
a_df = Function(P)
b_df = Function(P)

ap = lp.advect_rk3(p, V, u, "open")
AD = lp.AddDelete(p,npmin,npmax,[a_df,b_df])
for it in range(nt):

     #Advect particles
    u.assign(Expression(("1.0","0.0"),degree=1))
    ap.do_step(dt)
    
    #Add delete
    
    AD.do_sweep()


    # x,a,b = particle2numpy(p)

    # ## Change properties
    # a= 2*b
    # b= 5. + a        ##just as a minimal example 

    # # Update particle by creating a new particle
    # p = numpy2particle(x,a,b)

    # Create fenics functions for add delete
    lstsqa = lp.l2projection(p,P,1)
    lstsqa.project(a_df)

    lstsqb = lp.l2projection(p,P,1)
    lstsqb.project(b_df)

c=plot(a_df)
plt.colorbar(c)


# %%
