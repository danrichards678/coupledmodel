#%%
import numpy as np# numpy for arrays
from tqdm import tqdm
import SpecCAF.Solver as Solver
import SpecCAF.spherical as spherical
from matplotlib import pyplot as plt
import os.path
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 12,
    "figure.autolayout" : True,
    
})

sh = spherical.spherical(6)


#Set initial Conditon
f0=sh.spec_array()
f0[0]=1
#f0=array([ 1.00000000e+00+0.j,  8.21125683e-01+0.j,  2.96741458e-01+0.j,\
#       -1.16961065e-01+0.j,  2.47496224e-01+0.j,  3.05565284e-01+0.j,\
#       -2.41313588e-02+0.j, -4.84042102e-02+0.j,  1.71592547e-02+0.j,\
#        5.39720834e-04+0.j,  1.45737667e-01+0.j,  3.31027510e-02+0.j,\
#       -3.68306340e-02+0.j, -2.47069987e-04+0.j,  6.57737643e-03+0.j,\
#       -2.19545421e-03+0.j])


strain = np.linspace(0,20,100)
p=Solver.params(-20,strain,1)

# Overwrite gradu
# p.gradu =  np.array([[0., 0, 1.],\
#                       [0., 0., 0.],\
#                       [0, 0., 0.]])
            
p.gradu = np.array([[ 1.0, 0.0,  0.        ],\
       [0.0, -1.0,  0.        ],\
       [ 0.        ,  0.        ,  0.        ]])
p.D = 0.5*(p.gradu+p.gradu.T)
p.W = 0.5*(p.gradu-p.gradu.T)
p.D2 = np.einsum('ij,ji',p.D,p.D)
p.effectiveSR = np.sqrt(p.D2)
p.octSR = np.sqrt(2*p.D2/3)



sol=Solver.rk3solve(p,sh,f0)

fig,ax = plt.subplots(figsize=(3,3))

xx,yy,fgrid=sh.polefigure(sol.y[:,-1])
cs=ax.contourf(xx,yy,fgrid,20,vmin=0)

ax.axis('off')
ax.set_aspect(1)
cbar_ax = fig.add_axes([0.1, 0, 0.8, 0.03])
clb=fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',ticks=[0,0.065,0.13])
clb.set_label('$\\rho^*$')



plt.figure()
Avec=sh.a2(sol.y)
plt.plot(sol.t,Avec.reshape((9,sol.t.size)).T)
