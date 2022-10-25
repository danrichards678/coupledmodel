#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:05:28 2021

@author: daniel
"""

from fenics import*
from ufl import i, j, k, l
import numpy as np
import voigtp as voigt
import VoigtFenics as vf
import FenicsTransforms as ft
import fenicsinverse as fi
import MartinVoigt as mv
#import golf as g




class leoanisotropic:
    def __init__(self,domain,gamma=1.,beta=0.01,n=3.0):

        self.ndim = domain.mesh.geometric_dimension()
        self.T2 = domain.T2
        self.T4 = domain.T4
        self.gamma=gamma
        self.beta=beta
        self.T33 = domain.T3D
        self.V3 = domain.F3
        self.T3333 = domain.T3333
        self.T66 = domain.T66
        self.F = domain.F
        self.n = n


    def Wrapper(self,type,fabric,u=0):

        if type=='Glen':
            mu = self.Glen()
        elif type=='Caffe':
            mu = self.Caffe(fabric,u)
        elif type=='Taylor':
            mu = self.Taylor(fabric)
        elif type=='Static':
            mu = self.Static(fabric)
        elif type=='Golf':
            mu = self.Golf(fabric)
        
        return mu
        

    def Glen(self,fabric=0):
        #fabric is dummy
        I = Identity(self.ndim)
        mu = as_tensor(I[i,k]*I[j,l],(i,j,k,l))
        return mu

    
   

    def Caffe(self,fabric,u):
        # Strain-rate tensor
        def Deformability(u,A2,A4):
            D = sym(grad(u))
            I=Identity(2)
            Def = 5*(D[i,j]*D[k,l]*I[i,k]*A2[j,l] - D[i,j]*D[k,l]*A4[i,j,k,l])/inner(D,D)
            return Def

        def InvEnhancement(Def):
            

            # Fit for 1/E with Emax=10, Emin =0.1, Adj R2 = 1
            p1 = 0.6528
            p2 = -1.678
            p3 = 1.944
            q1 = -0.1487
            q2 = -0.1467
            q3 = 0.194
            InvE = (p1*Def**2. + p2*Def + p3) /(Def**3 + q1*Def**2 + q2*Def + q3)
            return InvE

        I = Identity(self.ndim)
        
        if u.vector().norm('l2')<1e-9: # no velocity field
            InvE = 1.0
        else:
            a2=fabric.a2
            a4=fabric.a4
            if self.ndim==2:
                a2 = vf.threeDto2D(a2)
                a4 = vf.threeDto2D(a4)
            Def = Deformability(u,a2,a4)
            InvE = InvEnhancement(Def)

        self.mu = pow(InvE,1.0/self.n)*as_tensor(I[i,k]*I[j,l],(i,j,k,l))
            
        return self.mu


    def Boehler(self,a2,a4,xi1,xi2):
        I = Identity(a2.ufl_shape[0])
        mu = as_tensor( (I[i,k]*I[j,l] + xi1*a4[i,j,k,l] \
            + xi2*(I[i,k]*a2[j,l] + a2[i,k]*I[j,l]) \
            -(1/3)*(xi1+2*xi2)*I[i,j]*a2[k,l]), (i,j,k,l) )

        return mu


    def BoehlerVoigt(self,a2,a4,xi1,xi2):
        ## Currently only for 2d flow
        mu = self.Boehler(a2,a4,xi1,xi2)

        muv = mv.tensor2voigt(mu)
        return muv

    def Static(self,fabric):
        xi1=2*((self.gamma+2.)/(4*self.gamma-1) - 1/self.beta)
        xi2=(1/self.beta) - 1.

        FluidityV = self.BoehlerVoigt(fabric.a2,fabric.a4,xi1,xi2)

        muV = fi.getMatrixInverse(FluidityV)

        mu = mv.voigt2tensor(muV)

        if self.ndim==2:
            mu = vf.threeDto2D(mu)
        # Normalise muisotropic0000 value, currently only for gamma=1
        beta = self.beta
        mu0000 = (0.145415111111111*beta**7 + 0.34438637037037*beta**6 + 0.320505679012346*beta**5 + 0.148922469135802*beta**4 + 0.0362192592592592*beta**3 + 0.00434883950617284*beta**2 + 0.000202271604938271*beta)/(0.100672*beta**7 + 0.290048*beta**6 + 0.336213333333333*beta**5 + 0.199300740740741*beta**4 + 0.0631466666666666*beta**3 + 0.0100124444444444*beta**2 + 0.000606814814814814*beta - 3.08395284618099e-20)
        return mu/mu0000

    def Staticnp(self,fabric):
        xi1=2*((self.gamma+2.)/(4*self.gamma-1) - 1/self.beta)
        xi2=(1/self.beta) - 1.

        a2np = ft.Fenics2Numpy(project(fabric.a2,self.T33))
        a4np = ft.Fenics2Numpy(project(fabric.a4,self.T3333))

        munp = InvBoehlernp(a2np,a4np,xi1,xi2)
    
        mu = Function(self.T4)
        mu = ft.Numpy2Fenics(mu,munp[:,:self.ndim,:self.ndim,:self.ndim,:self.ndim])

        beta = self.beta
        mu0000 = (0.145415111111111*beta**7 + 0.34438637037037*beta**6 + 0.320505679012346*beta**5 + 0.148922469135802*beta**4 + 0.0362192592592592*beta**3 + 0.00434883950617284*beta**2 + 0.000202271604938271*beta)/(0.100672*beta**7 + 0.290048*beta**6 + 0.336213333333333*beta**5 + 0.199300740740741*beta**4 + 0.0631466666666666*beta**3 + 0.0100124444444444*beta**2 + 0.000606814814814814*beta - 3.08395284618099e-20)

        return mu/mu0000




    def Taylor(self,fabric):
        xi1 = 2*(self.gamma-self.beta)
        xi2 = self.beta - 1
        mu = self.Boehler(fabric.a2,fabric.a4,xi1,xi2)

        if self.ndim==2:
            mu = vf.threeDto2D(mu)

        # Normalise by isotropic 0000 values
        muiso0000 = 4*self.beta/15 + 8.*self.gamma/45 + 5/9
        return mu/muiso0000


    # def Golf(self,fabric):

    #     a2np = ft.FenicsMatrix2Numpy(fabric.a2,self.F)

    #     munp = g.Golf(a2np,self.gamma,self.beta)

    #     mu = Function(self.T4)
    #     mu = ft.Numpy2Fenics(mu,munp[:,:self.ndim,:self.ndim,:self.ndim,:self.ndim])
        
    #     #Normalise by isotropic 0000 value
    #     a2iso = np.array([[1./3, 0., 0.],[0, 1./3, 0],[0, 0, 1./3]])
    #     muiso0000 = g.golfsingle(a2iso,self.gamma,self.beta)[0,0,0,0]


    #     return mu/muiso0000
        

def InvBoehlernp(a2,a4,xi1,xi2):
        Fluidity = Boehlernp(a2,a4,xi1,xi2)
        FluidityV = voigt.tensor2voigt(Fluidity,3)
        ViscosityV = np.linalg.inv(FluidityV)
        
        munp = voigt.voigt2tensor(ViscosityV,3)
        return munp

def Boehlernp(a2,a4,xi1,xi2):
        Ip = np.dstack([np.eye(3)]*a2.shape[0])
        Ip = np.moveaxis(Ip,-1,0)

        munp = np.einsum('pik,pjl->pijkl',Ip,Ip) + xi1*a4 \
           + xi2*( np.einsum('pik,pjl->pijkl',Ip,a2) \
                +  np.einsum('pik,pjl->pijkl',a2,Ip) )\
           -(1/3)*(xi1+2*xi2)*np.einsum('pij,pkl->pijkl',Ip,a2)

        return munp
