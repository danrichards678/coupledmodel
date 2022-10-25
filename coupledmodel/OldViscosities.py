#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:52:30 2020

@author: daniel
"""

import numpy as np
import voigt

I=np.eye(3)

           
def Boehler(A,A4,xi1,xi2):
    Ip = np.dstack([I]*A.shape[0])
    Ip = np.moveaxis(Ip,-1,0)
    return np.einsum('pik,pjl->pijkl',Ip,Ip) + xi1*A4 \
           + xi2*( np.einsum('pik,pjl->pijkl',Ip,A) + np.einsum('pik,pjl->pijkl',A,Ip) )\
           -(1/3)*(xi1+2*xi2)*np.einsum('pij,pkl->pijkl',Ip,A)    
           
def InvBoehler(A,A4,xi1,xi2):
    Fluidity = Boehler(A,A4,xi1,xi2)
    FluidityV = voigt.tensor2voigt(Fluidity)
    ViscosityV = np.linalg.inv(FluidityV)
    
    return voigt.voigt2tensor(ViscosityV)
           
def Taylor(A,A4,gamma,beta):
    xi1 = 2*(gamma-beta)
    xi2 = beta - 1
    return Boehler(A,A4,xi1,xi2)     


def Static(A,A4,gamma,beta):
    xi1=2*((gamma+2.)/(4*gamma-1) - 1/beta)
    xi2=(1/beta) - 1.
    return InvBoehler(A,A4,xi1,xi2)

def EulerAngles(A):
    ai,ev = np.linalg.eig(A.real)
    
    # Right hand orthonormal basis
    ev[0,2]=ev[1,0]*ev[2,1]-ev[2,0]*ev[1,1]
    ev[1,2]=ev[2,0]*ev[0,1]-ev[0,0]*ev[2,1]
    ev[2,2]=ev[0,0]*ev[1,1]-ev[1,0]*ev[0,1]
    
    #normalize
    norm=np.sqrt(ev[0,2]**2+ev[1,2]**2+ev[2,2]**2)
    ev[:,2]=ev[:,2]/norm
    
    euler=np.zeros(3)
    euler[1] = np.arccos(ev[2,2])
    if euler[1]>0:
        euler[0]=np.arctan2(ev[0,2],-ev[1,2])
        euler[2]=np.arctan2(ev[2,0],ev[2,1])
    else:
        euler[1]=np.arctan2(ev[1,0],ev[0,0])
        euler[2]=0.
    
    return euler

#def Golf(A,gamma,beta):
#    #Eigen decompoisiton
#    eigVals,eigVecs=np.linalg.eig(A.real)
#    #Sort with largest first
#    idx = eigVals.argsort()[::-1]   
#    eigVals = eigVals[idx]
#    eigVecs = eigVecs[:,idx]
#    # Get Viscosities
#    eta = GolfViscs(gamma,beta)
#    
#    M0=
#    return 
#    
    

def Deformability(D,A,A4):
    I = np.eye(3)
    IikAjl = np.einsum('ik,pjl->pijkl',I,A)    
    return (5*np.einsum('pij,pkl,pijkl->p',D,D,IikAjl-A4)/np.einsum('pij,pji->p',D,D))


def Enhancement(Def,Emax=10,Emin=0.1):
    # Fit for Emin=0.1, Emax=10
    tau = (8/21)*(Emax-1)/(1-Emin)
    
    E=np.zeros_like(Def)
    E[Def<1]=(1-Emin)*pow(Def[Def<1],tau) + Emin
    E[Def>=1]=(4*Def[Def>=1]**2*(Emax-1) + 25. -4*Emax)/21.
    E[Def<=0]=Emin
    
    
    return E


def Caffe(D,A,A4,n=3.0):
    Def=Deformability(D,A,A4)
    E=Enhancement(Def)
    return pow(E,-1./n)