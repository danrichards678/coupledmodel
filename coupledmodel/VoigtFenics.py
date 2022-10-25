#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:29:42 2020

@author: daniel
"""
import numpy as np
from fenics import *



def ind2voigt(i,j,ndim=2):

    if ndim==2:
        if i==j:
            v=i
        elif i*j==0 and i+j==1:
            v=2
    elif ndim==3:
        if i==j:
            v=i
        elif i+j==3:
            v=3
        elif i*j==0 and i+j==2:
            v=4
        elif i*j==0 and i+j==1:
            v=5 
    return v

def voigt2ind(iv,ndim):
    if ndim==2:
        v2i = [(0, 0), (1, 1), (0, 1)]
    elif ndim==3:
        v2i = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    ij = v2i[iv]
    i=ij[0]
    j=ij[1]

    return i,j





def tensor2voigt(T,ndim=2):

    if ndim==2:
        V = as_matrix([ [T[0,0,0,0], T[0,0,1,1], T[0,0,0,1]],\
                        [T[1,1,0,0], T[1,1,1,1], T[1,1,0,1]],\
                        [T[0,1,0,0], T[0,1,1,1], T[0,1,0,1]] ])

    elif ndim==3:
        Vlist = []
        for iv in range(6):
            i,j = voigt2ind(iv,ndim)
            Vlist += [[],]
            for jv in range(6):
                k,l = voigt2ind(jv,ndim)
                Vlist[iv] += [T[i,j,k,l],]


    return as_tensor(Vlist)


def voigt2tensor(V,ndim=2):
    Tlist = []
    for i in range(0,ndim):
        Tlist += [[],]
        for j in range(0,ndim):
            iv = ind2voigt(i,j,ndim)
            Tlist[i] += [[],]
            for k in range(0,ndim):
                Tlist[i][j] += [[],]
                for l in range(0,ndim):
                    jv = ind2voigt(k,l,ndim)

                    Tlist[i][j][k] += [V[iv,jv],]

    return as_tensor(Tlist)


def threeDto2D(T):
    T2dlist = []
    for i in range(2):
        T2dlist += [[],]
        for j in range(2):
            if len(T.ufl_shape)==2:
                T2dlist[i] += [T[i,j],]
            else:
                T2dlist[i] += [[],]
                for k in range(2):
                    T2dlist[i][j] += [[],]
                    for l in range(2):
                        T2dlist[i][j][k] += [T[i,j,k,l],]

    return as_tensor(T2dlist)