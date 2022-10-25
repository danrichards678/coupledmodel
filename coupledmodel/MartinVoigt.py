#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:29:42 2020

@author: daniel
"""
from math import isnan
import numpy as np
from fenics import *
# Voigt like notation similar to Martin et al 2009

## voigt-like: [0,0], [1,1], [2,2], [0,1] where other terms are zero

def ind2voigt(i,j):
    if i==j:
        v=i
    elif i*j==0 and i+j==1:
        v=3
    else:
        v=np.nan
    return v

def voigt2ind(iv):
    v2i = [(0, 0), (1, 1), (2, 2), (0, 1)]

    ij = v2i[iv]
    i=ij[0]
    j=ij[1]

    return i,j



def tensor2voigt(T):

    Vlist = []
    for iv in range(4):
        i,j = voigt2ind(iv)
        Vlist += [[],]
        for jv in range(4):
            k,l = voigt2ind(jv)
            Vlist[iv] += [T[i,j,k,l],]


    return as_tensor(Vlist)


def voigt2tensor(V):
    Tlist = []
    for i in range(0,3):
        Tlist += [[],]
        for j in range(0,3):
            iv = ind2voigt(i,j)
            Tlist[i] += [[],]
            for k in range(0,3):
                Tlist[i][j] += [[],]
                for l in range(0,3):
                    jv = ind2voigt(k,l)
                    if isnan(iv) or isnan(jv):
                        Vivjv = 0.
                    else:
                        Vivjv = V[iv,jv]

                    Tlist[i][j][k] += [Vivjv,]

    return as_tensor(Tlist)

