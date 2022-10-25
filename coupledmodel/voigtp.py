#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:29:42 2020

@author: daniel
"""
import numpy as np

def ind2voigt(i,j,ndim=3):

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

def voigt2ind(iv,ndim=3):
    if ndim==2:
        v2i = [(0, 0), (1, 1), (0, 1)]
    elif ndim==3:
        v2i = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    ij = v2i[iv]
    i=ij[0]
    j=ij[1]

    return i,j


def tensor2voigt(A,ndim=3):
    npoints = A.shape[0]
    if ndim==3:
        Avoigt=np.zeros((npoints,6,6))
    elif ndim==2:
        Avoigt=np.zeros((npoints,3,3))
    for i in range(ndim):
        for j in range(ndim):
            iv = ind2voigt(i,j,ndim)
            for k in range(ndim):
                for l in range(ndim):
                    jv = ind2voigt(k,l,ndim)
                    
                    Avoigt[:,iv,jv] = A[:,i,j,k,l]
    
    return Avoigt


def voigt2tensor(Avoigt,ndim=3):
    npoints = Avoigt.shape[0]
    A=np.zeros((npoints,ndim,ndim,ndim,ndim))
    for i in range(ndim):
        for j in range(ndim):
            iv = ind2voigt(i,j,ndim)
            for k in range(ndim):
                for l in range(ndim):
                    jv = ind2voigt(k,l,ndim)

                    A[:,i,j,k,l] = Avoigt[:,iv,jv]
    
    return A
