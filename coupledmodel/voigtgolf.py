#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:29:42 2020

@author: daniel
"""
import numpy as np


voigt2ind = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]


def ind2voigt(i,j):
    if i==j:
        v=i
    elif i+j==3: # 1,2
        v=4
    elif i*j==0 and i+j==2: # 2,0
        v=5
    elif i*j==0 and i+j==1: # 1,0
        v=3 
        
    return v


# Commented out because I haven't thought about how 2*d[01] etc affects this
# def tensor2voigt(A):
#     Avoigt=np.zeros((6,6))
#     for i in range(3):
#         for j in range(3):
#             iv = ind2voigt(i,j)
#             for k in range(3):
#                 for l in range(3):
#                     jv = ind2voigt(k,l)
#                     print(iv)
#                     print(jv)
#                     Avoigt[iv,jv] = A[i,j,k,l]
    
#     return Avoigt


def voigt2tensor(Avoigt):
    # Correct for 2*d off diags
    Avoigt[:,3:]=2*Avoigt[:,3:]
    A=np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            iv = ind2voigt(i,j)
            for k in range(3):
                for l in range(3):
                    jv = ind2voigt(k,l)

                    A[i,j,k,l] = Avoigt[iv,jv]

    
    
    return A