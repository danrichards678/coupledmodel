from fenics import *
import numpy as np


def transposeMatrix(m):
    return list(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    m = fenics2list(m)
    determinant = getMatrixDeternminant(m)
    
    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return as_tensor(cofactors)

def fenics2list(m):
    mlist = []
    for i in range(m.ufl_shape[0]):
        mlist += [[],]
        for j in range(m.ufl_shape[1]):
            mlist[i] += [m[i,j],]

    return mlist


# def inv4x4(a):
#     deta = det(a)

#     A00 = a[1,1]*a[2,2]*a[3,3] + a[1,2]*a[2,3]*a[3,1] + a[1,3]*a[2,1]*a[3,2]\
#             - a[1,3]*a[2,2]*a[3,1] - a[1,2]*a[2,1]*a[3,3] - a[1,1]*a[2,3]*a[3,2]
    
#     A01 = -(a[0,1]*a[2,2]*a[3,3] + a[0,2]*a[2,3]*a[3,1] + a[0,3]*a[2,1]*a[3,2])\
#          + (a[0,3]*a[2,2]*a[3,1] + a[0,2]*a[2,1]*a[3,3] + a[0,1]*a[2,3]*a[3,2])

#     A02 = +(a[0,1]*a[1,2]*a[3,3] + a[0,2]*a[1,3]*a[3,1] + a[0,3]*a[1,1]*a[3,2])\
#          - (a[0,3]*a[1,2]*a[3,1] + a[0,2]*a[1,1]*a[3,3] + a[0,1]*a[1,3]*a[3,2])
    
#     A03 = -(a[0,1]*a[1,2]*a[2,3] + a[0,2]*a[1,3]*a[2,1] + a[0,3]*a[1,1]*a[2,2])\
#          + (a[0,3]*a[1,2]*a[2,1] + a[0,2]*a[1,1]*a[2,3] + a[0,1]*a[1,3]*a[2,2])
