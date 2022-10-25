from fenics import*
import numpy as np


def Fenics2Numpy(f):
    
    if f.ufl_shape:
        q = f.vector().get_local()
        shape = list(f.ufl_shape)
        shape.append(q.size//np.prod(shape))
        q=q.reshape(tuple(shape))
        q = np.moveaxis(q,-1,0)
    else:
        q = f.vector().get_local()
    
    return q
        

def FenicsMatrix2Numpy(T,F):
    shape = list(T.ufl_shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            q = project(T[i,j],F).vector().get_local()
            if i==0 and j==0:
                Tnp = np.zeros((q.shape[0],shape[0],shape[1]))
            Tnp[:,i,j] = q

    return Tnp

def FenicsTensor2Numpy(T,F):
    shape = list(T.ufl_shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    q = project(T[i,j],F).vector().get_local()
                    if i==0 and j==0 and k==0 and l==0:
                        Tnp = np.zeros((q.shape[0],shape[0],shape[1]))
                    Tnp[:,i,j] = q

    return Tnp


# def NumpyTensor2Fenics(Tnp,F):
#     shape = Tnp.shape
#     componentList = []
#     for i in range(shape[1]):
#         componentList += [[],]
#         for j in range(shape[2]):
#             componentList[i] += [[],]
#             for k in range(shape[3]):
#                 componentList[i][j] += [[],]
#                 for l in range(shape[4]):
#                     componentList[i,j,k] += [Function(F)
#                     q.vector()[:] = Tnp[:]
#                     q = project(T[i,j],F).vector().get_local()
#                     if i==0 and j==0 and k==0 and l==0:
#                         Tnp = np.zeros((q.shape[0],shape[0],shape[1]))
#                     Tnp[:,i,j] = q

#     return Tnp


            



def Numpy2Fenics(f,fnp):
    f.vector()[:]=fnp.flatten()
    return f

def grad3d(u):
    if u.geometric_dimension()==2:
        ux,uy = split(u)
        gradu = as_matrix(((ux.dx(0), ux.dx(1), 0.), (uy.dx(0), uy.dx(1), 0), (0., 0., 0.)))
    else:
        gradu = grad(u)
        
    return gradu