import numpy as np
import voigtgolf as vg
import golfnew as golff90



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
        euler[0]=np.arctan2(ev[1,0],ev[0,0])
        euler[2]=0.
    
    return euler,ai


def etaIload(gamma,beta,n=1.):
    abcd = "{:.4f}".format(beta)[2:]
    e = "{:.2f}".format(gamma)[0]
    fg = "{:.2f}".format(gamma)[-2:]
    h = "{:.1f}".format(n)[0]
    i = "{:.1f}".format(n)[-1]

    path='./ViscosityFiles_ai/' +abcd + e + fg + h + i + '.Va'
    
    etaI = np.genfromtxt(path, delimiter = 14, skip_footer=1)
    return etaI
    



def Golf(a2,gamma,beta):
    mu = np.zeros((a2.shape[0],3,3,3,3))
    for i in range(a2.shape[0]):
        mu[i,...] = golfsingle(a2[i,:,:],gamma,beta)

    return mu


def golfsingle(a2,gamma,beta):
    euler,ai = EulerAngles(a2)
    etaI = etaIload(gamma,beta).flatten()

    eta36 = golff90.opilgge_ai_nl(ai,euler,etaI)

    #Transform to mu
    mu = vg.voigt2tensor(eta36)

    return mu

    
