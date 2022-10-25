import numpy as np 

def correction(fnp,sh):
    # Correct fabric based on constraints on a2 and formula for a2 = f(fnp)

    # Constraint: total fabric conserved
    fnp[:,0] = 1.0

    a2 = np.moveaxis(sh.a2(fnp.T),-1,0)
    

    # Constraint: 0<A22<1
    
    f20max = (2./3.)/0.29814239699997195952
    f20min = (-1./3.)/0.29814239699997195952

    f20 = fnp[:,sh.idx(2,0)]

    f20[f20.real>f20max] = f20[f20.real>f20max]*f20max/f20[f20.real>f20max].real
    f20[f20.real<f20min] = f20[f20.real<f20min]*f20max/f20[f20.real<f20min].real
    fnp[:,sh.idx(2,0)] = f20

    # Constrain: 0<A00<1
    # a00 = 1./3. - 0.14907119849998597976*f[sh.idx(2,0)]+ 2*0.18257418583505537115*f[:,sh.idx(2,2)]

    # Let a00=1
    f22max = (2./3 + 0.14907119849998597976*f20)/(2*0.18257418583505537115)

    # a00=0
    f22min = (-1./3+ 0.14907119849998597976*f20)/(2*0.18257418583505537115)

    f22 = fnp[:,sh.idx(2,0)]
    
    f22[f22.real>f22max] = f22[f22.real>f22max]*f22max/f22[f22.real>f22max].real
    f22[f22.real<f22min] = f22[f22.real<f22min]*f22max/f22[f22.real<f22min].real
   
    fnp[:,sh.idx(2,2)] = f22

    return fnp

def simplecorrector(fnp,k=2.0):
    fnp[:,0] = 1.0

    fnp[np.abs(fnp)>k] = fnp[np.abs(fnp)>k]*k/np.abs(fnp[np.abs(fnp)>k])
    return fnp