from fenics import *
import numpy as np




def Init(domain,element_pair=0,fabricdegree=0):
    domain.nD = domain.mesh.geometry().dim()
    if element_pair == 0:
        # - Taylor-Hood
        p = 2
        V_el = VectorElement("Lagrange", domain.mesh.ufl_cell(),p)
        Q_el = FiniteElement("Lagrange", domain.mesh.ufl_cell(),p-1)
        TH = V_el*Q_el

    elif element_pair == 1:
        V_el = VectorElement("Lagrange", domain.mesh.ufl_cell(),3)
        Q_el = FiniteElement("Lagrange", domain.mesh.ufl_cell(), 1)
        TH = V_el*Q_el

    elif element_pair == 2:
        # - Crouzeix-Raviart
        V_el = VectorElement("Crouzeix-Raviart", domain.mesh.ufl_cell(),1)
        Q_el = FiniteElement("DG", domain.mesh.ufl_cell(), 0)
        TH = V_el*Q_el

    elif element_pair == 3:
        # - CD
        V_el = VectorElement("Lagrange", domain.mesh.ufl_cell(),3)
        Q_el = FiniteElement("DG", domain.mesh.ufl_cell(), 1)
        TH = V_el*Q_el

    elif element_pair == 4:
        # - MINI
        p = 1
        P1_el = VectorElement("Lagrange", domain.mesh.ufl_cell(), p)
        B_el = VectorElement("Bubble", domain.mesh.ufl_cell(), p+3)
        V_el = P1_el+B_el
        Q_el = FiniteElement("Lagrange", domain.mesh.ufl_cell(), p)
        TH = V_el*Q_el

    domain.Q = FunctionSpace(domain.mesh,Q_el, constrained_domain=domain.pbc)
    domain.V = FunctionSpace(domain.mesh, V_el, constrained_domain=domain.pbc)
    domain.W = FunctionSpace(domain.mesh, TH, constrained_domain=domain.pbc)

    domain.F = FunctionSpace(domain.mesh,'DG',fabricdegree)
    domain.F2 = VectorFunctionSpace(domain.mesh,'DG',fabricdegree)
    domain.F3 = VectorFunctionSpace(domain.mesh,'DG',fabricdegree, dim=3)
    domain.T2 = TensorFunctionSpace(domain.mesh, "DG", fabricdegree, shape=(domain.nD, domain.nD))
    domain.T3D = TensorFunctionSpace(domain.mesh, "DG", fabricdegree, shape=(3,3))
    domain.T4 = TensorFunctionSpace(domain.mesh, "DG", fabricdegree, shape=(domain.nD,domain.nD,domain.nD,domain.nD))
    domain.T3333 = TensorFunctionSpace(domain.mesh, "DG", fabricdegree, shape=(3,3,3,3))
    domain.T66 = TensorFunctionSpace(domain.mesh, "DG", fabricdegree, shape=(6,6))
    domain.viscpoints = domain.F.tabulate_dof_coordinates().shape[0]

    domain.F_dg1 = FunctionSpace(domain.mesh,'DG',1)
