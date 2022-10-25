"""
Definitions of forms for energy and transport equations.
"""

__author__ = "Marie E. Rognes (meg@simula.no) and Lyudmyla Vynnytska (lyudav@simula.no)"
__copyright__ = "Copyright (C) 2011 %s" % __author__

from dolfin import *

def advection(phi, v, u, n, theta=1.0):
    """
    This form is called a_A in text

    u is the velocity
    phi is the first argument (Trial/Function)
    v is the second argument  (Test)

    n is the cell normal
    """

    un = abs(dot(u('+'), n('+')))

    a_cell = - theta*dot(u*phi, grad(v))*dx

    # Check this versus jump(v, n)
    jump_v = v('+')*n('+') + v('-')*n('-')
    jump_phi = phi('+')*n('+') + phi('-')*n('-')

    a_int = theta*(dot(u('+'), jump_v)*avg(phi) + 0.5*un*dot(jump_phi, jump_v))*dS
    a_ext = theta*dot(v, dot(u, n)*phi)*ds

    a = a_cell + a_int + a_ext
    return a

def diffusion(phi, v, kappa, alpha, n, h, theta=1.0):
    """
    This form is called a_D in text

    phi is the first argument (Trial/Function)
    v is the second argument  (Test)

    kappa is the diffusion _constant_

    alpha is the constant associated with the DG scheme
    n is the cell normal
    h is the cell size
    """

    # Contribution from the cells
    a_cell = theta*kappa*dot(grad(phi), grad(v))*dx

    # Contribution from the interior facets
    a_int0 = theta*kappa('+')*alpha('+')/h('+')*dot(jump(v, n), jump(phi, n))*dS
    a_int1 = - theta*kappa('+')*dot(avg(grad(v)), jump(phi, n))*dS
    a_int2 = - theta*kappa('+')*dot(jump(v, n), avg(grad(phi)))*dS
    a_int = a_int0 + a_int1 + a_int2

    # Contribution from the exterior facets?

    a = a_cell + a_int
    return a

def source(v, qs):
    """
    Corresponds to source term in text

    v is Test function

    qs is source function
    """

    # Contribution from the cells
    a = -v*qs*dx

    return a


def backward_euler(u, u_, dt):
    return (u - u_)/dt
