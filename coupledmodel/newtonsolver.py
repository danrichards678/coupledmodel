from fenics import *
import numpy as np
# From https://github.com/pf4d/cslvr/blob/16d5459d9ee79a83751fae4f68e50f4693d5a423/cslvr/model.py

def linear_solve_params():
		"""
		Returns a set of linear solver parameters.
		"""
		#lparams  = {"linear_solver"            : "default"}
		#lparams  = {"linear_solver"            : "superlu_dist"}
		lparams  = {"linear_solver"            : "mumps"}
		#lparams  = {"linear_solver"            : "umfpack"}
		#lparams  = {"linear_solver"            : "cg",
		#            "preconditioner"           : "default"}
		return lparams

def nonlinear_solve_params():
    """
    Returns a set of linear and nonlinear solver parameters.
    """
    nparams = {'newton_solver' :
                {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-9,
                'relaxation_parameter'     : 1.0,
                'maximum_iterations'       : 25,
                'error_on_nonconvergence'  : False,
                }}
    return nparams


def home_rolled_newton_method(R, U, J, bcs, atol=1e-7, rtol=1e-10,
	                              relaxation_param=1.0, max_iter=25,
	                              method='mumps', preconditioner='default',
	                              cb_ftn=None, bp_Jac=None, bp_R=None):
    """
    Appy Newton's method.
    :param R:                residual of system
    :param U:                unknown to determine
    :param J:                Jacobian
    :param bcs:              set of Dirichlet boundary conditions
    :param atol:             absolute stopping tolerance
    :param rtol:             relative stopping tolerance
    :param relaxation_param: ratio of down-gradient step to take each iteration.
    :param max_iter:         maximum number of iterations to perform
    :param method:           linear solution method
    :param preconditioner:   preconditioning method to use with ``Krylov``
                                solver
    :param cb_ftn:           at the end of each iteration, this is called
    """
    converged  = False
    lmbda      = relaxation_param   # relaxation parameter
    nIter      = 0                  # number of iterations

    ## Set PETSc solve type (conjugate gradient) and preconditioner
    ## (algebraic multigrid)
    #PETScOptions.set("ksp_type", "cg")
    #PETScOptions.set("pc_type", "gamg")
    #
    ## Since we have a singular problem, use SVD solver on the multigrid
    ## 'coarse grid'
    #PETScOptions.set("mg_coarse_ksp_type", "preonly")
    #PETScOptions.set("mg_coarse_pc_type", "svd")
    #
    ## Set the solver tolerance
    #PETScOptions.set("ksp_rtol", 1.0e-8)
    #
    ## Print PETSc solver configuration
    #PETScOptions.set("ksp_view")
    #PETScOptions.set("ksp_monitor")

    #PETScOptions().set('ksp_type',                      method)
    #PETScOptions().set('mat_type',                      'matfree')
    #PETScOptions().set('pc_type',                       preconditioner)
    #PETScOptions().set('pc_factor_mat_solver_package',  'mumps')
    #PETScOptions().set('pc_fieldsplit_schur_fact_type', 'diag')
    #PETScOptions().set('pc_fieldsplit_type',            'schur')
    #PETScOptions().set('fieldsplit_0_ksp_type',         'preonly')
    #PETScOptions().set('fieldsplit_0_pc_type',          'python')
    #PETScOptions().set('fieldsplit_1_ksp_type',         'preonly')
    #PETScOptions().set('fieldsplit_1_pc_type',          'python')
    #PETScOptions().set('fieldsplit_1_Mp_ksp_type',      'preonly')
    #PETScOptions().set('fieldsplit_1_Mp_pc_type',       'ilu')
    #PETScOptions().set('assembled_pc_type',             'hypre')

    # need to homogenize the boundary, as the residual is always zero over
    # essential boundaries :
    bcs_u = []
    for bc in bcs:
        bc = DirichletBC(bc)
        bc.homogenize()
        bcs_u.append(bc)

    # the direction of decent :
    d = Function(U.function_space())

    while not converged and nIter < max_iter:

        # assemble system :
        A, b    = assemble_system(J, -R, bcs_u)

        ## Create Krylov solver and AMG preconditioner
        #solver  = PETScKrylovSolver(method)#, preconditioner)

        ## Assemble preconditioner system
        #P, btmp = assemble_system(bp_Jac, -bp_R)

        ### Associate operator (A) and preconditioner matrix (P)
        ##solver.set_operators(A, P)
        #solver.set_operator(A)

        ## Set PETSc options on the solver
        #solver.set_from_options()

        ## determine step direction :
        #solver.solve(d.vector(), b, annotate=False)

        # determine step direction :
        solve(A, d.vector(), b, method, preconditioner)

        # calculate residual :
        residual  = b.norm('l2')

        # set initial residual :
        if nIter == 0:
            residual_0 = residual

        # the relative residual :
        rel_res = residual/residual_0

        # check for convergence :
        converged = residual < atol or rel_res < rtol

        # move U down the gradient :
        U.vector()[:] += lmbda*d.vector()

        # increment counter :
        nIter += 1

        # print info to screen :
        if MPI.rank == 0:
            string = "Newton iteration %d: r (abs) = %.3e (tol = %.3e) " \
                        +"r (rel) = %.3e (tol = %.3e)"
            print(string % (nIter, residual, atol, rel_res, rtol))

        # # call the callback function, if desired :
        # if cb_ftn is not None:
        # 	s    = "::: calling home-rolled Newton method callback :::"
        # 	print_text(s, cls=self.this)
        # 	cb_ftn()
    return U
