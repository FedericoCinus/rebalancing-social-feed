from collections.abc import Iterable
import cvxpy as cp
from numbers import Number
import numpy as np
import sys
sys.path.append('../src')
from utils import assert_results

# Optimal
def cvx_optimizer(A_eq: np.matrix, s: np.matrix,
                  max_iters: int=None, eps: float = None, solver: str='SCS', 
                  initialization: str = "uniform", verbosity=0, **kwargs) -> ():
    """Returns (A_opt, None, stats) using the chosen solver with the CVX routine
    """
    # variable
    verbose = verbosity >= 3
    n = len(s)
    I = np.identity(n)
    A_inner = cp.Variable((n, n), symmetric=True, integer=False)
    if max_iters is not None or eps is not None:
        print(f"Running scs with max_iters={max_iters} and eps={eps}")
    
    # OBJECTIVE 
    objective = cp.Minimize(cp.matrix_frac(s, I + I - A_inner)) 
    
    # CONSTRAINTS
    # 1. stochasticity
    constraints = [(A_inner[i, :] @ np.ones(n) == 1.) for i in range(n)]
    constraints += [(A_inner[:, i] @ np.ones(n) == 1.) for i in range(n)]
    # 2. no new edges
    constraints += [A_inner[i, j] == 0. for i in range(n) for j in range(i + 1, n) if A_eq[i, j] == 0.]
    # 3. keep signs
    constraints += [A_inner[i, j] >= 0. for i in range(n) for j in range(i, n)]
    
    prob = cp.Problem(objective, constraints)
    if solver is None:
        prob.solve(verbose=verbose)
    else:
        if eps is None and max_iters is None:
            prob.solve(solver=solver, verbose = verbose)
        elif eps is None:
            prob.solve(solver=solver, verbose = verbose, max_iters = max_iters)
        elif max_iters is None:
            prob.solve(solver=solver, verbose = verbose, eps=eps)
        else:
            prob.solve(solver=solver, verbose = verbose, max_iters = max_iters, eps = eps)
    A_opt = np.matrix(A_inner.value)
    ε = 1E-4
    A_opt[np.absolute(A_opt) < ε] = 0

    stats = {'solve_time': prob.solver_stats.solve_time, 'setup_time': prob.solver_stats.setup_time}
    return A_opt, [], stats