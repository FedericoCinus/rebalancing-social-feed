from collections.abc import Iterable
import ctypes
import networkx as nx
import numpy as np
import os
from pathlib import Path
import scipy
from sklearn.preprocessing import normalize
                        
#####################################################################       
###########################    PROJECTION    ########################
#####################################################################       

def cast_to_doubly_stochastic(M: scipy.sparse.csr, T: int = 1000, 
                              ε: float = 1e-3, verbose : bool = True):
    for i in range(T):
        M = normalize(M, axis=1, norm='l1')
        M = normalize(M, axis=0, norm='l1')
        if verbose:
            print(f"   casting adj to doubly stochastic {i}/{T}", end="\r")
        if np.all(np.absolute(M.sum(axis=1) - 1) < ε):
            if verbose:
                print(f"   Finished casting adj to doubly stochastic at iteration {i}/{T}")
            return M
    assert np.all(A.diagonal() > 0)
    print("Increase T")
    
    
    
#####################################################################       
###########################    TESTING    ###########################
#####################################################################

def is_symmetric(a, tol=1e-6):
    return np.all(np.abs(a - a.T) < tol)

def check_results(L_opt, s, A=None, b=None, ε=0.001):
    I = np.identity(L_opt.shape[0])
    __full_print = True
    if b is None:
        b = np.diagonal(L_opt)
        __full_print = False
    check_laplacian, check_budget = (np.all(np.absolute(L_opt.sum(axis=1))<ε), np.all((np.diagonal(L_opt)-b)<=ε))
    text = f'Symmetry: {is_symmetric(L_opt)},  Is laplacian: {check_laplacian}'
    if not check_laplacian:
        print(f"max/min sum row: {np.max(np.absolute(L_opt.sum(axis=1))):.3f}/{np.min(np.absolute(L_opt.sum(axis=1))):.3f}")
    text += f', Budget: {check_budget}' if __full_print else '' 
    print(text)
    print(f"L has negative entries: {np.all([L_opt[i, j] <= 0 for i in range(len(s)) for j in range(len(s)) if i!=j])}")
    if A is not None:
        print(f'No new edges: {np.all([L_opt[i, j] <= ε for i, j in np.argwhere((A + I) == 0)])}')
    print(f'objective value is: {(s.T@np.linalg.inv(np.identity(L_opt.shape[0])+ L_opt)@s).item():.3f}')
    print(f'Trace is: {np.trace(L_opt):.3f}')
 
    
    
def assert_results(L_opt, b, A=None, ε=0.001, use_assert=True):
    n = L_opt.shape[0]
    check_laplacian, check_budget = (np.all(np.absolute(L_opt.sum(axis=1))<ε), np.absolute(np.trace(L_opt)-b)<=ε)
    check_edge_sign = np.all([L_opt[i, j] <= 0 for i in range(n) for j in range(n) if i!=j])
    if use_assert:
        assert check_laplacian, f"check_laplacian:   {np.max(np.absolute(L_opt.sum(axis=1))):.5f}"
        assert check_budget, f"check_budget:  {np.max(np.absolute(np.trace(L_opt)-b)):.5f}"
        assert check_edge_sign, "check_edge_sign"
    else:
        if not check_laplacian:
            print(f"check_laplacian:   {np.max(np.absolute(L_opt.sum(axis=1))):.5f}")
        if not check_budget:
            print(f"check_budget:  {np.max(np.absolute(np.trace(L_opt)-b)):.5f}")
        if not check_edge_sign:
            print("check_edge_sign")
    
#####################################################################       
###########################    MULTIPROCESSING    ###################
#####################################################################


def set_mkl_threads(n):
    """Sets all the env variables to limit numb of threads
    """
    try:
        import mkl
        mkl.set_num_threads(n)
        return 0
    except:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:  # pylint: disable=bare-except
            pass
    v = f"{n}"
    os.environ["OMP_NUM_THREADS"] = v  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = v  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = v  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = v  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = v  # export NUMEXPR_NUM_THREADS=6