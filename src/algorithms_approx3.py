import copy
import numpy as np
import sys
sys.path.append('../src')
from utils import check_results
import scipy
from scipy.sparse import identity
from sklearn.preprocessing import normalize
import time
from utils import cast_to_doubly_stochastic


def popularity_heuristic_dir(A, _s: np.matrix, verbosity: int = 0):
    Is, Js = np.nonzero(A)
    A_pop_heur = A.copy()
    deg = np.array(A.sum(axis=0)).flatten()
    max_deg = np.max(deg)
    A_pop_heur.data = (deg[Js]).flatten()
    
    # make row-stochastic
    A_pop_heur = normalize(A_pop_heur, axis=1, norm='l1')

    return A_pop_heur, [], None


def opposite_view_heuristic_dir(A, s: np.matrix, verbosity: int = 0):
    Is, Js = np.nonzero(A)
    A_oppo_heur = A.copy()
    A_oppo_heur.data = np.array(s[Is]-s[Js]).flatten()**2
   
    # make row-stochastic
    A_oppo_heur = normalize(A_oppo_heur, axis=1, norm='l1')
    
    return A_oppo_heur, [], None


def neutral_view_heuristic_dir(A, s: np.matrix, verbosity: int = 0):
    ε = 1e-3
    Is, Js = np.nonzero(A)
    A_neutral_heur = A.copy()
    A_neutral_heur.data = np.array(1/(np.absolute(s[Js])+ε)).flatten()**2
   
    # make row-stochastic
    A_neutral_heur = normalize(A_neutral_heur, axis=1, norm='l1')
    
    return A_neutral_heur, [], None


def objective_f(z, A, sep=False):
    I = identity(len(z))
    if sep: 
        return ((0.5*z.T @ (I + scipy.sparse.diags(np.array(A.sum(axis=0)).flatten()) - (A+A.T)) @ z + z.T @ z).item(),
                (z.T @ z).item())
    return (0.5*z.T @ (I + scipy.sparse.diags(np.array(A.sum(axis=0)).flatten()) - (A+A.T)) @ z + z.T @ z).item()

def gradient_sparse(A, s, Is, Js, ε=1e-6):
    """Returns gradient data array"""
    I = scipy.sparse.identity(len(s))
    
    z_1, _ = scipy.sparse.linalg.bicg(2*I-A.T, s, tol=ε)
    z_1 = z_1.reshape((len(s), 1))
    z_2, _ = scipy.sparse.linalg.bicg(2*I-A, s, tol=ε)
    z_2 = z_2.reshape((len(s), 1))
    grad = np.multiply(z_1[Is], z_2[Js]).flatten()
    
    z_3, _ =  scipy.sparse.linalg.bicg(2*I-A.T, (scipy.sparse.diags(np.array(A.sum(axis=0)).flatten()) - I) @ z_2, tol=ε)
    z_3 = z_3.reshape((len(s), 1))
    grad = grad + np.multiply(z_3[Is], z_2[Js]).flatten()
    grad = grad + .5 * (z_2**2)[Js].flatten()
    return grad

def GD_optimizer_dir(A_initial, s: np.matrix, 
                     max_iters: int, early_stopping: bool = False,
                     routine: str = None, grad_params: dict = None, lr_params: dict = None, 
                     verbosity: int = 0, ε = 1e-6):
    """Returns the A optimized matrix and the objective values
       using input matrix A, inner opinion vector s, the percentage of budget to use,
       routine = "ADAM" / "NAG" / "simple"
    """
    setup_time = time.time()
    stats = {'increments': [], 'solve_time': None, 'setup_time': None}
    
    # 1. Initialize variables: Laplacian L, I, from A_init
    n = len(s)
    I = identity(n)
    edges = A_initial.count_nonzero()
    
    A_inner = copy.deepcopy(A_initial)
    Is, Js = np.nonzero(A_inner)
    A_prev = copy.deepcopy(A_inner)
    

    #### Initialization objective
    z_eq, _ = scipy.sparse.linalg.bicg(2*I-A_inner, s, tol=ε)
    z_eq = z_eq.reshape((len(s), 1))
    obj = objective_f(z_eq, A_inner)
    initial_PD_index = objective_f(s, A_inner)
    feasible_objectives = [obj]
    A_increment = np.inf
    
    #### Learning rate
    η = lr_params["lr_coeff"]
    budget = grad_params['budget'] if 'budget' in grad_params else 1.
    η_t = η
    
    # 2. Optimization loop
    time0 = time.time()
    stats['setup_time'] = time.time() - setup_time
    T = 1
    
    print(f"  Initializing LcGD {routine} solver with η={η}")
    
    for t in range(1, int(max_iters)+1):
        # 2.1 grad
        grad = gradient_sparse(A_inner, s, Is, Js, ε=1e-6)        
        
        # grad step
        A_prev = copy.deepcopy(A_inner)
        if verbosity >= 1:
            if t%T == 0 or t == 1:
                Δ_obj = (np.min(feasible_objectives[-4:-3]) - np.min(feasible_objectives[-2:-1])) if t > 10 else 0
                print((f"  t={t}, obj={obj/feasible_objectives[0]*100:.3f}, total_red={obj/initial_PD_index*100:.3f},"
                       f" Δ_obj={Δ_obj:.5f}, thr={edges*1e-6:.6f}"))
              
        assert len(A_inner.data) == len(grad), ("1", len(A_inner.data), len(grad), η_t)
        
        ## GRADIENT SCHEMES
        if routine == "ADAM":
            β1, β2, ε = (0.9, 0.999, 1e-08)
            μ = np.multiply(β1, μ) + np.multiply((1-β1), grad) if t>1 else np.multiply((1-β1), grad)
            v = np.multiply(β2, v) + np.multiply((1-β2), np.power(grad, 2)) if t>1 else np.multiply((1-β2), np.power(grad, 2))
            μ_hat = np.multiply(1 / (1 - np.power(β1, t)), μ) 
            v_hat = np.multiply(1 / (1 - np.power(β2, t)), v)
            #print(np.isnan(η_t/(np.sqrt(v_hat) + ε)), v_hat[-2], v[-2])
            A_inner.data -= np.multiply(η_t/(np.sqrt(v_hat) + ε), μ_hat)
            
        elif routine == "NAG":
            if t == 1:
                λ_prev, λ_curr, γ, y_prev = (0, 1, 1, A_inner.data.copy())
            y_curr = A_inner.data - η_t * grad
            A_inner.data = (1 - γ) * y_curr + γ * y_prev 
            y_prev = y_curr
            
            λ_curr = (1 + np.sqrt(1 + 4 * λ_prev**2)) / 2
            γ = (1 - λ_prev) / λ_curr
        else:
            A_inner.data -= np.multiply(η_t, grad)

        ### Projection
        A_inner[A_inner<0] = 0 # remove negative edges
        A_inner.data += 1/A_inner.shape[0] # ensuring no completely isolated nodes
        A_inner = normalize(A_inner, axis=1, norm='l1')
        
        ### Budget update
        A_inner.data = (1 - budget) * A_initial.data + budget * A_inner.data
        
        ### Objective value
        z_eq, _ = scipy.sparse.linalg.bicg(2*I-A_inner, s, tol=ε)
        z_eq = z_eq.reshape((len(s), 1))
        obj = objective_f(z_eq, A_inner)
        feasible_objectives.append(obj)
        
        
        # 5.1 Early stopping
        A_increment = scipy.sparse.linalg.norm(A_inner-A_prev)
        stats['increments'].append(A_increment) 
        
        if t > 25 and early_stopping:
            scale = 1e-6
            Δ_obj = (np.min(feasible_objectives[-4:-3]) - np.min(feasible_objectives[-2:-1]))
            
            
            # A. Time limit
            if (time.time()-time0)/(60*60)>=5.:
                print(f"Time limit: stopping at t={t}, A increment", A_increment)
                stats['solve_time'] = time.time() - time0
                return A_inner, feasible_objectives, stats
            
            
            
            # B. Tolerance (Num edges x scale factor)
            if (np.sign(Δ_obj) == 1.) and Δ_obj < edges*scale:
                print(f"Convergence1: stopping at t={t},  Δ_obj={Δ_obj:.5f} thr={edges*scale:.6f}, η={η:.5f}")
                stats['solve_time'] = time.time() - time0
                return A_inner, feasible_objectives, z_eq 
            
            # C. Increment of loss
            elif (np.sign(Δ_obj) == -1.):
                print(f"Convergence2: stopping at t={t},  Δup={Δ_obj:.5f}")
                stats['solve_time'] = time.time() - time0
                return A_inner, feasible_objectives, z_eq 
                               
        # D. Max iterations reached
        if t == max_iters and early_stopping: 
            Δ_obj = ( np.mean(feasible_objectives[-4:-3]) - np.mean(feasible_objectives[-2:-1]) )
            print(f"early_stopping={early_stopping}, you reached {t}/{max_iters}, A increment {A_increment}, Δ_obj={Δ_obj:.8f}, precision={A_inner.shape[0]*1e-5:.8f}. Please increase max_iters!")
        
    
    stats['solve_time'] = time.time() - time0
    return A_inner, feasible_objectives, z_eq