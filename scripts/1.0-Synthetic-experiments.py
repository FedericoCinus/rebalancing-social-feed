"""Experiments script for algorithms implemented in Python 3
"""
# Imports
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from pathlib import Path
import pickle
import json
import scipy
from scipy.sparse import identity
import seaborn as sb
import sys
from time import time
import warnings
sys.path.append('../src')
from algorithms_approx3 import (objective_f, GD_optimizer_dir, opposite_view_heuristic_dir, 
                                popularity_heuristic_dir, neutral_view_heuristic_dir)
from generative_graph_models import assert_adj_stochasticity

from generative_opinions_models import standardize
from algorithms_optimal import cvx_optimizer
from utils import set_mkl_threads, assert_results
from config import create_json_file
set_mkl_threads(16)
warnings.simplefilter(action='ignore', category=FutureWarning)


# parsing task
create_json_file()
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=int, choices=[0, 1, 2, 3, 4], help = '0 small, 1 runtime, 2 modularity, 3 large, 4 learning curves')
parser.add_argument("-i", "--max_iters", type=int, help = 'Maximum number of iterations, if early_stop is not True.')
parser.add_argument("-d", "--directed", type=int, choices=[0, 1], help = '0 undirected, 1 directed')
parser.add_argument("-c", "--constraint", type=int, choices=[0, 1], help = '0 trace, 1 degree.', default=1)
parser.add_argument("-u", "--uniform_init", type=int, choices=[0, 1], help = 'If 1 uniform init of graph weights.', default=0)
parser.add_argument("-s", "--early_stopping", type=int, choices=[0, 1], help = 'If 1, it stops at convergence.', default=1)
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3, 4], help = 'Verbosity levels.', default=0)
parser.add_argument("-w", "--test", type=int, choices=[0, 1], help = '0 is not a test, 1 is a test.',  default=0)
parser.add_argument("-f", "--folder", type=str, help = 'folder name.',  default=None)

args = vars(parser.parse_args())

max_n = 400 ## Max numb of nodes for optimal algorithm
ε = 1e-6
   
# Opening parameters from json file
with open(".json_parameters.json", 'r') as file:
    json_file = json.loads(json.load(file))
if args['task'] == 0: 
    config = json_file['task_0']['optimization']
    task = 'approximation_small'
elif args['task'] == 1: 
    config = json_file['task_1']['optimization']
    task = 'runtime'
elif args['task'] == 2: 
    config = json_file['task_2']['optimization']
    task = 'modularity'
elif args['task'] == 3:
    config =  json_file['task_3']['optimization']
    task = 'approximation_large'
elif args['task'] == 4:
    config =  json_file['task_4']['optimization']
    task = 'approximation_curves'

# Base folder path
constraint = "trace" if args['constraint'] == 0 else "degree"
directed = args['directed']==1
direction = "directed" if directed else "undirected"
folder_path = Path(config['folder']) / Path(direction) / Path(f"{task}_task__{constraint}_constraint")

# Testing parameters
is_test = args['test'] == 1
if is_test and args['max_iters'] == -777:
    args['max_iters'] = 1

################################################################################################


# Folder Names
with open(folder_path / Path("folder_names.txt"), "r") as file: # hashed names
    folders = [line.split("| ")[1][:-1] for line in file]
with open(folder_path / Path("folder_names.txt"), "r") as file: # real names
    folders_real_names = [line.split("| ")[0] for line in file]
folders = folders[0:1] if is_test else folders
folders = [args['folder'],] if args['folder'] is not None else folders

# Simulations
for i, simulation_folder in enumerate(folders):
    print(f"Starting experiments on folder {folders_real_names[i]}, \n{simulation_folder}")

    # 0. Load data
    print("Loading files .. ")
    s = np.load(folder_path / simulation_folder / Path("internal_opinions_sample00_python.npy"))
    path = folder_path / simulation_folder / Path("A_eq_sparse_sample00_python.npz")
    A_eq = scipy.sparse.load_npz(path)
    η = np.linalg.norm(s)
    print(f".. all files loaded. \n Graph has {len(s)} nodes, {len(A_eq.data)} variables, {η:.1f} norm, {np.quantile(np.array(A_eq.sum(axis=1)).flatten(), .5):.3f} avg degree ")
    
    
    # 0. Reading previous experiments
    save_path = Path(config['folder']) / Path("test") if is_test else folder_path / Path(simulation_folder)
    experiments_path = save_path / Path(f"experiments_data_method.pkl")
    
    if not os.path.exists(experiments_path):
        method_name2data = {}
    else:
        with open(experiments_path, "rb") as file:
            method_name2data = pickle.load(file)
        print(f"    existing experiments: {method_name2data.keys()}")   
    # --------------------------------  
    
    # 1. Compute
    print("Starting optimizations .. \n")
    # retrieving existing experiments
    existing_methods = [x['method_config']['name'] for _, x in method_name2data.items()]
    existing_routines = []
    existing_budgets = []
    for _, x in method_name2data.items():
        if x['method_config']['name'] == "GD_optimizer":
            existing_routines.append(x['method_config']['params']['routine'])
            existing_budgets.append(x['method_config']['params']['grad_params']['budget'])
            
    for j, method_config in enumerate(config['methods']):
        condition = method_config['name'] not in existing_methods # 1. method not in previous experiments
        if method_config['name'] == "GD_optimizer": # 2. or gradient descent routine not in previous experiments
            condition |= method_config['params']['routine'] not in existing_routines
            condition |= method_config['params']['grad_params']['budget'] not in existing_budgets
        if condition:
            # Parsing method's parameters
            method_config['params']['early_stopping'] = args['early_stopping']
            method_config['params']['verbosity'] = args['verbosity']
            meth = method_config['params']['routine'] if 'routine' in method_config['params'] else ''
            print("   Using method", method_config['name'], ':', meth)


            time0 = time()
            ## Optimal
            if method_config['name'] == 'SCS' and not directed:
                A, objectives, stats=cvx_optimizer(A_eq.todense(), s, **method_config['params']) if len(s)<=max_n else (0,[0],0)
            ## Gradient Descent
            elif method_config['name'] == 'GD_optimizer':
                A, objectives, stats = GD_optimizer_dir(A_eq, s, args['max_iters'], **method_config['params'])
                assert_adj_stochasticity(A, True)
            ## Baselines
            elif method_config['name'] == "opposite_view_heuristic":
                A, objectives, stats = opposite_view_heuristic_dir(A_eq, s, verbosity=method_config['params']['verbosity'])
            elif method_config['name'] == "popularity_heuristic":
                A, objectives, stats = popularity_heuristic_dir(A_eq, s, verbosity=method_config['params']['verbosity'])
            elif method_config['name'] == "neutral_view_heuristic":
                A, objectives, stats = neutral_view_heuristic_dir(A_eq, s, verbosity=method_config['params']['verbosity'])
            Δt = time() - time0
            
            
            # First objective
            initial_PD_index = objective_f(s, A_eq)

            # Natural objective (using input graph)
            I = scipy.sparse.identity(A_eq.shape[0])
            z, _ = scipy.sparse.linalg.bicg(2*I - A_eq, s, tol=ε)
            z = z.reshape((len(s), 1))
            obj_nat = objective_f(z, A_eq)
           
            # Optimized objective
            z, _ = scipy.sparse.linalg.bicg(2*I - A, s, tol=ε)
            z = z.reshape((len(s), 1))
            obj = objective_f(z, A)
            objectives.append(obj)
            
            print((f"   --> Method {meth} obtained obj%={(1-obj/obj_nat)*100:.4f}% "
                   f"  total reduc={(1-obj/initial_PD_index)*100:.4f}% in {Δt:.3f}sec"))
            print(f"     for dataset={folders_real_names[i]}")
            print(f"     code={simulation_folder} .\n")


            # 2. Save Results
            method_name2data[j] = {"method_config": method_config, "A": A, 
                                   "objectives": objectives,  
                                   "runtime": Δt, "grad_stats": stats}
            if not is_test:
                with open(experiments_path, "wb") as file:
                    pickle.dump(method_name2data, file)
    print("Ending optimizations.\n")