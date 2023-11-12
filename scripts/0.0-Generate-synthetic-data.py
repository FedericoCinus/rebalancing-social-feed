import argparse
from collections.abc import Iterable
import hashlib
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
from pathlib import Path
import pickle
import yaml
import json
import scipy
from scipy.sparse import identity
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append('../src')
from algorithms_optimal import cvx_optimizer
from generative_graph_models import define_graph_instance, assert_adj_stochasticity
from generative_opinions_models import define_set_opinions, standardize, normalize
from config import create_json_file
from utils import is_symmetric, set_mkl_threads, check_results
set_mkl_threads(4)

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=int, choices=[0, 1, 2, 3, 4], help = '0 small, 1 runtime, 2 modularity, 3 large, 4 learning curves')
parser.add_argument("-d", "--directed", type=int, choices=[0, 1], help = '0 undirected, 1 directed')
parser.add_argument("-c", "--constraint", type=int, choices=[0, 1], help = '0 trace, 1 degree', default=1)
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1], help = 'Verbosity levels.', default=0)
args = vars(parser.parse_args())


# Opening parameters from json file
create_json_file()
with open(".json_parameters.json", 'r') as file:
    json_file = json.loads(json.load(file))
if args['task'] == 0:
    parameters = json_file['task_0']['generation'] 
    task = 'approximation_small'
elif args['task'] == 1:
    parameters =  json_file['task_1']['generation']
    task = 'runtime'
elif args['task'] == 2:
    parameters =  json_file['task_2']['generation']
    task = 'modularity'
elif args['task'] == 3:
    parameters =  json_file['task_3']['generation']
    task = 'approximation_large'
elif args['task'] == 4:
    parameters =  json_file['task_4']['generation']
    task = 'approximation_curves'
    
constraint = "trace" if args['constraint'] == 0 else "degree"
directed = args['directed']==1

STAT = parameters['statistics']
NUMB_NODES = parameters['numb_nodes']
MODELS = parameters['models']
POLARIZATIONS = parameters['polarizations']
BUDGETS = parameters['budgets']

write_files = True
direction = "directed" if directed else "undirected"
folder_path = Path(parameters['folder']) / Path(direction) / Path(f"{task}_task__{constraint}_constraint")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)   #

#########################################################################################################
np.random.seed(0)
index = 0
for (sample_numb, numb_nodes, model, budget_perc, initial_pol) in product(list(range(STAT)), 
                                                                          NUMB_NODES, MODELS, 
                                                                          BUDGETS, POLARIZATIONS):
    net_model, e_param_name = (model["network_model"], model["network_param_name"])
    for e_param_value in model["network_param_values"]:
        for opinion_model in model["opinion_models"]:
            
            # 0. Graph
            print("Reading graph ..")
            if write_files:
                A_eq, G = define_graph_instance(net_model, {'n': numb_nodes, e_param_name: e_param_value},
                                                directed=directed, verbose=args['verbosity'] >= 1)
                assert_adj_stochasticity(A_eq, directed)
                numb_nodes = G.number_of_nodes()
            I = identity(numb_nodes)
            #############################################################################################
            
            # 1. Folder name and file trace
            folder_name = (f"{net_model}_net_model__{numb_nodes}_nodes__"
                           f"{e_param_value}_{e_param_name}__{int(budget_perc*100)}_percbudget__"
                           f"{opinion_model}_opinion_model__{initial_pol}_initial_pol__"
                           f"{constraint}_constraint__{sample_numb}")
            generative_parameters = {"net_model": net_model, "numb_nodes": numb_nodes,
                                     f"{e_param_name}": e_param_value, 
                                     "opinion_model": opinion_model, 
                                     "budget": budget_perc, "type_of_constraint": constraint,
                                     "initial_pol": initial_pol}
            
            hash_object = hashlib.md5(f'{folder_name}'.encode('utf-8'))
            hash_folder_name = hash_object.hexdigest()
            print("  ", folder_name, hash_folder_name)

            write_option = "w" if index == 0 else "a"
            with open(folder_path / Path("folder_names.txt"), write_option) as file:
                file.write(f"{folder_name}| {hash_folder_name}\n") # Save folder name and its hash
            index += 1
            if os.path.exists(folder_path / Path(hash_folder_name)):
                print(f"!!!! File {folder_name} already exists!")
            if not os.path.exists(folder_path / Path(hash_folder_name)) and write_files:
                os.makedirs(folder_path / Path(hash_folder_name))     # Create folder name
                with open(folder_path / Path(hash_folder_name) / Path("generative_parameters.pkl"), "wb") as file:
                    pickle.dump(generative_parameters, file)         # Save params dict in proper folder
                #############################################################################################
                
                # 1. graph
                nx.write_gexf(G, folder_path / Path(hash_folder_name) / Path(f"graph_sample00_python.gexf"))

                # 2. opinions (equilibrium: z_eq, internal: s_int)
                _z_eq = define_set_opinions(opinion_model, G, pol=initial_pol, standardized=None)

                s_int = standardize(np.array((2*I - A_eq) @ _z_eq), scale=1)
                
                z_eq, _ = scipy.sparse.linalg.bicg(2*I - A_eq, s_int, tol=1e-6)
                np.save(folder_path / Path(hash_folder_name) / f"eq_opinions_sample00_python", z_eq)
                np.save(folder_path / Path(hash_folder_name) / f"internal_opinions_sample00_python", s_int)


                scipy.sparse.save_npz(folder_path / Path(hash_folder_name) / f"A_eq_sparse_sample00_python.npz", A_eq)     
                Is, Js = np.nonzero(A_eq)
                assert len(Is) == len(A_eq.data), (len(Js), len(A_eq.data))
                
                
                ## check optimization 
                print(f"\n   s mean={np.mean(s_int):.3f}, min={np.min(s_int):.3f}, max={np.max(s_int):.3f}\n")