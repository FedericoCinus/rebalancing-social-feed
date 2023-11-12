# Imports
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sb
import scipy
import sys
sys.path.append('../src')
from algorithms_approx3 import objective_f

ε = 1e-6


# Parsing task
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=int, choices=[0, 1, 2, 777], help = '0 approximation, 1 runtime, 777 test task.', default=2)
parser.add_argument("-d", "--directed", type=int, choices=[0, 1], help = '0 undirected, 1 directed', default=1)
parser.add_argument("-c", "--constraint", type=int, choices=[0, 1], help = '0 trace, 1 degree.', default=1)
parser.add_argument("-x", "--n_cols", type=int, help = 'Number of columns in subplots.', default=1)
parser.add_argument("-s", "--scale", type=bool, help = 'Scale the axes properly.', default=False)
args = vars(parser.parse_args())

# Folders
figure_path = Path("../draft/figures/")
constraint = "trace" if args['constraint'] == 0 else "degree"
if args['task'] == 0:
    task = 'approximation'
elif args['task'] == 1:
    task = 'runtime'
elif args['task'] == 2:
    task = 'modularity'
    
directed = args['directed']==1
direction = "directed" if directed else "undirected"    
    
folder_path = Path("../datasets/experiments") / Path(direction) / Path(f"{task}_task__{constraint}_constraint")
with open(folder_path / Path("folder_names.txt"), "r") as file:
    folders = [line.split("| ")[1][:-1] for line in file]
with open(folder_path / Path("folder_names.txt"), "r") as file:
    folders_real_names = [line.split("| ")[0] for line in file]




# Figures
n_rows, n_cols, scale = (1, 1, args['scale']) #args['n_cols']

##############################################################################
# 1. Simulation folders loop
simulation2data = {"1-OBJ/UB": [], "Number of nodes": [], "β": [], 
                   "pol": [], "simulation": [], "method": [], "opinions": []}
for i, simulation_folder in enumerate(folders):
    
    # 2.0 samples
    print(folders_real_names[i])
    sample_idx = folders_real_names[i].split("__")[-1]
    
    # 2.1 generative params
    with open(folder_path / simulation_folder / Path("generative_parameters.pkl"), "rb") as file:
        params = pickle.load(file)
    
    # 2.2 retrieving params
    net_model, β, pol, opinion_model = (params['net_model'], params['β'], 
                                     params['initial_pol'], params['opinion_model'])
    numb_nodes = params['numb_nodes']

    # 2.3 initial opinions
    s = np.load(folder_path / simulation_folder / Path("internal_opinions_sample00_python.npy"))
    A_eq = scipy.sparse.load_npz(folder_path / simulation_folder / Path("A_eq_sparse_sample00_python.npz"))
    
    # 2.4 first objective
    initial_PD_index = objective_f(s, A_eq)

    # 2.5 natural objective (using input graph)
    I = scipy.sparse.identity(A_eq.shape[0])
    z, _ = scipy.sparse.linalg.bicg(2*I - A_eq, s, tol=ε)
    z = z.reshape((len(s), 1))
    obj_nat = objective_f(z, A_eq)
    

    # 3. Model loop
    filepath = folder_path / simulation_folder / Path("experiments_data_method.pkl")
    if os.path.exists(filepath):
        
        with open(filepath, "rb") as file:
            experiments_data = pickle.load(file)
        
      
        # 4. Heuristic value    
        for method_idx, data in experiments_data.items():
            method = data['method_config']['name']
            simulation2data["method"].append(method)
            
            A = data['A']
            z, _ = scipy.sparse.linalg.bicg(2*I - A, s, tol=ε)
            z = z.reshape((len(s), 1))
            obj = objective_f(z, A)
            ratio = obj/obj_nat
            simulation2data["1-OBJ/UB"].append(1-ratio)
            
            simulation2data["Number of nodes"].append(numb_nodes)
            simulation2data["β"].append(β)
            simulation2data["pol"].append(pol)
            simulation2data["simulation"].append(sample_idx)
            simulation2data["opinions"].append(opinion_model)

            
            
            
##############################################################################
cmap = matplotlib.cm.get_cmap('RdYlGn_r')
methods = ["opposite_view_heuristic", "GD_optimizer"]
method2nicename = {"opposite_view_heuristic": "oppo-view", "GD_optimizer": "LcGD"}
opinions = ["uniform", "gaussian"]
fig, axs = plt.subplots(1, len(opinions)*len(methods), figsize=(4.5*len(methods)*len(opinions), 3.5))

cmap = matplotlib.cm.get_cmap('RdYlGn_r').reversed()
df = pd.DataFrame(simulation2data)
i = 0
for op in opinions:
    for me in methods:
        means = df[(df.method==me) & (df.opinions==op)].pivot_table("1-OBJ/UB", "β", "pol", aggfunc='mean')
        sb.heatmap(means, ax=axs[i], cmap=cmap, vmin=0., vmax=.5, annot=False, linewidth=.5, fmt=".0%")
        axs[i].set_title(f"Method: {method2nicename[me]}\nOpinions: {op}")
        # ticks
        labelsx = [item.get_text() for item in axs[i].get_xticklabels()]
        labelsy = [item.get_text() for item in axs[i].get_yticklabels()]
        axs[i].set_xticklabels([str(round(float(label), 2)) for label in labelsx])
        axs[i].set_yticklabels([str(round(float(label), 2)) for label in labelsy])
        i += 1
sb.despine()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=.65)

plt.savefig(figure_path / Path('heatmap.pdf'), bbox_inches='tight')