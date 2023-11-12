"""Plot experiments results
"""
# Imports
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import powerlaw
import pickle
import scipy
import seaborn as sb
import sys
sys.path.append('../src')
import algorithms_approx3 as LcGD
from utils import check_results
# Color settings
import matplotlib

# Parsing task
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=int, choices=[4, 1], help = '4 approximation, 1 runtime.', default=0)
parser.add_argument("-d", "--directed", type=int, choices=[0, 1], help = '0 undirected, 1 directed', default=1)
parser.add_argument("-c", "--constraint", type=int, choices=[0, 1], help = '0 trace, 1 degree.', default=1)
parser.add_argument("-x", "--n_cols", type=int, help = 'Number of columns in subplots.', default=2)
parser.add_argument("-s", "--scale", type=bool, help = 'Scale the axes properly.', default=False)
parser.add_argument("-w", "--test", type=int, choices=[0, 1], help = '0 is not a test, 1 is a test.',  default=0)

args = vars(parser.parse_args())

# Params
plot_budgets = False
use_specific_order = True

# Folders
figure_path = Path("../draft/figures/")
constraint = "trace" if args['constraint'] == 0 else "degree"
direction = "directed" if args['directed'] else "undirected"
task = 'approximation' if args['task'] == 4 else 'runtime'
folder_path = Path("../datasets/experiments") / Path(direction) / Path(f"{task}_curves_task__{constraint}_constraint")
with open(folder_path / Path("folder_names.txt"), "r") as file:
    folders2real_names = {line.split("| ")[1][:-1]: line.split("| ")[0] for line in file}
folders = list(folders2real_names.keys())
    
# Ordering
cut_objective_curve = False
use_specific_order = False
if use_specific_order:
    folders = []


is_test = args['test'] == 1
folders = folders[0:8] if is_test else folders

# Line colors
method2color = {}
methods2plot_list = ["LcGD  lr=0.2  budget=0.25", "LcGD  lr=0.2  budget=0.5",  
                            "LcGD  lr=0.2  budget=0.75",  "LcGD  lr=0.2  budget=1.0" , 
                            "oppo-view", "pop", "neutral-view"]

cmap = matplotlib.cm.get_cmap('RdYlGn')
for i, m in enumerate(methods2plot_list):
    method2color[m] = cmap(.15+i/(len(methods2plot_list)-3))

ε = 1e-6

# ------------------------------------------------------------------------------
# Figures
n_rows, n_cols, scale = (2, len(folders)//2, args['scale'])
print(n_rows, n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))


# Sub-plots loop
for k, simulation_folder in enumerate(folders):

    i = k//n_cols
    j = k%n_cols
    print('\n', i, j, folders2real_names[simulation_folder])
    # params
    with open(folder_path / simulation_folder / Path("generative_parameters.pkl"), "rb") as file:
        params = pickle.load(file)
        #print(params['net_model'])
    # data
    with open(folder_path / simulation_folder / Path("experiments_data_method.pkl"), "rb") as file:
        experiments_data = pickle.load(file)
    s = np.load(folder_path / simulation_folder / Path("internal_opinions_sample00_python.npy"))
    A_eq = scipy.sparse.load_npz(folder_path / simulation_folder / Path("A_eq_sparse_sample00_python.npz"))
    z_eq = np.load(folder_path / simulation_folder / Path("eq_opinions_sample00_python.npy"))


    # First objective
    I = scipy.sparse.identity(A_eq.shape[0])
    z, _ = scipy.sparse.linalg.bicg(2*I - A_eq, s, tol=ε)
    z = z.reshape((len(s), 1))
    #initial_PD_index = LcGD.objective_f(s, A_eq)
    #axs[i, j].axhline(0., linestyle="dashed", color="red", linewidth=1.75, 
    #                        label="original equilibrium", alpha=0.95)  # upper bound obj_eq/obj_eq
    
    # Natural objective (using input graph)
    obj_nat = LcGD.objective_f(z, A_eq)


    
    #color = cmap(.15+k/4)
    
    # 4. Heuristcs
    for method_idx, data in experiments_data.items():
        
        
        if 'routine' not in data['method_config']['params']:
            label = data['method_config']["name"] 
        else:
            label = data['method_config']['params']['routine']    
        if label == "opposite_view_heuristic":
            label = "oppo-view"
        elif label == "popularity_heuristic":
            label = "pop"
        elif label == "simple_GD":
            label = "no-accellerated-LcGD"
        elif label == "NAG":
            label = "NAG-LcGD"
        elif label == "ADAM":
            label = "LcGD"
        elif label == "neutral_view_heuristic":
            label = "neutral-view"

        if label in ("pop", "oppo-view", "neutral-view"):
            A = data['A']
            z, _ = scipy.sparse.linalg.bicg(2*I - A, s, tol=ε)
            z = z.reshape((len(s), 1))
            obj = LcGD.objective_f(z, A)
            if label == "oppo-view":
                color = "olive" 
            elif label == "pop":
                color = "red" 
            else:
                color = "olivedrab" 
            axs[i, j].axhline(1-obj/obj_nat, linestyle="dashed", color=color, linewidth=1.75, 
                            label=label, alpha=0.95)  # upper bound obj_eq/obj_eq
        else:
            label += f"  lr={data['method_config']['params']['lr_params']['lr_coeff']}" 
            label += f"  budget={data['method_config']['params']['grad_params']['budget']}"
            if label in methods2plot_list:
                y = 1 - np.array(data['objectives']) / obj_nat
                # displaying minimum
                y = y[:np.argmin(y)] if cut_objective_curve else y
                axs[i, j].plot(y, color=method2color[label], linewidth=4., label=label, alpha=0.9)

    #############################################################
    ###################### Aestetic  #############################
    #############################################################
    ### Label
    if i==0 and j==2:
        axs[i, j].legend(loc='upper center', bbox_to_anchor=(-.25, 1.5), ncol = 3)
    axs[i, j].set_xlabel("Iterations", fontsize=15)
    axs[i, j].set_ylabel(r"$\rho_{0}$", fontsize=15)

    ### Title
    if params['net_model'] == 'erdos':
        t = f"{params['net_model']} n={params['numb_nodes']} p={params['p']}"
    elif params['net_model'] == 'bara':
        t = f"{params['net_model']} n={params['numb_nodes']} m={params['m']}"
    else:
        t = f"{params['net_model']}   n={params['numb_nodes']}"
    if params['net_model'] not in ('brexit', 'vaxNoVax'):
        t += f"\n opinions={params['opinion_model']} polarity={params['initial_pol']}"
    axs[i, j].set_title(t)
    axs[i, j].grid()


    if scale:
        axs[i, j].set_ylim(bottom=1)
    ##############################################################
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
plt.tight_layout()
plt.subplots_adjust(left=0., bottom=0., right=0.9, top=0.95, wspace=.45, hspace=.45)   
sb.despine()
plt.savefig(figure_path / Path(f'approximation.pdf'), bbox_inches='tight', dpi=200)
    