"""Plot experiments results
"""
# Imports
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
import seaborn as sb
import sys
sys.path.append('../src')
from utils import check_results


# Parsing task
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=int, choices=[0, 1], help = '0 approximation, 1 runtime.', default=0)
parser.add_argument("-c", "--constraint", type=int, choices=[0, 1], help = '0 trace, 1 degree.', default=0)
parser.add_argument("-x", "--n_cols", type=int, help = 'Number of columns in subplots.', default=2)
parser.add_argument("-s", "--scale", type=bool, help = 'Scale the axes properly.', default=True)
parser.add_argument("-w", "--test", type=int, choices=[0, 1], help = '0 is not a test, 1 is a test.',  default=0)

args = vars(parser.parse_args())

# Folders
figure_path = Path("../draft/figures/")
constraint = "trace" if args['constraint'] == 0 else "degree"
task = 'approximation' if args['task'] == 0 else 'runtime'
folder_path = Path("../datasets/experiments") / Path(f"{task}_task__{constraint}_constraint")
with open(folder_path / Path("folder_names.txt"), "r") as file:
    folders = [line.split("| ")[1][:-1] for line in file]
with open(folder_path / Path("folder_names.txt"), "r") as file:
    folders_real_names = [line.split("| ")[0] for line in file]


is_test = args['test'] == 1
folders = folders[0:1] if is_test else folders


# Figures

n_rows, n_cols, scale = (len(folders)//args['n_cols']+1, 2*args['n_cols'], args['scale']) if not is_test else (1,1,args['scale'])
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4.5))


if not is_test:
    # Sub-plots loop
    for k, simulation_folder in enumerate(folders):

        i = 2*k//n_cols
        j = 2*k%n_cols
        print('\n', i, j, folders_real_names[k])
        # params
        with open(folder_path / simulation_folder / Path("generative_parameters.pkl"), "rb") as file:
            params = pickle.load(file)
        print("   opening files")
        s = np.load(folder_path / simulation_folder / "internal_opinions_sample00_python.npy")
        z = np.load(folder_path / simulation_folder / "eq_opinions_sample00_python.npy")
        G = nx.read_gexf(folder_path / simulation_folder / Path("graph_sample00_python.gexf"))
        #axs[i, j].hist(s, bins=int(len(s)*.25), label = "Internal opinions (s)")
        #axs[i, j+1].hist(z, bins=int(len(s)*.25), label = "Equilibrium opinions (z)", color="orange")
        print("   plotting")
        axs[i, j].hist(s, label = "Internal opinions (s)")
        axs[i, j+1].hist(z, label = "Equilibrium opinions (z)", color="orange")

        #############################################################
        ###################### Estetic  #############################
        #############################################################
        ### Label
        #if i==0 and j==2:
        #    axs[i, j].legend(loc='upper center', bbox_to_anchor=(-0.1, 1.75), 
        #                     fancybox=True, framealpha=1, shadow=True, borderpad=1)
        print("   labeling")
        axs[i, j].set_xlabel("Internal opinions (s)", fontsize=15)
        axs[i, j+1].set_xlabel("Equilibrium opinions (z)", fontsize=15)
        axs[i, j].set_ylabel("Frequency", fontsize=15)

        ### Title
        t = f"{params['net_model']} edges/nodes={G.number_of_edges()}/{G.number_of_nodes()}\n"
        t += (f"distrib={params['opinion_model']} pol={params['initial_pol']}\n"
              f"budget={int(params['budget']*100)}%")# constraint={params['type_of_constraint']}")
        axs[i, j].set_title(t)
        #axs[i, j].grid()


        if scale:
            axs[i, j].set_ylim(bottom=1)
        ##############################################################
    print("Producing figure")
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=.35, hspace=.65)   
    sb.despine()
    plt.savefig(figure_path / Path('distributions.pdf'), bbox_inches='tight')