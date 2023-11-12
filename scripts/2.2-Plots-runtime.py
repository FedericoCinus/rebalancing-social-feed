# Imports
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sb
import scipy
import sys
sys.path.append('../src')


# Parsing task
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=int, choices=[0, 1, 777], help = '0 approximation, 1 runtime, 777 test task.', default=1)
parser.add_argument("-c", "--constraint", type=int, choices=[0, 1], help = '0 trace, 1 degree.', default=1)
parser.add_argument("-x", "--n_cols", type=int, help = 'Number of columns in subplots.', default=1)
parser.add_argument("-s", "--scale", type=bool, help = 'Scale the axes properly.', default=True)
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

# Color settings
method2color = {}
method2color["simple_GD"] = 'silver'
method2color["ADAM"] = 'blue'
method2color["AdaGrad"] = 'limegreen'


# Figures
n_rows, n_cols, scale = (1, 1, args['scale']) #args['n_cols']
fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 3.5))

##############################################################################
# 1. Simulation folders loop
simulation2data = {" ": [], "Number of nodes": [], "Number of edges": [], "Simulation": [], "Running time (sec)": []}
for i, simulation_folder in enumerate(folders):
    print(folders_real_names[i])
    sample_idx = folders_real_names[i].split("__")[-1]
    
    # 2.1 generative params
    with open(folder_path / simulation_folder / Path("generative_parameters.pkl"), "rb") as file:
        params = pickle.load(file)
    
    # 2.2 setting name
    if params['net_model'] == 'erdos':
        t = f"{params['net_model']} p={params['p']}\n"
    elif params['net_model'] == 'bara':
        t = f"{params['net_model']} m={params['m']}\n"
    else:
        t = f"{params['net_model']}\n"
    t += (f"{params['opinion_model']} opinions pol={params['initial_pol']}\n")
          #f"budget={int(params['budget']*100)}% constraint={params['type_of_constraint']}")
    
    
    
    # 2.3 initial opinions
    s = np.load(folder_path / simulation_folder / Path("internal_opinions_sample00_python.npy"))
    G = nx.read_gexf(folder_path / simulation_folder / Path("graph_sample00_python.gexf"))
    
    # 3. Model loop
    with open(folder_path / simulation_folder / Path("experiments_data_method.pkl"), "rb") as file:
            experiments_data = pickle.load(file)

        # 1. Opt value
        
    for method_idx, data in experiments_data.items():
        # 2.4 Experimental data
        data = experiments_data[method_idx]
        # 3.1 name for each method
        if data['method_config']['name'] != 'SCS':
            name = data['method_config']['params']['routine'] 
            name = "LcGD" #name if name != "simple_GD" else "LcGD"
            #name += ' ' + str(data["method_config"]["params"]['lr_params'])

        elif not isinstance(data['IpL'], int):
            name = data['method_config']['name']
        
        # 3.2 y values for each method
        if not isinstance(data['IpL'], int):
            runtime = data['runtime']
            numb_nodes = params['numb_nodes']
            
        
        simulation2data[" "].append(name)
        simulation2data["Number of nodes"].append(numb_nodes)
        simulation2data["Number of edges"].append(G.number_of_edges())
        simulation2data["Simulation"].append(sample_idx)
        simulation2data["Running time (sec)"].append(runtime)
            
            
pd.DataFrame(simulation2data).to_csv("../datasets/runtime.csv")       
##############################################################################
#sb.scatterplot(data=simulation2data, x="Number of edges", y="Running time (sec)", 
#               ax=axs, hue=" ",)# markers=" ", errorbar=("ci", 100),)
sb.lineplot(data=simulation2data, x="Number of edges", y="Running time (sec)", 
               ax=axs, hue=" ", markers=" ", errorbar=("ci", 100),)
if scale:
    axs.set_xscale("log")
    #axs.set_yscale("log")
axs.grid(alpha=.5)

sb.despine()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=.65)

plt.savefig(figure_path / Path('runtime.pdf'), bbox_inches='tight')