import pickle
import numpy as np
import networkx as nx
from pathlib import Path

standardize = lambda x, scale=1: scale*(x - np.mean(x)) / np.std(x)
normalize = lambda x, lb, up: (up - lb) * (x - np.min(x)) / (np.max(x) - np.min(x)) + lb
polarization_f = lambda x, pol: np.power(np.absolute(x), 1 / pol) if x >= 0 else -np.power(np.absolute(x), 1 / pol)
polarization_f = np.vectorize(polarization_f)


def define_set_opinions(o_name: str, G: nx.Graph, pol: float = 1.,
                        standardized: int = None, o_min: float = -.5, o_max: float = .5):
    n = G.number_of_nodes()
    if o_name in {'uniform', 'constant'}:
        if o_name == 'uniform':
            opinions = np.random.uniform(o_min, o_max, size=(n, 1))
        elif o_name == 'constant':
            assert o_max == o_min, f'Set o_max ({o_max}) = o_min ({o_min})'
            opinions = np.repeat(o_min, n).reshape((n, 1))
            normalized = None
            print(f'Constant opinions with polarization {pol}, normalized = {normalized}')
            
        opinions = polarization_f(opinions, pol) # insert polarization
    
    # community-based opinions
    elif o_name == 'gaussian':
        communities = nx.algorithms.community.kernighan_lin_bisection(G.to_undirected())
        opinions = sampling_gaussian_opinions(n, communities, pol, o_min, o_max)
    # real opinions
    elif o_name in ('brexit', 'vaxNoVax'):
        folder = f'../datasets/graphs/directed/{o_name}'
        with open(Path(f'{folder}/username2index.pkl'), "rb") as file:
            _username2index = pickle.load(file)
        index2username = {v:k for k,v in _username2index.items()}


        with open(Path(f"{folder}/propagations_and_polarities.pkl"), "rb") as file:
            propagations, polarities = pickle.load(file)

        node2polarities = {}
        for prop_idx, active_nodes in enumerate(propagations):
            for node in active_nodes:
                if node in node2polarities:
                    node2polarities[node].append(polarities[prop_idx])
                else:
                    node2polarities[node] = [polarities[prop_idx]]
                    
        opinions = np.array([np.mean(node2polarities[index2username[i]]) for i in range(G.number_of_nodes())])
    
    
    # scaling
    if standardized == 0:
        opinions = normalize(opinions, o_min, o_max) 
    elif standardized == 1: # standardize
        opinions = standardize(opinions)

    return opinions.reshape((len(opinions), 1))


def sampling_gaussian_opinions(n, communities, pol, o_min, o_max):
    # comupting the range and centre of the opinion spectrum in input
    Δopinion, middle_point = ((o_max - o_min)/2, (o_max + o_min)/2)
    δopinion = Δopinion / ((len(communities))//2)
    # defining the locations of the normal distributions on for each community
    μs= np.zeros(len(communities)) 
    # computing pairs of opposite opinion locations with distance proportional to pol
    for i in range(0, len(communities)//2+1, 2):
        μs[i] = middle_point + (.1*pol) * δopinion * (i+1)
        μs[i+1] = middle_point - (.1*pol) * δopinion * (i+1)
    # sampling community opinions
    opinions = np.zeros(n)
    for community_idx, users in enumerate(communities):
        for u in users:
            opinions[u] = np.random.normal(μs[community_idx], Δopinion/6)
    return opinions
