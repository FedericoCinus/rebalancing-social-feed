"""Graph generation and loading
"""
import sys
sys.path.append('../src')
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
import scipy
from scipy.sparse import identity
from sklearn.preprocessing import normalize
from utils import cast_to_doubly_stochastic


def assert_adj_stochasticity(A, directed):
    if directed:
        row = np.all(np.absolute(np.array(A.sum(axis=1)).flatten() - np.ones(A.shape[0])) < 1e-6)
        assert row, np.argmin(np.array(A.sum(axis=1)).flatten())
    else:
        col = np.all(np.absolute(np.array(A.sum(axis=0)).flatten() - np.ones(A.shape[0])) < 1e-3)
        row = np.all(np.absolute(np.array(A.sum(axis=1)).flatten() - np.ones(A.shape[0])) < 1e-3)
        assert (col and row)
        
def make_doubly_stochastic(M, g_name='', ε=1e-3):
    from sklearn.preprocessing import normalize
    i = 0
    while np.any(np.absolute(M.sum(axis=1)-np.ones(M.shape[0])) > ε):
        M = normalize(M, axis=1, norm='l1')
        M = normalize(M, axis=0, norm='l1')
        if i == 25000:
            print(f"      Reached max iterations, graph {g_name} cannot be transformed into doubly stochastic")
            return None
        i += 1
    return M

def normalize_sparse_L(L_sparse: scipy.sparse.csr_matrix, verbose: bool = False):
    """Normalize the laplacian of initial graph. It introduces self loops.
    """
    L_sparse_inner = L_sparse.copy()
    
    L_sparse_inner = np.multiply(float(1/L_sparse_inner.diagonal().max()), L_sparse_inner)
    return L_sparse_inner

def initialize_sparse_L(L_eq: np.matrix, verbose: bool = False) -> np.matrix:
    """Returns sparse laplacian with uniform weights
    """
    Is, Js = np.nonzero(L_eq) # indices of the laplacian entries
    L_sparse_init = L_eq.copy() # intializing new laplacian
    unw_degrees = np.array(L_eq.astype(bool).sum(axis=1)-1).flatten() # unweighted degrees
    L_sparse_init.data = np.array([-1 if i!= j else unw_degrees[i] for i,j in zip(Is, Js)]).astype(np.float64)
    
    # projection of laplacian
    I = identity(L_sparse_init.shape[0])
    A = I - L_sparse_init # retrieve adj
    A = cast_to_doubly_stochastic(A, verbose=verbose) # make doubly stochastic
    L_sparse_init = I - A # retrieve laplacian
    
    #L_sparse_init /= L_sparse_init.diagonal().max()
    return L_sparse_init
     
    
def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)

############################################################################################

def define_graph_instance(g_name: str, kwargs: dict, directed: bool, define_weights = False, 
                          base="../datasets", verbose: bool=False):
    """Returns boolean adjacency, identity, laplacian and graph nx instance
    """
    α = -2 #exponent powerlaw for weights
    
    # 1. Barabasi-Albert model
    if g_name == 'barabasi':
        if directed:
            G = nx.scale_free_graph(n=kwargs['n']).to_directed()
        else:
            G = nx.barabasi_albert_graph(**kwargs)
        if define_weights:
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = rndm(1, 10, α, 1)[0]
    
    # 2. Erdos-Renyi model
    elif g_name == 'erdos':
        G = nx.fast_gnp_random_graph(**kwargs, directed=directed)
        if define_weights:
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = rndm(1, 10, α, 1)[0]
    
    # 3. Stochastic Block model
    elif g_name == 'sbm':
        G = stochastic_block_model_graph(**kwargs, directed=directed)
        if define_weights:
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = rndm(1, 10, α, 1)[0]
    
    
    ###############################################################
    # 4. Real graphs
    # directed
    elif g_name in ('brexit', 'vaxNoVax'):
        G = nx.read_edgelist(Path(f'{base}/graphs/directed/{g_name}/') / Path("edgelist.txt"), 
                             nodetype=int, create_using=nx.DiGraph)
        username2index = {u: i for i, u in enumerate(G.nodes())}
        with open(Path(f'{base}/graphs/directed/{g_name}/username2index.pkl'), "wb") as file:
            pickle.dump(username2index, file)
        G = nx.relabel_nodes(G, username2index) 
    elif g_name in ("ego-twitter", "facebook-wosn-wall", "munmun_twitter_social", "soc-epinions1",
                    "flickr-growth", "digg-friends", "epinions_large"):
        G = nx.read_edgelist(Path(f'{base}/graphs/directed/real_large/{g_name}/{g_name}'), create_using=nx.DiGraph)
        G = nx.reverse(G)
        G = nx.relabel_nodes(G, {u: i for i, u in enumerate(G.nodes())})
    # undirected   
    elif g_name in ("cs_aarhus", "dimacs10-football", "political-books", "students"):
        G = nx.read_graphml(Path(f'{base}/graphs/undirected/real_small/{g_name}/{g_name}-network.graphml'))
        G = nx.relabel_nodes(G, {u: i for i, u in enumerate(G.nodes())})
    
     
    
    ##########################################
    # Checking directionality and connectivity
    if (not directed) and G.is_directed():
        print("   ! Casting graph G from directed to undirected..")
        G = nx.to_undirected(G)
    if (not directed) and (not nx.is_connected(G)):
        print("   ! Selecting largest connected component and relabeling nodes..")
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
        G = nx.convert_node_labels_to_integers(G)
    if (directed) and (not nx.is_weakly_connected(G)):
        print("   ! Selecting largest weakly connected component and relabeling nodes..")
        Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)[0]
        G = G.subgraph(Gcc)
        G = nx.convert_node_labels_to_integers(G)
            
    if directed:
        remove = [node for node, degree in dict(G.out_degree()).items() if degree == 0]
        i = 0
        while len(remove)>0:
            print(f"   ! Removing out-disconnected nodes.. \n  iter={i}, remaining {G.number_of_nodes()}")
            nodes = list(G.nodes())
            for u in remove:
                nodes.remove(u)
            G = G.subgraph(nodes)
            G = nx.convert_node_labels_to_integers(G)
            remove = [node for node, degree in dict(G.out_degree()).items() if degree == 0]
            i += 1
    
    # Normalize
    if directed:
        A = normalize(nx.adjacency_matrix(G), axis=1, norm='l1')
    else:
        A = make_doubly_stochastic(nx.adjacency_matrix(G), g_name)

    return A, G



def stochastic_block_model_graph(n: int, β=.05, directed: bool=False):
    '''Returns nx.Graph according to stochastic_block_model
    '''
    N = 100
    i = 0
    lengths = []
    is_connected = False
    sizes = [int(n*.55), n-int(n*.55)]
    α, β = (16/n, 2/n*β)
    probs = np.array([[α, β], 
                      [β, α]])
    while not is_connected:
        print(f"   SBM generation {i}/{N}, n components: {len(lengths)}", end='\r')
        G = nx.stochastic_block_model(sizes, probs, directed=directed)
        is_connected = nx.is_weakly_connected(G) if directed else nx.is_connected(G)
        components = nx.weakly_connected_components(G) if directed else nx.connected_components(G)
        lengths = [len(c) for c in sorted(components, key=len, reverse=True)]
        i += 1
        if i > N:
            print(f"   reached max iterations in SBM {i}/{N}")
            break
    return G