import networkx as nx
from queue import PriorityQueue
import random
from tqdm import tqdm

def k_core_self_loop(G, k=None, core_number=None):
    def k_filter(v, k, c):
        return c[v] >= k
    
    
    def _core_subgraph(G, k_filter, k=None, core=None):
        """Returns the subgraph induced by nodes passing filter `k_filter`.

        Parameters
        ----------
        G : NetworkX graph
           The graph or directed graph to process
        k_filter : filter function
           This function filters the nodes chosen. It takes three inputs:
           A node of G, the filter's cutoff, and the core dict of the graph.
           The function should return a Boolean value.
        k : int, optional
          The order of the core. If not specified use the max core number.
          This value is used as the cutoff for the filter.
        core : dict, optional
          Precomputed core numbers keyed by node for the graph `G`.
          If not specified, the core numbers will be computed from `G`.

        """
        if core is None:
            core = core_number(G)
        if k is None:
            k = max(core.values())
        nodes = (v for v in core if k_filter(v, k, core))
        return G.subgraph(nodes).copy()
    
    return _core_subgraph(G, k_filter, k, core_number(G))



def core_number(G, w='weight'):
    q = PriorityQueue()
    #degree = list(G.degree(weight='weight'))
    degree = list(G.degree(weight=w))
    for i in range(0,len(degree)): # fix self-loop contati due volte
        n,d = degree[i]
        if G.has_edge(n,n):
            degree[i] = (n, d - G[n][n][w] if w in G[n][n] else 1)
    for (n,d) in list(degree):
        q.put((d,n))
    nbrs = {v: set(nx.all_neighbors(G, v)) for v in G}
    nodes = set(G.nodes())
    curr_degree = dict(degree)
    cores = dict.fromkeys(G.nodes())
    k=0
    while not q.empty():
        (d,v) = q.get()
        if nodes and v in nodes:
            cores[v] = max(k,curr_degree[v])
            k=cores[v]
            nodes.remove(v)
            for n in nbrs[v]:
                if n in nodes and n != v: 
                    value = 0
                    if G.has_edge(v,n):
                        value = G[v][n][w] if w in G[v][n] else 1
                    if G.is_directed and G.has_edge(n,v):
                        value = G[n][v][w] if w in G[n][v] else 1
                    curr_degree[n]-= value
                    q.put((n,curr_degree[n]))
    cores = {k: round(v, 4) for k, v in cores.items()}
    return cores


def core_number_worlds(worlds, w, use_tqdm=True):
    cores_per_world = []
    for W in tqdm(worlds, disable=not use_tqdm):
        all_cores_W = core_number(W, w)
        cores_per_world.append(all_cores_W)
    return cores_per_world