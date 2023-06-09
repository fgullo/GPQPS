import csv
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import time
from tqdm import tqdm
import scipy
import networkx as nx



def get_sl2_summary(S, graph_type):
    #s = nx.from_numpy_matrix(np.matrix(S.A), create_using=graph_type)
    s = nx.from_numpy_array(np.matrix(S.A), create_using=graph_type)
    return s


def get_summary(G, n, k, max_iter, is_directed, is_weighted, seed, mini_batch=False):
    t1 = time.time()
    A = get_adjacency_matrix(G, n, is_directed, is_weighted)
    t2 = time.time()
    print('Running time adj_matrix (NEW): %f' % (t2 - t1))

    # partition given by the clustering
    t3 = time.time()
    partition, node2supernode = kmeans(A, k, max_iter, seed, mini_batch=mini_batch)
    t4 = time.time()
    print('Running time kmeans: %f' % (t4 - t3))

    S, S_prob, S_avgweight = density_matrix(A, G, partition, node2supernode, is_directed, is_weighted)
    return S, S_prob, partition, node2supernode, S_avgweight


def get_adjacency_matrix(G, n, is_directed, is_weighted, w='weight'):
    A = scipy.sparse.lil_matrix((n, n), dtype=float)

    for edge in G.edges():
        if is_weighted:
            A[int(edge[0]), int(edge[1])] = round(G.edges[edge][w], 2)
        else:
            A[int(edge[0]), int(edge[1])] = 1

        if not is_directed:
            A[int(edge[1]), int(edge[0])] = A[int(edge[0]), int(edge[1])]
    return A


def kmeans(A, k, max_iter, seed, mini_batch=False):
    if not mini_batch:
        kmeans = KMeans(n_clusters=k, init='k-means++', algorithm='auto', max_iter=max_iter, random_state=seed).fit(A)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=3, max_iter=max_iter, random_state=seed).fit(A)

    # id_supernode : list of id_nodes in it
    partition = {i: np.where(kmeans.labels_ == i)[0] for i in range(k)}

    # supernodes dictionary node_id: supernode_id
    node2supernode = {node: i for i, nodes in partition.items() for node in nodes}

    return partition, node2supernode


def lifted_density_matrix(S, n, partition, supernode):
    t0 = time.time()
    S_lifted = scipy.sparse.lil_matrix((n, n), dtype=float)
    S_grass = scipy.sparse.lil_matrix((n, n), dtype=float)

    for i in tqdm(range(n)):
        for j in range(n):
            s_i = supernode[i]
            s_j = supernode[j]

            S_lifted[i, j] = S[s_i, s_j]

            if s_i != s_j:
                S_grass[i, j] = S_lifted[i, j]
            elif i != j:
                S_grass[i, j] = S_lifted[i, j] * len(partition[s_j]) / (len(partition[s_j]) - 1)
            else:
                S_grass[i, j] = 0

    t1 = time.time()
    print('Running time A_S_lifted + Grass: %f' % (t1 - t0))
    return S_lifted, S_grass


def density_matrix(A, G, partition, node2supernode, is_directed, is_weighted):
    t0 = time.time()
    k = len(partition)

    S = scipy.sparse.dok_matrix((k, k))  # matrix of the expected weights
    S_prob = scipy.sparse.dok_matrix((k, k)) # matrix of the edge probabilities
    S_avgweight = scipy.sparse.dok_matrix((k, k)) # matrix of the average weights

    for edge in G.edges():
        node1 = int(edge[0])
        node2 = int(edge[1])
        edge_weight = A[node1, node2] if is_weighted else 1
        
        supernode1 = min(node2supernode[node1], node2supernode[node2])
        supernode2 = max(node2supernode[node1], node2supernode[node2])
        
        #we always consider edges as directed
        #if the graph is undirected, we double the count/weight of every undirected edge that is not a self-loop (as it corresponds to two directed edges)
        if is_directed or node1 == node2:
            S[supernode1, supernode2] += edge_weight
            S_prob[supernode1, supernode2] += 1
            if supernode1 != supernode2:
                S[supernode2, supernode1] += edge_weight
                S_prob[supernode2, supernode1] += 1
        else:
            S[supernode1, supernode2] += 2*edge_weight
            S_prob[supernode1, supernode2] += 2
            if supernode1 != supernode2:
                S[supernode2, supernode1] += 2*edge_weight
                S_prob[supernode2, supernode1] += 2
    
    #make the summary an undirected graph
    for i, j in zip(S.nonzero()[0], S.nonzero()[1]):
        S[i,j] = S[j,i]
        S_prob[i,j] = S_prob[j,i]
    
    #build superedges
    for i, j in zip(S.nonzero()[0], S.nonzero()[1]):
        den = len(partition[i]) * len(partition[j]) #i == j
        if i != j:
            den *= 2

        S_avgweight[i,j] = 0 if S_prob[i,j] == 0 else round(S[i,j]/S_prob[i,j], 5)
        S[i,j] = round(S[i,j]/den, 5)
        S_prob[i,j] = round(S_prob[i,j]/den, 5)

    t1 = time.time()
    print('Running time S (kxk): %f' % (t1 - t0))

#    if is_weighted:
#        S_prob = None

    return S, S_prob, S_avgweight