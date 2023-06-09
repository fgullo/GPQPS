import time
import scipy
import scipy.special
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
import networkx.algorithms.community as nx_comm
import pickle

import src.riondato as riondato

def load_original_graph(input_graph_path, sep=',', w='weight'):
    with open(input_graph_path) as infile:
        edges = [tuple([float(k) for k in line.strip().split(sep)]) for line in infile.readlines() if not line.startswith('#')]

    if len(edges[0]) == 3:
        edges = [(int(k[0]), int(k[1]), {w: k[2]}) for k in edges]
    else:
        edges = [(int(k[0]), int(k[1])) for k in edges]
        
    nodes = set([node for edge in edges for node in [edge[0], edge[1]]])

    G = nx.Graph()

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    node2index = {n: i for i, n in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, node2index)
    
    return G


def get_largest_cc(G):
    ccs = nx.weakly_connected_components(G) if nx.is_directed(G) else nx.connected_components(G)
    largest_cc = max(ccs, key=len)
    G = G.subgraph(largest_cc).copy()
    
    node2index = {n: i for i, n in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, node2index)
    
    return G


def get_size_largest_cc_and_number_cc(G):
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_cc = G.subgraph(ccs[0])
    
    return largest_cc.number_of_nodes(), len(ccs)


def compute_centrality_metrics(scores_G, scores_sG, tot_relevant, tot_retrieved_list, print_output=True):
    results = {}
    sorted_G = sorted(scores_G.items(), key=lambda x: -x[1])
    max_value_G = sorted_G[tot_relevant-1][1]
    relevant_set = [k[0] for k in sorted_G if k[1] >= max_value_G]
    for tot_retrieved in tot_retrieved_list:
        sorted_sG = sorted(scores_sG.items(), key=lambda x: -x[1])
        max_value_sG = sorted_sG[tot_retrieved-1][1]
        retrieved_set = [k[0] for k in sorted_sG if k[1] >= max_value_sG]

        p = precision(relevant_set, retrieved_set)
        r = recall(relevant_set, retrieved_set)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        results[(tot_relevant,tot_retrieved)] = [p,r,f1]
        
        if print_output:
            print("target retrieved: {}".format(tot_retrieved))
            print("tot relevant: {}".format(len(relevant_set)))
            print("tot retrieved: {}".format(len(retrieved_set)))
            print("Precision: {:.2f}".format(p))
            print("Recall: {:.2f}".format(r))
            print('F1-score: {:.2f}\n'.format(f1))
        
    return results       
        
def compute_core_metrics(core_G, core_sG, tot_core_relevant, tot_core_retrieved_list, print_output=True):
    results = {}
    max_core_G = max(core_G.values())
    relevant_set = [k for k, v in core_G.items() if v >= max_core_G - tot_core_relevant + 1]
    for tot_core_retrieved in tot_core_retrieved_list:
        sorted_core_sG_values = sorted(set(core_sG.values()), reverse=True)
        tot_core_retrieved_actual = min(tot_core_retrieved,len(sorted_core_sG_values))
        flag_value = sorted_core_sG_values[tot_core_retrieved_actual - 1]
        retrieved_set = [k for k, v in core_sG.items() if v >= flag_value]

        p = precision(relevant_set, retrieved_set)
        r = recall(relevant_set, retrieved_set)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        results[(tot_core_relevant,tot_core_retrieved)] = [p,r,f1]
        
        if print_output:
            print("#retrieved cores (target): {}, #retrieved cores (actual): {}".format(tot_core_retrieved,tot_core_retrieved_actual))
            print("tot relevant: {}".format(len(relevant_set)))
            print("tot retrieved: {}".format(len(retrieved_set)))
            print("Precision: {:.2f}".format(p))
            print("Recall: {:.2f}".format(r))
            print('F1-score: {:.2f}\n'.format(f1))
        
    return results
        

def compute_core_metrics_probabilistic(core_G, core_worlds, tot_core_relevant, tot_core_retrieved_list, print_output=True):
    results = {}
    max_core_G = max(core_G.values())
    relevant_set = [k for k, v in core_G.items() if v >= max_core_G - tot_core_relevant + 1]

    for tot_core_retrieved in tot_core_retrieved_list:
        retrieved_set = core_aggregation_intersection(core_worlds, tot_core_retrieved)
        p = precision(relevant_set, retrieved_set)
        r = recall(relevant_set, retrieved_set)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        results[(tot_core_relevant,tot_core_retrieved)] = [p,r,f1]
        
        if print_output:
            print("#retrieved cores (target): {}".format(tot_core_retrieved))
            print("tot relevant: {}".format(len(relevant_set)))
            print("tot retrieved: {}".format(len(retrieved_set)))
            print("Precision: {:.2f}".format(p))
            print("Recall: {:.2f}".format(r))
            print('F1-score: {:.2f}\n'.format(f1))

    return results


def core_aggregation_intersection(cores_per_world, ncores):
    output = None
    for all_cores in cores_per_world:
        cores = get_ncores(all_cores, ncores)
        output = cores if not output else output.intersection(cores)
    return output

        
def get_ncores(all_cores, ncores):
    sorted_core_values = sorted(set(all_cores.values()), reverse=True)
    ncores_actual = min(ncores, len(sorted_core_values))
    flag_value = sorted_core_values[ncores_actual - 1]
    cores = {k for k, v in all_cores.items() if v >= flag_value}    
    return cores


def precision(relevant_set, retrieved_set):
    return len(set(relevant_set).intersection(retrieved_set)) / len(retrieved_set) if len(retrieved_set) > 0 else 0


def recall(relevant_set, retrieved_set):
    return len(set(relevant_set).intersection(retrieved_set)) / len(relevant_set) if len(relevant_set) > 0 else 0


def assign_centrality_score(scores_s, node2supernode, partition, centrality_type, w='weight', node_set=None):
    scores_sG = {}
    if not node_set:
        for node_id, supernode_id in node2supernode.items():
            scores_sG[node_id] = scores_s[supernode_id]
        
            if centrality_type in [nx.pagerank, nx.closeness_centrality]:
                scores_sG[node_id] = scores_s[supernode_id] / len(partition[supernode_id])
    else:
        for node_id in node_set:
            supernode_id = node2supernode[node_id]
            scores_sG[node_id] = scores_s[supernode_id]
            
            if centrality_type in [nx.pagerank, nx.closeness_centrality]:
                scores_sG[node_id] = scores_s[supernode_id] / len(partition[supernode_id])
            
    return scores_sG


def assign_core_number(core_s, node2supernode):
    core_sG = {}
    for node_id, supernode_id in node2supernode.items():
        core_sG[node_id] = core_s[supernode_id]
            
    return core_sG


# Assign to each node the community of the supernode it belongs to
def assign_community(community_s, partition):
    community_sG = []
    for i, community_set in enumerate(community_s):
        tmp = []
        for supernode_id in community_set:
            tmp.extend([node_id for node_id in partition[supernode_id]])
        community_sG.append(tmp)
    return community_sG


def world(graph_type, matrix, aw_matrix, prob_matrix):
    #s = nx.from_numpy_matrix(np.matrix(matrix.A), create_using=graph_type)
    #s_aw = nx.from_numpy_matrix(np.matrix(aw_matrix.A), create_using=graph_type)
    s = nx.from_numpy_array(np.matrix(matrix.A), create_using=graph_type)
    s_aw = nx.from_numpy_array(np.matrix(aw_matrix.A), create_using=graph_type)
    world = s.copy()
    world_aw = s_aw.copy()
    if prob_matrix != None:
        #s_prob = nx.from_numpy_matrix(np.matrix(prob_matrix.A), create_using=graph_type)
        s_prob = nx.from_numpy_array(np.matrix(prob_matrix.A), create_using=graph_type)
        for edge in s_prob.edges(data=True):
            sample = random.uniform(0,1)
            if edge[-1]['weight'] < sample:
                world.remove_edge(edge[0], edge[1])
                world_aw.remove_edge(edge[0], edge[1])
    else:
        for edge in s.edges(data=True):
            sample = random.uniform(0,1)
            if edge[-1]['weight'] < sample:
                world.remove_edge(edge[0], edge[1])
                world_aw.remove_edge(edge[0], edge[1])
    return world, world_aw


def partition_aggregation(partitions, n_nodes, max_iter, seed, mini_batch=True):
    data_matrix = [[] for i in range(n_nodes)]
    kavg = 0
    for partition in partitions:
        kavg += len(partition)
        for i, cluster in enumerate(partition):
            for u in cluster:
                data_matrix[u].append(i)

    kavg = round(kavg/len(partitions))
    
    if not mini_batch:
        kmeans = KMeans(n_clusters=kavg, init='k-means++', algorithm='auto', max_iter=max_iter, random_state=seed).fit(data_matrix)
    else:
        kmeans = MiniBatchKMeans(n_clusters=kavg, init='k-means++', n_init=3, max_iter=max_iter, random_state=seed).fit(data_matrix)

    # cluster_id : list of id_nodes in it
    #agg_partition = {i: np.where(kmeans.labels_ == i)[0] for i in range(kavg)}
    agg_partition = [[] for i in range(kavg)]
    u = 0
    for c in kmeans.labels_:
        agg_partition[c].append(u)
        u += 1
    
    return agg_partition


# average cluster coefficient on possible worlds
def avg_cluster_coefficient_probabilistic(worlds, weighted, use_tqdm=True):
    avg_clustering = 0
    for W in tqdm(worlds, disable=not use_tqdm):
        c = nx.average_clustering(W, weight=weighted)
        avg_clustering += c
    return avg_clustering/len(worlds)


# modularity on aggregate partition (over the possible worlds)
def modularity_probabilistic_agg(G, worlds, partition, weighted, n_nodes, max_iter, seed, mini_batch=True, use_tqdm=True):
    communities = []
    for W in tqdm(worlds, disable=not use_tqdm):
        #community_W = nx_comm.greedy_modularity_communities(W, weight=weighted)
        community_W = nx_comm.louvain_communities(W, weight=weighted)
        community_WG = assign_community(community_W, partition)
        communities.append(community_WG)

    agg_partition = partition_aggregation(communities, n_nodes, max_iter, seed, mini_batch)
    modularity = nx_comm.modularity(G, agg_partition, weight=weighted)
    return modularity


def centrality_summary(s, partition, node2supernode, centrality_type, w, n_nodes, node_set, print_output=True):
    t0 = time.time()
    if centrality_type == nx.closeness_centrality:
        if not node_set:
            scores_s = centrality_type(s, distance=w)
        else:
            scores_s = {}
            print("COMPUTING CLOSENESS CENTRALITY FOR SAMPLED NODES")
            super_node_set = {node2supernode[sampled_node] for sampled_node in node_set}
            for sampled_super_node in super_node_set:
                scores_s[sampled_super_node] = centrality_type(s, u=sampled_super_node, distance=w)
    else:
        scores_s = centrality_type(s, weight=w)
    # rank on original graph G computed from summary s
    scores_sG = assign_centrality_score(scores_s, node2supernode, partition, centrality_type, w, node_set)
    t1 = time.time()
    
    if print_output:
        weighted = 'unweighted' if not w else 'weighted'
        print('S -', weighted, '- Centrality:', centrality_type)
        print('Avg:', round(sum([scores_sG[u] for u in range(n_nodes)])/len(scores_sG),5), ' Std-dev:', round(np.std([scores_sG[u] for u in range(n_nodes)]),6))
        print('Running time:', round(t1-t0,5))
        print()
    
    return scores_sG


def score_set_aggregation_intersection(scores, topk):
    sorted_scores = sorted(scores[0].items(), key=lambda x: -x[1])
    max_value = sorted_scores[topk-1][1]
    agg_set = {k[0] for k in sorted_scores if k[1] >= max_value}
    for i in range(1,len(scores)):
        sorted_scores = sorted(scores[i].items(), key=lambda x: -x[1])
        max_value = sorted_scores[topk-1][1]
        i_set = {k[0] for k in sorted_scores if k[1] >= max_value}
        agg_set = agg_set.intersection(i_set) 
    return agg_set


def compute_centrality_metrics_probabilistic(scores_G, scores_sG, tot_relevant, tot_retrieved_list, print_output=True):
    results = {}
    sorted_G = sorted(scores_G.items(), key=lambda x: -x[1])
    max_value_G = sorted_G[tot_relevant-1][1]
    relevant_set = [k[0] for k in sorted_G if k[1] >= max_value_G]
    for tot_retrieved in tot_retrieved_list:
        retrieved_set_overall = score_set_aggregation_intersection(scores_sG, tot_retrieved)

        p = precision(relevant_set, retrieved_set_overall)
        r = recall(relevant_set, retrieved_set_overall)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        results[(tot_relevant,tot_retrieved)] = [p,r,f1]
        
        if print_output:
            print("tot relevant: {}".format(len(relevant_set)))
            print("tot retrieved: {}".format(len(retrieved_set_overall)))
            print("Precision: {:.2f}".format(p))
            print("Recall: {:.2f}".format(r))
            print('F1-score: {:.2f}\n'.format(f1))
    
    return results

        
def centrality_summary_probabilistic(ss, partition, node2supernode, centrality_type, w, world_type, node_set, print_output=True):
    t0 = time.time()
    all_scores_sG = []
    for s in tqdm(ss):
        if centrality_type == nx.closeness_centrality:
            if not node_set:
                scores_s = centrality_type(s, distance=w)
            else:
                scores_s = {}
                print("COMPUTING CLOSENESS CENTRALITY FOR SAMPLED NODES")
                super_node_set = {node2supernode[sampled_node] for sampled_node in node_set}
                for sampled_super_node in super_node_set:
                    scores_s[sampled_super_node] = centrality_type(s, u=sampled_super_node, distance=w)
        else:
            scores_s = centrality_type(s, weight=w)
        # rank on original graph G computed from summary s
        scores_sG = assign_centrality_score(scores_s, node2supernode, partition, centrality_type, w, node_set)
        all_scores_sG.append(scores_sG)
    t1 = time.time()
    
    if print_output:
        print('Worlds -', world_type, '- Centrality:', centrality_type)
        print('Running time:', round(t1-t0,5))
        print()
    
    return all_scores_sG
        
        
def assign_core_number_probabilistic(cores_per_world, node2supernode, use_tqdm=True):
    coresWG_per_world = []
    for all_cores_W in tqdm(cores_per_world, disable=not use_tqdm):
        all_cores_WG = assign_core_number(all_cores_W, node2supernode)
        coresWG_per_world.append(all_cores_WG)
    return coresWG_per_world


def dump_summary_S2L(graph_path, summary_path, seed, k, max_iter, is_directed, is_weighted, sep):
    print('loading input graph')
    G = load_original_graph(graph_path, sep)
    
    print('generating summary')
    S, S_prob, partition, node2supernode, S_avgweight = riondato.get_summary(G, G.number_of_nodes(), k, max_iter, 
                                                            is_directed, is_weighted, seed)
    
    print('dumping summary')
    summary_obj = (S, S_prob, partition, node2supernode, S_avgweight)
    with open(summary_path, 'wb') as summary_file:
        pickle.dump(summary_obj, summary_file, pickle.HIGHEST_PROTOCOL)
        
        
def dump_summary_S2L_riondato_code(graph_path, riondato_path, summary_path, is_directed, is_weighted, sep):
    print('loading input graph')
    G = load_original_graph(graph_path, sep)
    
    print('loading output of Riondato code')
    with open(riondato_path, 'r') as infile:
        clusters = infile.readlines()
    clusters = [c.replace('\n', '').strip().split(' ') for c in clusters]

    print('generating summary')
    partition = {i: [int(c) for c in cluster] for i, cluster in enumerate(clusters)}
    node2supernode = {node: i for i, nodes in partition.items() for node in nodes}
    
    A = riondato.get_adjacency_matrix(G, G.number_of_nodes(), is_directed, is_weighted)
    S, S_prob, S_avgweight = riondato.density_matrix(A, G, partition, node2supernode, is_directed, is_weighted)
    
    print('dumping summary')
    summary_obj = (S, S_prob, partition, node2supernode, S_avgweight)
    with open(summary_path, 'wb') as summary_file:
        pickle.dump(summary_obj, summary_file, pickle.HIGHEST_PROTOCOL)
        
        
def save_GCC(input_graph_path, sep=',', w='weight'):
    # input graph
    G = load_original_graph(input_graph_path, sep=sep)
    G = get_largest_cc(G)
    
    basepath, extension = input_graph_path.split('.')
    output_graph_path = basepath + '-GCC.' + extension
    with open(output_graph_path, 'w') as outfile:
        outfile.write('')
            
    for edge in G.edges():
        line = [str(edge[0]), str(edge[1])]
        if nx.is_weighted(G):
            line.append(G[edge][w])

        with open(output_graph_path, 'a') as outfile:
            outfile.write(sep.join(line) + '\n')