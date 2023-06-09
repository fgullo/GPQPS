import networkx as nx
from tqdm import tqdm
import scipy
import time


def parse_sweg_partition(sweg_partition_path,sep='\t'):
    with open(sweg_partition_path) as infile:
        data = [line.strip() for line in infile.readlines()]  # supernode_id node_id0 node_id1 node_id2 ...

    partition = {int(l.split(sep)[0]): [int(k) for k in l.split(sep)[1:]] for l in data}
    node2supernode = {node: i for i, nodes in partition.items() for node in nodes}
    return partition, node2supernode


def parse_sweg_edges(sweg_edges_path, sep='\t'):
    with open(sweg_edges_path) as infile:
        edges_data = infile.read().strip().split('\n')

    edges = [(int(edge.split(sep)[0]), int(edge.split(sep)[1])) for edge in edges_data]
    return edges


def get_sweg_summary(partition, edges):
    s = nx.Graph()

    s.add_nodes_from(partition.keys())
    s.add_edges_from(edges)
    return s

