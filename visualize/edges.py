import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import torch
import argparse
from graph_nn_physics import Graph
from .dataset import get_hdf5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('path')
    parser.add_argument('rollout')
    parser.add_argument('radius')
    args = parser.parse_args()

    fig = plt.figure()

    data = torch.tensor(get_hdf5(args.dir, args.path, args.rollout))
    init_nodes = data[0]

    graph = Graph(init_nodes)
    graph.gen_edges(args.radius)
    print(graph.n_edges)

    G = nx.Graph()
    G.add_nodes_from([x for x in range(0, graph.n_nodes)])
    edges = torch.cat([graph.senders.unsqueeze(1), graph.receivers.unsqueeze(1)], dim=1)

    for i in edges.numpy():
        G.add_edge(*i)

    ngraph = nx.MultiDiGraph(G)

    pos = {}
    for i, p in enumerate(graph.nodes.numpy()):
        pos[i] = p

    nx.draw(G, pos=pos, node_size=25)
    plt.show()
