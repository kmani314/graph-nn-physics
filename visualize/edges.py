import numpy as np
import time
import torch
import datetime
import argparse
import torch.nn.functional as F
from graphviz import Digraph
from graph_nn_physics import Graph
from .dataset import get_hdf5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('path')
    parser.add_argument('rollout')
    parser.add_argument('radius')
    parser.add_argument('particles')
    args = parser.parse_args()

    data = torch.tensor(get_hdf5(args.dir, args.path, args.rollout))
    init_nodes = data[50]

    graph = Graph(init_nodes)
    start = time.time()
    graph.gen_edges(float(args.radius))
    tstr = str(datetime.timedelta(seconds=time.time() - start) / datetime.timedelta(milliseconds=1))
    # print('K-D Tree time: {}ms'.format(tstr))

    edges = torch.cat([graph.senders.unsqueeze(1), graph.receivers.unsqueeze(1)], dim=1)
    edge_label_colors = np.random.rand(graph.n_edges, 3)

    norm = []

    edge_labels = {(*edges[i].tolist(),): norm[i] for i in range(edges.size(0))}

    pos = {}
    for i, p in enumerate(graph.nodes.numpy()):
        pos[i] = p

    # edge_props = torch.arange(0, edges.size(0), dtype=torch.float32).unsqueeze(1).repeat(1, 5)
    edge_props = torch.arange(0, edges.size(0), dtype=torch.float32)
    receivers = graph.receivers

    if edges.size(0) < graph.n_nodes:
        edge_props = F.pad(edge_props, (0, graph.n_nodes - edge_props.size(0)))
        receivers = F.pad(graph.receivers, (0, graph.n_nodes - edge_props.size(0)))

    edge_props = edge_props.unsqueeze(1)
    receivers = receivers.unsqueeze(1)
    zeros = torch.zeros_like(edge_props)
    scattered_edge_states = zeros.scatter_add(0, receivers, edge_props).squeeze().numpy()

    G = Digraph(engine='neato')

    min_x = min([p[0].item() for p in graph.nodes])
    max_x = max([p[0].item() for p in graph.nodes])
    min_y = min([p[1].item() for p in graph.nodes])
    max_y = max([p[1].item() for p in graph.nodes])
    coord_scale = 100

    for i, n in enumerate(graph.nodes):
        G.node(
            str(i),
            label='{}: {}'.format(i, scattered_edge_states[i]),
            # label='{}'.format(i),
            fontcolor='white',
            style='filled',
            color='white',
            fillcolor='sandybrown',
            shape='circle',
            pos='{},{}!'.format(
                coord_scale * (n[0] - min_x) / (max_x - min_x),
                coord_scale * (n[1] - min_y) / (max_y - min_y),
            ),
            width='1.25',
            fixedsize='true',
            pin='true',
        )

    for i, n in enumerate(edges):
        G.edge(
            str(n[0].item()),
            str(n[1].item()),
            label=str(edge_props[i].item()),
            penwidth='2'
        )

    G.save()
