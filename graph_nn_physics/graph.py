from scipy.spatial import KDTree
import numpy as np
import torch

class Graph():
    def __init__(self, nodes, edges=None, senders=None, receivers=None, globals=None):
        self.nodes = nodes
        self.n_nodes = self.nodes.size(0)
        self.node_dim = self.nodes.size(1)

        self.edges = edges
        self.receivers = receivers
        self.senders = senders

        if edges is not None:
            self.n_edges = self.edges.size(0)
            self.edge_dim = self.edges.size(1)

        self.globals = globals
        if self.globals is None:
            self.globals = torch.tensor([], dtype=torch.double)

    def gen_edges(self, radius):
        self.radius = radius
        self.tree = KDTree(self.nodes)

        edges = self.tree.query_pairs(radius, output_type='ndarray')
        edges = np.concatenate((edges, np.flip(edges, 1)))

        self.receivers = torch.tensor(edges[:, 0], dtype=torch.int64)
        self.senders = torch.tensor(edges[:, 1], dtype=torch.int64)
        self.n_edges = self.receivers.size(0)
        self.edges = torch.zeros(self.n_edges, 1)
