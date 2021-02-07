# from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
import numpy as np
import torch


class Graph():
    def __init__(self, nodes):
        self.nodes = nodes
        self.n_nodes = self.nodes.size(0)
        self.node_dim = self.nodes.size(1)

        self.vels = torch.tensor([])
        self.types = torch.tensor([])
        self.pos = torch.tensor([])

        self.edges = torch.tensor([])
        self.receivers = torch.tensor([], dtype=torch.int64)
        self.senders = torch.tensor([], dtype=torch.int64)
        self.globals = torch.tensor(0.)

    def gen_edges(self, radius):
        self.radius = radius
        tree = KDTree(self.nodes)

        # self.receivers = tree.query_radius(self.nodes, r=radius)
        # print(self.receivers)
        # self.senders = torch.tensor(np.repeat(range(self.n_nodes), [len(a) for a in self.receivers]))
        # self.receivers = torch.tensor(np.concatenate(self.receivers, axis=0))
        # self.n_edges = self.senders.size(0)
        edges = tree.query_pairs(radius, output_type='ndarray')
        edges = np.concatenate((edges, np.flip(edges, 1)))

        self.receivers = torch.tensor(edges[:, 0], dtype=torch.int64)
        self.senders = torch.tensor(edges[:, 1], dtype=torch.int64)
        self.n_edges = self.receivers.size(0)
        self.edges = torch.zeros(self.n_edges, 1)

    def to(self, device):
        self.senders = self.senders.to(device)
        self.receivers = self.receivers.to(device)
        self.nodes = self.nodes.to(device)
        self.globals = self.globals.to(device)
        self.edges = self.edges.to(device)
        self.vels = self.vels.to(device)
        self.pos = self.pos.to(device)
        self.types = self.types.to(device)
