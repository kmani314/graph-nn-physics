from scipy.spatial import KDTree
# from memory_profiler import profile
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
        else:
            self.senders = torch.tensor([], dtype=torch.int64)
            self.receivers = torch.tensor([], dtype=torch.int64)
            self.edges = torch.tensor([], dtype=torch.double)

        self.globals = globals
        if self.globals is None:
            self.globals = torch.tensor([], dtype=torch.double)

    # @profile
    def gen_edges(self, radius):
        self.radius = radius
        self.tree = KDTree(self.nodes)
        self.edges = self.edges.unsqueeze(1)

        for i, node in enumerate(self.nodes):
            neighbors = self.tree.query_ball_point(node, radius)

            senders = torch.tensor(len(neighbors) * [i])
            self.senders = torch.cat([self.senders, senders])

            receivers = torch.tensor(neighbors, dtype=torch.int64)
            self.receivers = torch.cat([self.receivers, receivers])

            self.edges = torch.cat([self.edges, torch.zeros(len(neighbors), 1)])

        self.n_edges = self.edges.size(0)
