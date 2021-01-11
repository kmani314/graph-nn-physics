from scipy.spatial import KDTree
import torch

class Graph():
    def __init__(self, nodes, edges, senders, receivers, globals):
        self.nodes = nodes
        self.n_nodes = self.nodes.size(0)
        self.node_dim = self.nodes.size(1)
        self.edges = edges
        self.n_edges = self.edges.size(0)
        self.edge_dim = self.edges.size(1)
        self.receivers = receivers
        self.senders = senders
        self.globals = globals

    def gen_edges(self, radius):
        self.tree = KDTree(self.nodes)

        for i, node in enumerate(self.nodes):
            neighbors = self.tree.query_ball_point(node, radius)

            senders = torch.tensor(len(neighbors) * [i])
            self.senders = torch.cat([self.senders, senders])

            receivers = torch.tensor(neighbors)
            self.receivers = torch.cat([self.receivers, receivers])

            self.edges = torch.cat([self.edges, torch.zeros(len(neighbors))])

        self.n_edges = self.edges.shape(0)
