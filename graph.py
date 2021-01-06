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
