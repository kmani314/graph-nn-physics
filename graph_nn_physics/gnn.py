import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_memlab import profile

class GraphNetwork(nn.Module):
    def __init__(
            self,
            node_dim=6,
            edge_dim=1,
            global_dim=1,
            mp_steps=8,
            proc_hidden_dim=128,
            proc_hidden=1,
            encoder_hidden_dim=16,
            encoder_hidden=1,
            decoder_hidden_dim=16,
            decoder_hidden=1,
            dim=3,
            ve_dim=128, ee_dim=128):

        super(GraphNetwork, self).__init__()

        # embeds nodes and edges into latent representations
        self._node_encoder = self._construct_mlp(
            node_dim, encoder_hidden_dim, encoder_hidden, ve_dim, batch_norm=True)

        self._edge_encoder = self._construct_mlp(
            edge_dim, encoder_hidden_dim, encoder_hidden, ee_dim, batch_norm=True)

        # phi_e/phi_v, process edges and nodes into intermediate latent states
        self._edge_processors = []
        self._node_processors = []

        for i in range(0, mp_steps):
            self._edge_processors.append(self._construct_mlp(
                ee_dim + 2 * ve_dim,
                proc_hidden_dim,
                proc_hidden,
                ee_dim,
                batch_norm=True))

            self._node_processors.append(self._construct_mlp(
                ee_dim + ve_dim,
                proc_hidden_dim,
                proc_hidden,
                ve_dim,
                batch_norm=True))

        # the decoder goes from the latent node dimension to
        # dimensional acceleration to be used in an euler integrator
        self._decoder = self._construct_mlp(
            ve_dim,
            decoder_hidden_dim,
            decoder_hidden,
            dim
        )
        # print(self._node_encoder)

        self._edge_processors = nn.ModuleList(self._edge_processors)
        # print(self._edge_processors)
        self._node_processors = nn.ModuleList(self._node_processors)

        self.mp_steps = mp_steps
        self.dim = dim
        self.ve_dim = ve_dim
        self.ee_dim = ee_dim

    def _construct_mlp(self, input, hidden_dim, hidden, output, batch_norm=False):
        layers = [nn.Linear(input, hidden_dim), nn.ReLU()]

        for i in range(0, hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output))

        if batch_norm:
            layers.append(nn.LayerNorm(output))

        return nn.Sequential(*layers)

    # def _repeat_global_tensor(self, tensor, num, pad):
    #     global_tensor = tensor.repeat(num)
    #     global_tensor = global_tensor.unsqueeze(1)
    #     global_tensor = self._pad_items([global_tensor], pad)[0]
    #     return global_tensor

    def _encode(self, graph_batch):
        graph_batch.nodes = self._node_encoder(graph_batch.nodes)
        graph_batch.edges = self._edge_encoder(graph_batch.edges)

        graph_batch.node_dim = self.ve_dim
        graph_batch.edge_dim = self.ee_dim

        return graph_batch

    def _phi_e(self, graph_batch, processor):
        # update edge embeddings based on previous embeddings, related nodes, and globals

        # construct tensor of [e_k, v_r_k, v_s_k, u] for each edge for each graph in batch
        senders = torch.index_select(graph_batch.nodes, 0, graph_batch.senders)
        receivers = torch.index_select(graph_batch.nodes, 0, graph_batch.receivers)

        # global_tensor = self._repeat_global_tensor(graph_batch.globals, graph_batch.n_edges, graph_batch.n_edges)
        graph_batch.edges = torch.cat([graph_batch.edges, senders, receivers], dim=1)

        graph_batch.edges = processor(graph_batch.edges)

        return graph_batch

    def _phi_v(self, graph_batch, processor):
        receivers = graph_batch.receivers.unsqueeze(1).repeat(1, self.ve_dim)
        zeros = torch.zeros_like(graph_batch.edges)
        scattered_edge_states = zeros.scatter_add(0, receivers, graph_batch.edges)
        scattered_edge_states = scattered_edge_states[:graph_batch.n_nodes]

        # global_tensor = self._repeat_global_tensor(graph_batch.globals, graph_batch.n_nodes, graph_batch.n_nodes)
        graph_batch.nodes = torch.cat([graph_batch.nodes, scattered_edge_states], dim=1)

        graph_batch.nodes = processor(graph_batch.nodes)

        return graph_batch

    def _process(self, latent_graph_tuple):
        # for np, ep in zip(self._node_processors, self._edge_processors):
        for _ in range(self.mp_steps):
            np = self._node_processors[0]
            ep = self._edge_processors[0]

            prev_graph = latent_graph_tuple

            latent_graph_tuple = self._phi_e(latent_graph_tuple, ep)
            latent_graph_tuple = self._phi_v(latent_graph_tuple, np)

            # residual connections
            latent_graph_tuple.nodes += prev_graph.nodes
            latent_graph_tuple.edges += prev_graph.edges

        return latent_graph_tuple

    def _decode(self, latent_graph_tuple):
        return self._decoder(latent_graph_tuple.nodes)

    def forward(self, graph_batch):
        # take in a list of graphs, batch them, return new graphs
        graph_batch = self._encode(graph_batch)
        graph_batch = self._process(graph_batch)
        return self._decode(graph_batch)
