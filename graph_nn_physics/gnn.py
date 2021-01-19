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
            proc_hidden=2,
            encoder_hidden_dim=16,
            encoder_hidden=2,
            decoder_hidden_dim=16,
            decoder_hidden=2,
            dim=3,
            ve_dim=128, ee_dim=128):

        super(GraphNetwork, self).__init__()

        # embeds nodes and edges into latent representations
        self._node_encoder = self._construct_mlp(
            node_dim + global_dim, encoder_hidden_dim, encoder_hidden, ve_dim, batch_norm=True)

        self._edge_encoder = self._construct_mlp(
            edge_dim, encoder_hidden_dim, encoder_hidden, ee_dim)

        # phi_e/phi_v, process edges and nodes into intermediate latent states
        self._edge_processors = []
        self._node_processors = []

        for i in range(0, mp_steps):
            self._edge_processors.append(self._construct_mlp(
                ee_dim + 2 * ve_dim + global_dim,
                proc_hidden_dim,
                proc_hidden,
                ee_dim,
                batch_norm=True))

            self._node_processors.append(self._construct_mlp(
                ee_dim + ve_dim + global_dim,
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

        self._edge_processors = nn.ModuleList(self._edge_processors)
        self._node_processors = nn.ModuleList(self._node_processors)

        self.mp_steps = mp_steps
        self.dim = dim
        self.ve_dim = ve_dim
        self.ee_dim = ee_dim

    def _construct_mlp(self, input, hidden_dim, hidden, output, batch_norm=False):
        layers = [nn.Linear(input, hidden_dim), nn.ReLU()]

        if batch_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        for i in range(0, hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, output))

        return nn.Sequential(*layers)

    def _pad_items(self, items, length, value=0):
        out = []
        for item in items:
            out.append(F.pad(item, (0, 0, 0, length - item.size(0))))
        return out

    def _repeat_global_tensor(self, tensor, num, pad):
        global_tensor = tensor.repeat(num)
        global_tensor = global_tensor.unsqueeze(1)
        global_tensor = self._pad_items([global_tensor], pad)[0]
        return global_tensor

    # @profile
    def _encode(self, graph_batch, batch_nm, batch_em):
        for i in graph_batch:
            global_tensor = i.globals.repeat(i.n_nodes)
            global_tensor = global_tensor.unsqueeze(1)
            i.nodes = torch.cat([i.nodes, global_tensor], dim=1)

        padded_nodes = self._pad_items([x.nodes for x in graph_batch], batch_nm)
        padded_edges = self._pad_items([x.edges for x in graph_batch], batch_em)

        batched_nodes = torch.stack(padded_nodes)
        batched_edges = torch.stack(padded_edges)

        latent_nodes = self._node_encoder(batched_nodes)
        latent_edges = self._edge_encoder(batched_edges)

        for i, graph in enumerate(graph_batch):
            graph.nodes = latent_nodes[i]
            graph.node_dim = self.ve_dim
            graph.edges = latent_edges[i]
            graph.edge_dim = self.ee_dim

        return graph_batch

    # @profile
    def _phi_e(self, graph_batch, processor, batch_nm, batch_em):
        # update edge embeddings based on previous embeddings, related nodes, and globals

        # construct tensor of [e_k, v_r_k, v_s_k, u] for each edge for each graph in batch
        batched_tensor_tuple = []
        for graph in graph_batch:
            ee = graph.edges
            ve = graph.nodes
            senders = torch.index_select(ve, 0, graph.senders)
            senders = self._pad_items([senders], batch_em)[0]
            receivers = torch.index_select(ve, 0, graph.receivers)
            receivers = self._pad_items([receivers], batch_em)[0]

            global_tensor = self._repeat_global_tensor(graph.globals, graph.n_edges, batch_em)
            batched_tensor_tuple.append(torch.cat([ee, senders, receivers, global_tensor], dim=1))

        batched_tensor_tuple = torch.stack(batched_tensor_tuple)

        batched_tensor_tuple = processor(batched_tensor_tuple)

        for i, graph in enumerate(graph_batch):
            graph.edges = batched_tensor_tuple[i]

        return graph_batch

    # @profile
    def _phi_v(self, graph_batch, processor, batch_nm, batch_em):
        node_update_batch = []
        for graph in graph_batch:
            # mask out padded embedded edges
            masked_edges = torch.narrow(graph.edges, 0, 0, graph.n_edges)

            # sum receivers for every node
            receivers = graph.receivers.unsqueeze(1).repeat(1, self.ve_dim)
            zeros = torch.zeros_like(masked_edges)
            scattered_edge_states = zeros.scatter_add(0, receivers, masked_edges)
            indices = torch.unique_consecutive(receivers)
            scattered_edge_states = torch.index_select(scattered_edge_states, 0, indices)

            # this should only be necessary if there are isolated nodes with no receiver edges
            scattered_edge_states = self._pad_items([scattered_edge_states], batch_nm)[0]
            global_tensor = self._repeat_global_tensor(graph.globals, graph.n_edges, batch_nm)
            phi_v_input = torch.cat([scattered_edge_states, graph.nodes, global_tensor], dim=1)
            node_update_batch.append(phi_v_input)

        node_update_batch = torch.stack(node_update_batch)
        node_update_batch = processor(node_update_batch)

        for i, graph in enumerate(graph_batch):
            graph.nodes = node_update_batch[i]

        return graph_batch

    # @profile
    def _process(self, latent_graph_tuple, batch_nm, batch_em):
        for np, ep in zip(self._node_processors, self._edge_processors):
            latent_graph_tuple = self._phi_e(latent_graph_tuple, ep, batch_nm, batch_em)
            latent_graph_tuple = self._phi_v(latent_graph_tuple, np, batch_nm, batch_em)

        return latent_graph_tuple

    # @profile
    def _decode(self, latent_graph_tuple):
        acc = torch.stack([i.nodes for i in latent_graph_tuple])
        acc = self._decoder(acc)
        return acc

    # @profile
    def forward(self, graph_batch):
        batch_em = max([x.edges.size(0) for x in graph_batch])
        batch_nm = max([x.nodes.size(0) for x in graph_batch])

        # take in a list of graphs, batch them, return new graphs
        graph_batch = self._encode(graph_batch, batch_nm, batch_em)
        graph_batch = self._process(graph_batch, batch_nm, batch_em)
        acc = self._decode(graph_batch)

        return acc
