import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class GraphNetwork(nn.Module):
    # ne_dim: vertex embedding dimension
    # ee_dim: edge embedding dimension
    def __init__(
            self,
            # r_conn,
            node_dim=6,
            edge_dim=1,
            global_dim=1,
            mp_steps=16,
            proc_hidden=256,
            encoder_hidden_dim=128, decoder_hidden_dim=128,
            dim=3,
            max_node=10,
            max_edge=10,
            ve_dim=16, ee_dim=16, relative_encoder=True):

        super(GraphNetwork, self).__init__()
        # embeds nodes and edges into latent representations
        self._node_encoder = self._construct_mlp(
            node_dim + global_dim, encoder_hidden_dim, ve_dim)

        # relative encoder adds [dim] coords and norm to each edge
        if relative_encoder:
            edge_dim += dim + 1

        self._edge_encoder = self._construct_mlp(
            edge_dim, encoder_hidden_dim, ee_dim)

        # phi_e/phi_v, process edges and nodes into intermediate latent states
        self._edge_processor = self._construct_mlp(
            ee_dim + 2 * ve_dim + global_dim,
            proc_hidden, ee_dim,
            batch_norm=True)

        self._node_processor = self._construct_mlp(
            ee_dim + ve_dim + global_dim,
            proc_hidden,
            ve_dim,
            batch_norm=True)

        # as written in the paper, the decoder goes from the latent node dimension to
        # dimensional acceleration to be used in an euler integrator
        self._decoder = self._construct_mlp(
            ve_dim,
            decoder_hidden_dim,
            dim
        )

        self.relative_encoder = relative_encoder
        self.dim = dim
        self.ve_dim = ve_dim
        self.ee_dim = ee_dim
        self.max_node = max_node
        self.max_edge = max_edge

    def _construct_mlp(self, input, hidden, output, batch_norm=False):
        layers = [nn.Linear(input, hidden), nn.ReLU()]

        if batch_norm:
            layers.append(nn.LayerNorm(hidden))

        for i in range(0, hidden):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.LayerNorm(hidden))

        layers.append(nn.Linear(hidden, output))
        layers.append(nn.ReLU())
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

    def _encode(self, graph_batch):
        for i in graph_batch:
            global_tensor = i.globals.repeat(i.n_nodes)
            global_tensor = global_tensor.unsqueeze(1)
            i.nodes = torch.cat([i.nodes, global_tensor], dim=1)

        padded_nodes = self._pad_items([x.nodes for x in graph_batch], self.max_node)
        padded_edges = self._pad_items([x.edges for x in graph_batch], self.max_edge)

        batched_nodes = torch.stack(padded_nodes)
        batched_edges = torch.stack(padded_edges)

        # this part is difficult to batch but it doesn't matter because it's fast
        if self.relative_encoder:
            batched_relative_edges = []
            for i, graph in enumerate(graph_batch):
                # mask positions to force the network to learn
                # positional invariance
                senders = torch.index_select(batched_nodes[i], 0, graph.senders)
                senders = torch.narrow(senders, 1, 0, self.dim)
                receivers = torch.index_select(batched_nodes[i], 0, graph.receivers)
                receivers = torch.narrow(receivers, 1, 0, self.dim)

                positional = senders - receivers
                norm = torch.norm(positional, dim=1).unsqueeze(1)
                relative_edges = torch.cat([positional, norm], dim=1)
                relative_edges = self._pad_items([relative_edges], self.max_edge)[0]

                batched_relative_edges.append(torch.cat([batched_edges[i], relative_edges], dim=1))
            batched_edges = torch.stack(batched_relative_edges)

        latent_nodes = self._node_encoder(batched_nodes)
        latent_edges = self._edge_encoder(batched_edges)

        for i, graph in enumerate(graph_batch):
            graph.nodes = latent_nodes[i]
            graph.node_dim = self.ve_dim
            graph.edges = latent_edges[i]
            graph.edge_dim = self.ee_dim

        return graph_batch

    def _phi_e(self, graph_batch):
        # update edge embeddings based on previous embeddings, related nodes, and globals

        # construct tensor of [e_k, v_r_k, v_s_k, u] for each edge for each graph in batch
        batched_tensor_tuple = []
        for graph in graph_batch:
            ee = graph.edges
            ve = graph.nodes
            senders = torch.index_select(ve, 0, graph.senders)
            senders = self._pad_items([senders], self.max_edge)[0]
            receivers = torch.index_select(ve, 0, graph.receivers)
            receivers = self._pad_items([receivers], self.max_edge)[0]

            global_tensor = self._repeat_global_tensor(graph.globals, graph.n_edges, self.max_edge)
            batched_tensor_tuple.append(torch.cat([ee, senders, receivers, global_tensor], dim=1))

        batched_tensor_tuple = torch.stack(batched_tensor_tuple)

        next_latent_state = self._edge_processor(batched_tensor_tuple)

        for i, graph in enumerate(graph_batch):
            graph.edges = next_latent_state[i]

        return graph_batch

    def _phi_v(self, graph_batch):
        node_update_batch = []
        for graph in graph_batch:
            # mask out padded embedded edges
            masked_edges = torch.narrow(graph.edges, 0, 0, graph.n_edges)

            # sum receivers for every node
            scattered_edge_states = scatter(masked_edges, graph.receivers, dim=0)

            # this should only be necessary if there are isolated nodes with no receiver edges
            scattered_edge_states = self._pad_items([scattered_edge_states], self.max_node)[0]
            global_tensor = self._repeat_global_tensor(graph.globals, graph.n_edges, self.max_edge)
            phi_v_input = torch.cat([scattered_edge_states, graph.nodes, global_tensor], dim=1)
            node_update_batch.append(phi_v_input)

        next_latent_state = torch.stack(node_update_batch)
        next_latent_state = self._node_processor(next_latent_state)

        for i, graph in enumerate(graph_batch):
            graph.nodes = next_latent_state[i]

        return graph_batch

    # def _process(self, latent_graph_tuple):

    # def forward(self, graph_batch, batch_node_max, batch_edge_max):
        # take in a list of graphs, batch them, return new graphs
