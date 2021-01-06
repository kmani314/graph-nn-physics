import torch.nn as nn
import torch.nn.functional as F
import torch


class GraphNetwork(nn.Module):
    # ne_dim: vertex embedding dimension
    # ee_dim: edge embedding dimension
    def __init__(
            self,
            # r_conn,
            node_dim,
            edge_dim,
            global_dim,
            mp_steps,
            proc_hidden,
            encoder_hidden_dim, decoder_hidden_dim,
            ve_dim=16, ee_dim=16, relative_encoder=True):

        super(GraphNetwork, self).__init__()
        # embeds nodes and edges into latent representations
        self._node_encoder = self._construct_mlp(
            node_dim + global_dim, encoder_hidden_dim, ve_dim)

        # relative encoder adds 3 coords and norm to each edge
        if relative_encoder:
            edge_dim += 4

        self._edge_encoder = self._construct_mlp(
            edge_dim, encoder_hidden_dim, ee_dim)

        # phi_e/phi_v, process edges and nodes into intermediate latent states
        self._edge_processor = self._construct_mlp(
            edge_dim + 2 * node_dim + global_dim,
            proc_hidden, edge_dim,
            batch_norm=True)

        self._node_processor = self._construct_mlp(
            edge_dim + node_dim + global_dim,
            proc_hidden,
            edge_dim,
            batch_norm=True)

        self.relative_encoder = relative_encoder

    def _construct_mlp(self, input, hidden, output, batch_norm=False):
        layers = nn.ModuleList([nn.Linear(input, hidden)])

        if batch_norm:
            layers.append(nn.LayerNorm(hidden))

        for i in range(0, hidden):
            layers.append(nn.Linear(hidden, hidden))

            if batch_norm:
                layers.append(nn.LayerNorm(hidden))

        layers.append(nn.Linear(hidden, output))
        return layers

    def _pad_items(self, items, length):
        out = []
        for item in items:
            out.append(F.pad(item, (0, 0, 0, length - item.size(0))))
        return out

    def _encode(self, graph_batch, n_nodes, n_edges):
        for i in graph_batch:
            global_tensor = i.globals.repeat(i.n_nodes)
            global_tensor = global_tensor.unsqueeze(1)
            i.nodes = torch.cat([i.nodes, global_tensor], dim=1)

        padded_nodes = self._pad_items([x.nodes for x in graph_batch], n_nodes)
        padded_edges = self._pad_items([x.edges for x in graph_batch], n_edges)

        batched_nodes = torch.stack(padded_nodes)
        batched_edges = torch.stack(padded_edges)

        if self.relative_encoder:
            batched_relative_edges = []
            for i, graph in enumerate(graph_batch):
                # mask positions to force the network to learn
                # positional invariance
                senders = torch.index_select(batched_nodes[i], 0, graph.senders)
                senders = torch.narrow(senders, 1, 0, 3)
                receivers = torch.index_select(batched_nodes[i], 0, graph.receivers)
                receivers = torch.narrow(receivers, 1, 0, 3)

                positional = senders - receivers
                norm = torch.norm(positional, dim=1).unsqueeze(1)
                relative_edges = torch.cat([positional, norm], dim=1)
                relative_edges = self._pad_items([relative_edges], n_edges)
                relative_edges = relative_edges[0]

                batched_relative_edges.append(torch.cat([batched_edges[i], relative_edges], dim=1))
            batched_edges = torch.stack(batched_relative_edges)

        latent_nodes = batched_nodes

        for _, l in enumerate(self._node_encoder):
            latent_nodes = l(latent_nodes)

        latent_edges = batched_edges

        for _, l in enumerate(self._edge_encoder):
            latent_edges = l(latent_edges)

        return latent_nodes, latent_edges

    # def _process(self, batched_latent_graph):

    # def forward(self, graph_batch):
        # take in a list of graphs, batch them, return new graphs
