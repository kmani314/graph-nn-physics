from .graph import Graph
import torch

def graph_preprocessor(graph, vels, types):
    attrs = graph.attrs
    pos = graph.nodes

    end_vel = vels
    squashed = end_vel.view(end_vel.size(0), end_vel.size(1) * end_vel.size(2))

    radius = attrs['default_connectivity_radius']

    lower = torch.tensor(attrs['bounds'][:, 0])
    upper = torch.tensor(attrs['bounds'][:, 1])

    dist = torch.cat([pos - lower, upper - pos], dim=1)
    dist = torch.clamp(dist / radius, -1, 1)

    # if using a particle type embedding, move this elsewhere
    nodes = torch.cat([squashed, dist, types], dim=1).float()

    graph = Graph(pos)
    graph.attrs = attrs
    graph.types = types
    graph.gen_edges(float(radius))

    senders = torch.index_select(graph.nodes, 0, graph.senders)
    senders = torch.narrow(senders, 1, 0, attrs['dim'])
    receivers = torch.index_select(graph.nodes, 0, graph.receivers)
    receivers = torch.narrow(receivers, 1, 0, attrs['dim'])

    positional = (senders - receivers) / graph.radius
    norm = torch.norm(positional, dim=1).unsqueeze(1)
    graph.edges = torch.cat([positional, norm], dim=1)
    graph.nodes = nodes

    # not used for training, instead for inference
    graph.pos = pos
    graph.vels = end_vel

    return graph

def decoder_normalizer(acc, mean, std):
    return (acc - mean) / std

def normalized_to_real(acc, mean, std):
    return acc * std + mean
