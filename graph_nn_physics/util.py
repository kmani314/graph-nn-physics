import torch
from .graph import Graph

def combine_std(a, b):
    return (a ** 2 + b ** 2) ** 0.5

def sequence_postprocessor(subseq, stats):
    vels = subseq[1:] - subseq[:-1]
    vels = vels.permute(1, 0, 2)
    vels = decoder_normalizer(vels, stats['vel_mean'], stats['vel_std'])
    vels = vels.reshape(vels.shape[0], -1)
    return vels

def graph_preprocessor(position, stats, types):
    vels = sequence_postprocessor(position, stats)

    radius = stats['default_connectivity_radius']

    types = torch.zeros_like(types)
    bounds = torch.tensor(stats['bounds'])
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    pos = position[-1]
    graph = Graph(pos)

    dist = torch.cat([pos - lower, upper - pos], dim=1)
    dist = torch.clamp(dist / radius, -1., 1.)

    # if using a particle type embedding, move this elsewhere
    nodes = torch.cat([vels, dist, types], dim=1).float()

    graph.attrs = stats
    graph.types = types
    graph.gen_edges(float(radius))

    senders = torch.index_select(pos, 0, graph.senders)
    receivers = torch.index_select(pos, 0, graph.receivers)

    positional = (senders - receivers) / graph.radius

    norm = torch.linalg.norm(positional, dim=1, keepdims=True)

    graph.edges = torch.cat([positional, norm], dim=1)
    graph.nodes = nodes
    graph.pos = position

    return graph

def decoder_normalizer(acc, mean, std):
    return (acc - mean) / std

def normalized_to_real(acc, mean, std):
    return acc * std + mean
