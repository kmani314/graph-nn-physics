import torch

def graph_preprocessor(graph, vels, types):
    attrs = graph.attrs
    pos = graph.nodes

    end_vel = vels
    types = torch.zeros_like(types)

    vels = torch.split(vels, 1)
    vels = torch.cat(vels, dim=2).squeeze()

    radius = attrs['default_connectivity_radius']

    bounds = torch.tensor(attrs['bounds'])
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    dist = torch.cat([pos - lower, upper - pos], dim=1)

    # dist = torch.clamp(dist / radius, -1, 1)

    # if using a particle type embedding, move this elsewhere
    nodes = torch.cat([vels, dist, types], dim=1).float()

    graph.attrs = attrs
    graph.types = types
    graph.gen_edges(float(radius))

    senders = torch.index_select(pos, 0, graph.senders)
    receivers = torch.index_select(pos, 0, graph.receivers)

    # positional = (senders - receivers) / graph.radius
    positional = (senders - receivers)

    norm = torch.linalg.norm(positional, dim=1, keepdims=True)

    graph.edges = torch.cat([positional, norm], dim=1)
    graph.nodes = nodes

    # not used for training, instead for inference
    norm = torch.linalg.norm(vels)

    graph.pos = pos
    graph.vels = end_vel
    # graph.dist = dist
    graph.norm = norm
    graph.vel_avg = torch.mean(vels)

    return graph

def decoder_normalizer(acc, mean, std):
    return (acc - mean) / std

def normalized_to_real(acc, mean, std):
    return acc * std + mean
