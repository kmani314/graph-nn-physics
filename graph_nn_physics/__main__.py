from .graph import Graph
from .gnn import GraphNetwork
import torch_scatter
import torch

if __name__ == '__main__':
    print(torch.ops.torch_scatter.cuda_version())
    device = torch.device('cpu')
    network = GraphNetwork(
        node_dim=6,
        edge_dim=1,
        global_dim=1,
        mp_steps=8,
        proc_hidden=128,
        encoder_hidden_dim=16, decoder_hidden_dim=16,
        dim=3,
        ve_dim=16, ee_dim=16, relative_encoder=True
    )

    network.to(device=device)

    nodes = torch.tensor(
        [[1, 2, 4, 3, 6, 9.0], [1, 8, 9, 6, 9, 6], [1, 1, 2, 6, 9, 6], [1, 2, 4, 6, 9, 6]], device=device)

    edges = torch.tensor([[1], [1], [1], [1], [1.0], [1]], device=device)

    senders = torch.tensor([1, 2, 3, 0, 2, 2], device=device)
    receivers = torch.tensor([2, 1, 0, 2, 1, 3], device=device)
    globals = torch.tensor([1], device=device)

    batch = [Graph(nodes, edges, senders, receivers, globals) for i in range(0, 5)]

    with torch.autograd.profiler.profile() as prof:
        batch = network(batch, 10, 10)

    print(prof.key_averages().table(sort_by="cuda_time_total"))
