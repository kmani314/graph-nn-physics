from graph import Graph
from gnn import GraphNetwork
import torch

network = GraphNetwork(
    node_dim=6,
    edge_dim=1,
    global_dim=1,
    mp_steps=16,
    proc_hidden=256,
    encoder_hidden_dim=128, decoder_hidden_dim=128,
    dim=3,
    max_node=10,
    max_edge=10,
    ve_dim=16, ee_dim=16, relative_encoder=True
)

nodes = torch.tensor(
    [[1.5, 2, 4, 3, 6, 9], [1.5, 8, 9, 6, 9, 6], [1.5, 1, 2, 6, 9, 6], [1.5, 2, 4, 6, 9, 6]])

edges = torch.tensor([[1], [1], [1], [1], [1.0]])

senders = torch.tensor([1, 2, 3])
receivers = torch.tensor([2, 1, 0])
globals = torch.tensor([1])

batch = [Graph(nodes, edges, senders, receivers, globals) for i in range(0, 5)]

batch = network._encode(batch)
batch = network._phi_e(batch)
batch = network._phi_v(batch)
