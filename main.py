from graph import Graph
from gnn import GraphNetwork
import torch

network = GraphNetwork(6, 1, 1, 1, 24, 128, 128)

nodes = torch.tensor(
    [[1.5, 2, 4, 3, 6, 9], [1.5, 8, 9, 6, 9, 6], [1.5, 1, 2, 6, 9, 6], [1.5, 2, 4, 6, 9, 6]])

edges = torch.tensor([[1], [1], [1], [1], [1]])

senders = torch.tensor([0, 1, 2, 3])
receivers = torch.tensor([3, 2, 1, 0])
globals = torch.tensor([1])

graphs = [Graph(nodes, edges, senders, receivers, globals) for i in range(0, 5)]

network._encode(graphs, 10, 10)
