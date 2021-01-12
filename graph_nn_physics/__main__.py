from .gnn import GraphNetwork
from .data import SimulationDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from torch.autograd.profiler import profile
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    args = parser.parse_args()

    # print(torch.ops.torch_scatter.cuda_version())
    device = torch.device('cpu')

    network = GraphNetwork(
        node_dim=25,
        edge_dim=1,
        global_dim=1,
        mp_steps=4,
        proc_hidden_dim=128,
        encoder_hidden_dim=16, decoder_hidden_dim=16,
        dim=3,
        ve_dim=16, ee_dim=16, relative_encoder=True
    )

    network.to(device=device)

    dataset = SimulationDataset(args.dataset, args.group, 5)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    with profile() as prof:
        for i in loader:
            out = network(i[0])
            break
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
