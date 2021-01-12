from gnn import GraphNetwork
from data import SimulationDataset, collate_fn
import torch
from torch.utils.data import DataLoader
# from torch.autograd.profiler import profile
import time as timer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    args = parser.parse_args()

    # print(torch.ops.torch_scatter.cuda_version())
    device = torch.device('cuda')

    network = GraphNetwork(
        node_dim=25,
        edge_dim=1,
        global_dim=1,
        mp_steps=8,
        proc_hidden_dim=64,
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
        collate_fn=lambda x: collate_fn(x, device)
    )

    # with profile(use_cuda=True, record_shapes=True) as prof:
    start = timer.time()
    for i in loader:
        out = network(i[0])
        print(out)
        break
    print('Iteration total: {}'.format(timer.time() - start))
    # print(prof.key_averages(group_by_input_shape=True))
