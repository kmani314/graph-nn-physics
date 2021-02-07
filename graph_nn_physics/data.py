from .util import graph_preprocessor, decoder_normalizer, combine_std, sequence_postprocessor
from torch.utils.data import Dataset
from .noise import gen_noise
from .graph import Graph
import numpy as np
import torch
import h5py

def collate_fn(batch, device):
    lengths = torch.tensor([x[0].n_nodes for x in batch])
    lengths = torch.cat([torch.tensor([0]), lengths[:-1]], dim=0)
    offsets = torch.cumsum(lengths, dim=0)

    nodes = []
    edges = []
    senders = []
    receivers = []

    for i, pair in enumerate(batch):
        graph = pair[0]
        nodes.append(graph.nodes)
        edges.append(graph.edges)
        senders.append(graph.senders + offsets[i])
        receivers.append(graph.receivers + offsets[i])

    graph = Graph(torch.cat(nodes, dim=0))
    graph.edges = torch.cat(edges, dim=0)
    graph.senders = torch.cat(senders, dim=0)
    graph.receivers = torch.cat(receivers, dim=0)
    graph.to(device=device)

    # at inference time (batch_size = 1)
    graph.attrs = batch[0][0].attrs
    graph.pos = batch[0][0].pos
    graph.types = batch[0][0].types

    gt = [x[1].to(device=device) for x in batch]
    gt = torch.cat(gt, dim=0)

    return (graph, gt)

class SimulationDataset(Dataset):
    def __init__(self, file, group, vel_seq, noise_std, normalization=True):
        self.file = h5py.File(file, 'r')
        self.group = group
        self.noise_std = noise_std
        self.rollouts = len(self.file[self.group]['positions'].keys())
        self.vel_seq = vel_seq
        self.normalization = normalization

    def __len__(self):
        return self.rollouts * (self.file[self.group]['positions'].get('0').shape[0] - self.vel_seq)

    def __getitem__(self, idx):
        dataset = self.file[self.group]

        num = str(idx // (dataset['positions'].get('0').shape[0] - self.vel_seq))
        begin = idx % (dataset['positions'].get('0').shape[0] - self.vel_seq - 2)

        positions = dataset['positions'].get(num)
        particle_types = dataset['particle_types'].get(num)

        arr = np.zeros(positions.shape, dtype='double')
        positions.read_direct(arr)
        rollout = torch.tensor(arr).float()

        arr = np.zeros(particle_types.shape, dtype='double')
        particle_types.read_direct(arr)
        types = torch.tensor(arr).unsqueeze(1).float()

        subseq = rollout[begin: begin + self.vel_seq]
        noise = gen_noise(subseq, self.noise_std)
        subseq += noise

        previous_vel = subseq[-1] - subseq[-2]

        t_idx = begin + self.vel_seq
        next_vel = (rollout[t_idx] + noise[-1]) - subseq[-1]

        gt = next_vel - previous_vel
        attrs = self.file[self.group].attrs

        acc_std = torch.tensor(combine_std(attrs['acc_std'], self.noise_std))
        amean = torch.tensor(attrs['acc_mean'])
        gt = decoder_normalizer(gt, amean, acc_std)

        graph = graph_preprocessor(subseq, attrs, types)
        return [graph, gt.float()]
