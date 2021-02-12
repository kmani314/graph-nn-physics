from .util import graph_preprocessor, decoder_normalizer, combine_std
from torch.utils.data import Dataset
from .noise import gen_noise
from .graph import Graph
import numpy as np
import torch
import h5py

def collate_fn(batch):
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

    # at inference time (batch_size = 1)
    graph.attrs = batch[0][0].attrs
    graph.pos = batch[0][0].pos
    graph.types = batch[0][0].types

    gt = [x[1] for x in batch]
    gt = torch.cat(gt, dim=0)

    return (graph, gt)

class SimulationDataset(Dataset):
    def __init__(self, file, group, vel_seq, noise_std, normalization=True, only_first=False):
        self.dataset = h5py.File(file, 'r', swmr=True, libver='latest')[group]
        self.only_first = only_first
        self.noise_std = noise_std
        self.rollouts = len(self.dataset['positions'].keys())
        self.vel_seq = vel_seq
        self.normalization = normalization

    def __len__(self):
        self.len = self.rollouts * (self.dataset['positions'].get('0').shape[0] - self.vel_seq)
        return self.len

    def __getitem__(self, idx):
        # idx = np.random.randint(0, high=self.len)

        num = str(idx // (self.dataset['positions']['0'].shape[0] - self.vel_seq))
        begin = idx % (self.dataset['positions']['0'].shape[0] - self.vel_seq - 2)

        if self.only_first:
            begin = 0

        particle_types = self.dataset['particle_types'].get(num)
        arr = np.zeros(particle_types.shape, dtype='double')
        particle_types.read_direct(arr)
        types = torch.tensor(arr).unsqueeze(1).float()

        positions = self.dataset['positions'].get(num)
        arr = np.zeros((self.vel_seq + 1, positions.shape[1], positions.shape[2]), dtype='double')
        positions.read_direct(arr, np.s_[begin: begin + self.vel_seq + 1])
        rollout = torch.tensor(arr).float()

        subseq = rollout[:self.vel_seq]
        noise = gen_noise(subseq, self.noise_std)
        subseq += noise

        previous_vel = subseq[-1] - subseq[-2]
        next_vel = (rollout[self.vel_seq] + noise[-1]) - subseq[-1]

        gt = next_vel - previous_vel
        attrs = self.dataset.attrs

        acc_std = torch.tensor(combine_std(attrs['acc_std'], self.noise_std))
        amean = torch.tensor(attrs['acc_mean'])
        gt = decoder_normalizer(gt, amean, acc_std)

        graph = graph_preprocessor(subseq, attrs, types)
        return [graph, gt.float()]
