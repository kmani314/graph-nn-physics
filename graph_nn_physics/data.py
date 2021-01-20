from torch.utils.data import Dataset
from numpy import random
from .graph import Graph
from .util import graph_preprocessor, decoder_normalizer
import numpy as np
import torch
import h5py

def collate_fn(batch, device):
    graphs = [x[0] for x in batch]
    for i in graphs:
        i.to(device)

    gt = [x[1].to(device=device) for x in batch]

    return (graphs, gt)

class SimulationDataset(Dataset):
    def __init__(self, file, group, vel_seq, normalization=True):
        self.file = h5py.File(file, 'r')
        self.group = group
        self.rollouts = len(self.file[self.group]['positions'].keys())
        self.vel_seq = vel_seq
        self.normalization = normalization

    def __len__(self):
        sum = 0
        for i in range(0, self.rollouts):
            positions = self.file[self.group]['positions'].get(str(i))
            sum += positions.shape[0]
        return sum

    def __getitem__(self, _):
        num = random.randint(self.rollouts)

        positions = self.file[self.group]['positions'].get(str(num))
        particle_types = self.file[self.group]['particle_types'].get(str(num))

        arr = np.zeros(positions.shape, dtype='float')
        positions.read_direct(arr)
        rollout = torch.tensor(arr).float()

        arr = np.zeros(particle_types.shape, dtype='float')
        particle_types.read_direct(arr)
        types = torch.tensor(arr).unsqueeze(1).float()

        begin = random.randint(rollout.size(0) - self.vel_seq - 1) + 1
        vels = rollout[begin:begin + self.vel_seq] - rollout[begin - 1: begin + (self.vel_seq - 1)]

        attrs = self.file[self.group].attrs

        # get next acceleration as next vel - last vel given to network
        idx = begin + self.vel_seq
        gt = rollout[idx] - 2 * rollout[idx - 1] + rollout[idx - 2]

        if self.normalization:
            mean = torch.tensor(attrs['vel_mean'])
            std = torch.tensor(attrs['vel_std'])
            vels = decoder_normalizer(vels, mean, std)

            amean = torch.tensor(attrs['acc_mean'])
            astd = torch.tensor(attrs['acc_std'])
            gt = decoder_normalizer(gt, amean, astd)

        pos = rollout[idx - 2]
        graph = Graph(pos)
        graph.attrs = attrs

        graph = graph_preprocessor(graph, vels, types)

        return [graph, gt]
