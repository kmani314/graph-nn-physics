from torch.utils.data import Dataset
from numpy import random
from .graph import Graph
from .util import graph_preprocessor
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
    def __init__(self, file, group, vel_seq):
        self.file = h5py.File(file, 'r')
        self.group = group
        self.rollouts = len(self.file[self.group]['positions'].keys())
        self.vel_seq = vel_seq

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

        begin = random.randint(rollout.size(0) - self.vel_seq) + self.vel_seq - 1
        vels = []

        for i in reversed(range(0, self.vel_seq)):
            vels.append(rollout[begin - i] - rollout[begin - i - 1])

        vels = torch.stack(vels, dim=1)

        attrs = self.file[self.group].attrs
        mean = torch.tensor(attrs['vel_mean'])
        std = torch.tensor(attrs['vel_std'])

        # normalization
        vels = torch.div(torch.sub(vels, mean), std).float()

        # get next acceleration as next vel - last vel given to network
        gt = vels[:, -1] - vels[:, -2]
        end_vel = vels[:, :-1]
        end_vel = end_vel.view(end_vel.size(0), end_vel.size(1) * end_vel.size(2))

        pos = rollout[begin]
        graph = Graph(pos)
        graph.attrs = attrs

        graph = graph_preprocessor(graph, vels, types)

        return [graph, gt]
