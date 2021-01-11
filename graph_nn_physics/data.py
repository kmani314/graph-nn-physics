from torch.utils.data import Dataset
from numpy import random
from graph import Graph
import torch
import h5py

class SimulationDataset(Dataset):
    def __init__(self, file, group, vel_seq):
        self.file = h5py.File(file, 'r')
        self.group = group
        self.rollouts = len(self.file[self.group].keys())
        self.vel_seq = vel_seq

    def __len__(self):
        sum = 0
        for i in self.file[self.group]['position']:
            sum += len(i)
        return sum

    def __getitem__(self, idx):
        num = random.randint(self.rollouts)

        rollout = torch.tensor(
            self.file[self.group]['position'][num]
        )

        types = torch.tensor(
            self.file[self.group]['particle_types'][num]
        )

        begin = random.randint(rollout.size(0) - self.vel_seq) + self.vel_seq
        vels = []

        for i in range(self.vel_seq, 0):
            vels.append(rollout[begin - i] - rollout[begin - i - 1])

        vels = torch.stack(vels)
        pos = rollout[begin]

        # if using a particle type embedding, move this elsewhere
        nodes = torch.cat([pos, vels, types], dim=1)

        graph = Graph(nodes)
        graph.gen_edges(self.file[self.group].attrs['default_connectivity_radius'])

        return graph
