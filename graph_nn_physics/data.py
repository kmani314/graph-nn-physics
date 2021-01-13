from torch.utils.data import Dataset
from numpy import random
from .graph import Graph
# import time as timer
import numpy as np
import torch
import h5py

def collate_fn(batch, device):
    graphs = [x[0] for x in batch]

    for i in graphs:
        i.senders = i.senders.to(device)
        i.receivers = i.receivers.to(device)
        i.nodes = i.nodes.to(device)
        i.globals = i.globals.to(device)
        i.edges = i.edges.to(device)

    gt = [x[1].to(device=device) for x in batch]

    return (graphs, gt)

class SimulationDataset(Dataset):
    def __init__(self, file, group, vel_seq):
        self.file = h5py.File(file, 'r')
        self.group = group
        self.rollouts = len(self.file[self.group].keys())
        self.vel_seq = vel_seq

    def __len__(self):
        sum = 0
        for i in self.file[self.group]['positions']:
            sum += len(i)
        return sum

    def __getitem__(self, _):
        num = random.randint(self.rollouts)

        positions = self.file[self.group]['positions'].get(str(num))
        particle_types = self.file[self.group]['particle_types'].get(str(num))

        arr = np.zeros(positions.shape, dtype='float64')
        positions.read_direct(arr)
        rollout = torch.tensor(arr)

        arr = np.zeros(particle_types.shape, dtype='float64')
        particle_types.read_direct(arr)
        types = torch.tensor(arr).unsqueeze(1)

        # print('Dataset IO: {}'.format(timer.time() - start))
        # start2 = timer.time()
        begin = random.randint(rollout.size(0) - self.vel_seq) + self.vel_seq - 1
        vels = []

        for i in reversed(range(0, self.vel_seq)):
            # print(rollout[begin - i])
            # print(rollout[begin - i - 1])
            # print(rollout[begin - i] - rollout[begin - i - 1])
            vels.append(rollout[begin - i] - rollout[begin - i - 1])

        vels = torch.stack(vels, dim=1)
        # print(vels.shape)

        attrs = self.file[self.group].attrs
        mean = torch.tensor(attrs['vel_mean'])
        std = torch.tensor(attrs['vel_std'])
        radius = attrs['default_connectivity_radius']

        # normalization
        # print('Pre normalization: {}'.format(vels))
        vels = torch.div(torch.sub(vels, mean), std)
        # print('Post normalization: {}'.format(vels))

        # get next acceleration as next vel - last vel given to network
        gt = vels[:, -1] - vels[:, -2]
        # print(gt)
        vels = vels[:, :-1]
        vels = vels.view(vels.size(0), vels.size(1) * vels.size(2))
        # print(vels.shape)

        pos = rollout[begin]
        # print('Positions: {}'.format(rollout[begin - self.vel_seq:begin]))

        # boundaries
        lower = torch.tensor(attrs['bounds'][:, 0])
        upper = torch.tensor(attrs['bounds'][:, 1])

        dist = torch.cat([torch.sub(pos, lower), torch.sub(upper, pos)], dim=1)
        dist = torch.clamp(torch.div(dist, radius), -1, 1)

        # if using a particle type embedding, move this elsewhere
        nodes = torch.cat([pos, vels, dist, types], dim=1)

        graph = Graph(pos, globals=torch.tensor(1))
        graph.gen_edges(float(radius))
        graph.nodes = nodes

        return [graph, gt]
