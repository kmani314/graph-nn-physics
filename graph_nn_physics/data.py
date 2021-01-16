from torch.utils.data import Dataset
from numpy import random
from .graph import Graph
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

        arr = np.zeros(positions.shape, dtype='float64')
        positions.read_direct(arr)
        rollout = torch.tensor(arr)

        arr = np.zeros(particle_types.shape, dtype='float64')
        particle_types.read_direct(arr)
        types = torch.tensor(arr).unsqueeze(1)

        begin = random.randint(rollout.size(0) - self.vel_seq) + self.vel_seq - 1
        vels = []

        for i in reversed(range(0, self.vel_seq)):
            vels.append(rollout[begin - i] - rollout[begin - i - 1])

        vels = torch.stack(vels, dim=1)

        attrs = self.file[self.group].attrs
        mean = torch.tensor(attrs['vel_mean'])
        std = torch.tensor(attrs['vel_std'])
        radius = attrs['default_connectivity_radius']

        # normalization
        vels = torch.div(torch.sub(vels, mean), std)

        # get next acceleration as next vel - last vel given to network
        gt = vels[:, -1] - vels[:, -2]
        end_vel = vels[:, :-1]
        end_vel = end_vel.view(end_vel.size(0), end_vel.size(1) * end_vel.size(2))

        pos = rollout[begin]

        # boundaries
        lower = torch.tensor(attrs['bounds'][:, 0])
        upper = torch.tensor(attrs['bounds'][:, 1])

        dist = torch.cat([torch.sub(pos, lower), torch.sub(upper, pos)], dim=1)
        dist = torch.clamp(torch.div(dist, radius), -1, 1)

        # if using a particle type embedding, move this elsewhere
        nodes = torch.cat([end_vel, dist, types], dim=1)

        graph = Graph(pos)
        graph.gen_edges(float(radius))

        senders = torch.index_select(graph.nodes, 0, graph.senders)
        senders = torch.narrow(senders, 1, 0, attrs['dim'])
        receivers = torch.index_select(graph.nodes, 0, graph.receivers)
        receivers = torch.narrow(receivers, 1, 0, attrs['dim'])

        positional = torch.div((senders - receivers), graph.radius)
        norm = torch.norm(positional, dim=1).unsqueeze(1)
        graph.edges = torch.cat([positional, norm], dim=1)
        graph.nodes = nodes

        graph.attrs = attrs
        # not used for training, instead for inference
        graph.vels = end_vel

        return [graph, gt]
