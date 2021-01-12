from torch.utils.data import Dataset
from numpy import random
from .graph import Graph
import torch
import h5py

def collate_fn(batch):
    graphs = [x[0] for x in batch]
    gt = [x[1] for x in batch]
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

        rollout = torch.tensor(
            self.file[self.group]['positions'][str(num)]
        )

        types = torch.tensor(
            self.file[self.group]['particle_types'][str(num)]
        ).unsqueeze(1)

        begin = random.randint(rollout.size(0) - self.vel_seq) + self.vel_seq - 1
        vels = []

        for i in reversed(range(0, self.vel_seq + 1)):
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
        vels = vels[:, :-1]
        vels = vels.view(vels.size(0), vels.size(1) * vels.size(2))

        pos = rollout[begin]

        # boundaries
        lower = torch.tensor(attrs['bounds'][:, 0])
        upper = torch.tensor(attrs['bounds'][:, 1])

        dist = torch.cat([torch.sub(pos, lower), torch.sub(upper, pos)], dim=1)
        dist = torch.clamp(torch.div(dist, radius), -1, 1)

        # if using a particle type embedding, move this elsewhere
        nodes = torch.cat([pos, vels, dist, types], dim=1)

        graph = Graph(nodes, globals=torch.tensor(1))
        graph.gen_edges(radius)

        return [graph, gt]
