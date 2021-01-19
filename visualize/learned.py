from graph_nn_physics.util import graph_preprocessor, normalized_to_real
from graph_nn_physics.data import SimulationDataset, collate_fn
from graph_nn_physics.hyperparams import params
from graph_nn_physics.gnn import GraphNetwork
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from graph_nn_physics.graph import Graph
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch


def animation_func(num, data, points, speed, dim):
    # points._offsets3d = (data[num * speed][:, 0], data[num * speed][:, 2], data[num * speed][:, 1])  # , data[num][:, 1])
    points._offsets3d = [data[num * speed][:, i] for i in range(dim)] + [0]
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    parser.add_argument('model')
    parser.add_argument('steps')
    parser.add_argument('speed')
    args = parser.parse_args()

    device = torch.device(params['device'])
    dataset = SimulationDataset(args.dataset, args.group, params['vel_context'])

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, device)
    )

    network = GraphNetwork(
        node_dim=(params['vel_context'] + 2) * params['dim'] + 1,
        edge_dim=params['dim'] + 1,
        global_dim=1,
        mp_steps=params['mp_steps'],
        proc_hidden_dim=params['proc_hidden_dim'],
        encoder_hidden_dim=params['encoder_hidden_dim'],
        decoder_hidden_dim=params['decoder_hidden_dim'],
        dim=params['dim'],
        ve_dim=params['embedding_dim'],
        ee_dim=params['embedding_dim'],
    )

    network.load_state_dict(torch.load(args.model))
    network.eval()
    network.to(device=device)

    sample_state = next(iter(loader))

    pos = []
    pos.append(sample_state[0][0].pos)
    curr_graph = sample_state[0]
    radius = curr_graph[0].attrs['default_connectivity_radius']

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for i in range(int(args.steps)):
            output = network(curr_graph)

            curr_graph = curr_graph[0]
            curr_graph.to('cpu')
            output = output.to(device='cpu')
            attrs = curr_graph.attrs
            mean = torch.tensor(attrs['acc_mean'])
            std = torch.tensor(attrs['acc_std'])

            trimmed = torch.narrow(output[0], 0, 0, curr_graph.n_nodes)
            acc = normalized_to_real(trimmed, mean, std)

            # omit delta_t
            new_vels = curr_graph.vels[:, -1] + acc
            new_pos = pos[-1].to(device='cpu') + new_vels
            types = curr_graph.types
            vels = torch.cat([curr_graph.vels[:, 1:], new_vels.unsqueeze(1)], dim=1)
            pos.append(new_pos)
            new_pos = new_pos.float()

            curr_graph = Graph(new_pos.detach().cpu())
            curr_graph.attrs = attrs
            curr_graph = graph_preprocessor(curr_graph, vels, types)

            curr_graph.to(device)
            curr_graph = [curr_graph]
            print('Inferred step {}'.format(i))

    pos = np.stack([x.detach().cpu().numpy() for x in pos])
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    dim = curr_graph[0].attrs['dim']

    ax.set_xlim3d([0, 0.9])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 0.9])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 0.9])
    ax.set_zlabel('Z')

    # points = ax.scatter(pos[0][:, 0], pos[0][:, 2], pos[0][:, 1])  # , data[0][:, 1], s=1000, alpha=0.8)
    points = [pos[0][:, i] for i in range(dim)]
    points = ax.scatter(points, 0, s=10)
    anim = animation.FuncAnimation(fig, animation_func, int(pos.shape[0] / int(args.speed)), fargs=(pos, points, int(args.speed), dim), interval=1)
    plt.show()
