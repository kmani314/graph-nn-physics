from graph_nn_physics.util import graph_preprocessor, normalized_to_real, sequence_postprocessor
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
    points._offsets3d = (data[num * speed][:, 0], 0, data[num * speed][:, 1])
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
    dataset = SimulationDataset(args.dataset, args.group, params['vel_context'], 0, normalization=params['normalization'])

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, device)
    )

    network = GraphNetwork(
        node_dim=(params['vel_context'] + 1) * params['dim'] + 1,
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
    pos.append(sample_state[0].pos[-1])
    curr_graph = sample_state[0]
    radius = curr_graph.attrs['default_connectivity_radius']

    criterion = torch.nn.MSELoss()

    torch.set_printoptions(threshold=64000)
    with torch.no_grad():
        for i in range(int(args.steps)):
            acc = network(curr_graph).to('cpu')

            curr_graph.to('cpu')
            attrs = curr_graph.attrs

            amean = torch.tensor(attrs['acc_mean'])
            astd = torch.tensor(attrs['acc_std'])
            acc = normalized_to_real(acc, amean, astd)

            prev_vel = curr_graph.pos[-1] - curr_graph.pos[-2]
            new_vel = prev_vel + acc

            new_pos = curr_graph.pos[-1] + new_vel
            seq = torch.cat([curr_graph.pos[1:], new_pos.unsqueeze(0)]).float()
            new_pos = new_pos.float()
            pos.append(new_pos)

            curr_graph = graph_preprocessor(seq, attrs, curr_graph.types)

            curr_graph.to(device)
            print('Inferred step {}'.format(i))

    # pos = np.stack([x.cpu().numpy() for x in pos])
    pos = torch.stack(pos).cpu().numpy()

    fig = plt.figure()
    ax = p3.Axes2D(fig)

    dim = curr_graph.attrs['dim']

    ax.set_xlim3d([0, 1])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 1])
    ax.set_ylabel('Y')

    # ax.set_zlim3d([0, 1])
    # ax.set_zlabel('Z')

    points = ax.scatter(pos[0][:, 0], 0, pos[0][:, 1])  # , data[0][:, 1], s=1000, alpha=0.8)
    anim = animation.FuncAnimation(fig, animation_func, int(pos.shape[0] / int(args.speed)), fargs=(pos, points, int(args.speed), dim), interval=1)
    plt.show()
