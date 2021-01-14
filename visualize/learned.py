import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
# import time
import torch
# import datetime
import argparse
from graph_nn_physics.graph import Graph
from graph_nn_physics.hyperparams import params
from graph_nn_physics.gnn import GraphNetwork
from graph_nn_physics.data import SimulationDataset, collate_fn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    parser.add_argument('model')
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
        edge_dim=1,
        global_dim=1,
        mp_steps=params['mp_steps'],
        proc_hidden_dim=params['proc_hidden_dim'],
        encoder_hidden_dim=params['encoder_hidden_dim'],
        decoder_hidden_dim=params['decoder_hidden_dim'],
        dim=params['dim'],
        ve_dim=params['embedding_dim'],
        ee_dim=params['embedding_dim'],
        relative_encoder=params['relative_encoder']
    )

    network.load_state_dict(torch.load(args.model))
    network.eval()
    network.to(device=device)

    sample_state = next(iter(loader))

    pos = []
    pos.append(sample_state[0][0].nodes[:, 0:params['dim']])
    curr_graph = sample_state[0]
    radius = curr_graph[0].attrs['default_connectivity_radius']

    for i in range(64):
        print('Inferred step {}'.format(i))
        output = network(curr_graph)

        curr_graph = curr_graph[0]

        mean = torch.tensor(curr_graph.attrs['acc_mean'], device=device)
        std = torch.tensor(curr_graph.attrs['acc_std'], device=device)

        trimmed = torch.narrow(output[0], 0, 0, curr_graph.n_nodes)
        acc = torch.div(torch.sub(trimmed, mean), std)

        # omit delta_t
        new_vels = curr_graph.vels + acc
        new_pos = pos[-1] + curr_graph.vels

        pos.append(new_pos)

        curr_graph = Graph(new_pos.cpu(), globals=torch.tensor(1))
        curr_graph.vels = new_vels
        curr_graph.gen_edges(radius)
        curr_graph.to(device)
        curr_graph = [curr_graph]

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([0, 0.9])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 0.9])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 0.9])
    ax.set_zlabel('Z')

    points = ax.scatter(data[0][:, 0], data[0][:, 2], data[0][:, 1])  # , data[0][:, 1], s=1000, alpha=0.8)
    anim = animation.FuncAnimation(fig, animation_func, int(data.shape[0] / 8), fargs=(data, points), interval=1)
    plt.show()
