from graph_nn_physics.util import graph_preprocessor, normalized_to_real
from graph_nn_physics.data import SimulationDataset, collate_fn
from graph_nn_physics.hyperparams import params
from graph_nn_physics.gnn import GraphNetwork
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import torch

def animation_func(num, data, points):
    points.set_offsets(data[num])
    return points

def infer_trajectory(network, curr_graph, steps, progress=False):
    pos = []
    device = curr_graph.device

    pos.append(curr_graph.pos[-1].to(device='cpu'))

    attrs = curr_graph.attrs
    amean = torch.tensor(attrs['acc_mean'])
    astd = torch.tensor(attrs['acc_std'])

    bar = range(steps)
    if progress:
        bar = tqdm(bar)

    with torch.no_grad():
        for i in bar:
            acc = network(curr_graph).to('cpu')

            curr_graph.to('cpu')
            acc = normalized_to_real(acc, amean, astd)

            prev_vel = curr_graph.pos[-1] - curr_graph.pos[-2]
            new_vel = prev_vel + acc

            new_pos = curr_graph.pos[-1] + new_vel
            seq = torch.cat([curr_graph.pos[1:], new_pos.unsqueeze(0)]).float()
            new_pos = new_pos.float()
            pos.append(new_pos)

            curr_graph = graph_preprocessor(seq, attrs, curr_graph.types)
            curr_graph.to(device)

    return torch.stack(pos).cpu().numpy()

def render_rollout_tensor(pos, title=None):
    fig, ax = plt.subplots()
    plt.axis('scaled')
    fig.set_size_inches(6, 6, True)
    ax.axis([0, 1, 0, 1])
    points = ax.scatter(pos[0][:, 0], pos[0][:, 1])

    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = np.transpose(img, (2, 0, 1))
    frames = [img]

    if title is not None:
        plt.title(title)

    for i in pos:
        points.set_offsets(i)
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.transpose(img, (2, 0, 1))
        frames.append(img)

    frames = torch.tensor(np.stack(frames)).unsqueeze(0)
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    parser.add_argument('model')
    parser.add_argument('steps')
    parser.add_argument('--gif')
    parser.add_argument('--title')
    args = parser.parse_args()

    device = torch.device('cpu')
    dataset = SimulationDataset(args.dataset, args.group, params['vel_context'], 0, only_first=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
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

    pos = infer_trajectory(network, sample_state, int(args.steps), progress=True)
    render_rollout_tensor(pos)

    curr_graph = sample_state[0]
    fig, ax = plt.subplots()
    points = ax.scatter(pos[0][:, 0], pos[0][:, 1])
    plt.axis('scaled')

    if args.title is not None:
        plt.title(args.title)

    bounds = []
    axes = []
    margin = 0.1

    for i in curr_graph.attrs['bounds']:
        for j in range(len(i)):
            x = -margin if j % 2 == 0 else margin
            axes.append(i[j] + x)
            bounds.append(i[j])

    ax.axis(axes)
    ax.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0], bounds[3] - bounds[2], ec='r', fill=False, linewidth=1.5))

    dim = curr_graph.attrs['dim']

    anim = animation.FuncAnimation(fig, animation_func, int(pos.shape[0]), fargs=(pos, points), interval=1)
    if args.gif is not None:
        print('Saving animation...')
        fig.set_size_inches(6, 6, True)
        anim.save(args.gif, writer='ffmpeg', fps=60, dpi=150)
    plt.show()
