from torch.utils.tensorboard import SummaryWriter
from .data import SimulationDataset, collate_fn
from torch.utils.data import DataLoader
from .util import decoder_normalizer
from .hyperparams import params
from .gnn import GraphNetwork
from os.path import join
from tqdm import tqdm
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    parser.add_argument('save_dir')
    args = parser.parse_args()

    device = torch.device(params['device'])

    network = GraphNetwork(
        node_dim=(params['vel_context'] + 3) * params['dim'] + 1,
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

    network.to(device=device)

    dataset = SimulationDataset(args.dataset, args.group, params['vel_context'], normalization=params['normalization'])

    loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, device)
    )

    writer = SummaryWriter()
    optimizer = torch.optim.Adam(network.parameters(), lr=params['lr'])
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, params['gamma'])
    criterion = torch.nn.MSELoss()

    for epoch, batch in enumerate(tqdm(loader, total=params['epochs'])):
        if epoch >= params['epochs']:
            break

        if (epoch + 1) % params['model_save_interval'] == 0:
            torch.save(network.cpu().state_dict(), join(args.save_dir, '{}.pt'.format(epoch + 1)))
            network.to(device=device)

        optimizer.zero_grad()

        output = network(batch[0])

        loss = torch.tensor(0, device=device, dtype=torch.float32)
        differences = []

        for i, gt in enumerate(batch[1]):
            trimmed = torch.narrow(output[i], 0, 0, batch[0][i].n_nodes)
            differences.append(trimmed - gt)
            loss += criterion(gt.float(), trimmed.float())

        norm_difference = 0
        with torch.no_grad():
            norm_difference = torch.mean(
                torch.tensor([torch.linalg.norm(x) for x in differences])
            )

        loss = torch.div(loss, params['batch_size'])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % params['decay_interval'] == 0:
            decay.step()

        # writer.add_scalar('Acceleration norm', norm, epoch)
        # writer.add_scalar('Ground truth norm', gt_norm, epoch)
        writer.add_scalar('Relative', norm_difference, epoch)
        writer.add_scalar('MSELoss', loss, epoch)
        writer.add_scalar('ExponentialLR', decay.get_last_lr()[0], epoch)

        del loss, output
