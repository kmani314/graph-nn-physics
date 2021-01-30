from torch.utils.tensorboard import SummaryWriter
from .data import SimulationDataset, collate_fn
from torch.utils.data import DataLoader
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
    parser.add_argument('--run_name')
    args = parser.parse_args()

    device = torch.device(params['device'])

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

    network.to(device=device)

    dataset = SimulationDataset(args.dataset, args.group, params['vel_context'], params['noise_std'], normalization=params['normalization'])

    torch.set_printoptions(precision=12)
    loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        collate_fn=lambda x: collate_fn(x, device)
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=params['lr'])
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, params['gamma'])
    criterion = torch.nn.MSELoss()

    logging = args.run_name is not None

    if logging:
        writer = SummaryWriter(join('runs', args.run_name))

    for epoch, batch in enumerate(tqdm(loader, total=params['epochs'])):
        if epoch >= params['epochs']:
            break

        if (epoch + 1) % params['model_save_interval'] == 0:
            torch.save(network.cpu().state_dict(), join(args.save_dir, f'{args.run_name}-{epoch + 1}.pt'))
            network.to(device=device)

        optimizer.zero_grad()

        output = network(batch[0])

        loss = torch.tensor(0, device=device, dtype=torch.float32)

        norm = []
        gt_norm = []

        for i, gt in enumerate(batch[1]):
            trimmed = torch.narrow(output[i], 0, 0, batch[0][i].n_nodes)

            norm.append(torch.linalg.norm(trimmed, keepdims=True, dim=1))
            gt_norm.append(torch.linalg.norm(gt, keepdims=True, dim=1))
            # print(gt.float())

            loss += criterion(gt.float(), trimmed.float())

        norm = torch.mean(torch.cat(norm))
        gt_norm = torch.mean(torch.cat(gt_norm))

        loss /= params['batch_size']

        loss.backward()
        optimizer.step()

        if (epoch + 1) % params['decay_interval'] == 0:
            decay.step()

        if logging:
            writer.add_scalar('stats/predicted', norm, epoch)
            writer.add_scalar('stats/gt', gt_norm, epoch)
            writer.add_scalar('loss', loss, epoch)
            writer.add_scalar('stats/learning rate', decay.get_last_lr()[0], epoch)

        del loss, output
