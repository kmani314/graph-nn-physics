from .gnn import GraphNetwork
from .data import SimulationDataset, collate_fn
from .hyperparams import params
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from os.path import join
from torch.utils.data import DataLoader
# import torch.autograd.profiler as profiler
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    parser.add_argument('save_dir')
    args = parser.parse_args()

    device = torch.device(params['device'])

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

    network.to(device=device)
    network.float()

    dataset = SimulationDataset(args.dataset, args.group, params['vel_context'])

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

        # with profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True) as prof:
        output = network(batch[0])

        loss = torch.tensor(0, device=device, dtype=torch.float32)

        for i, gt in enumerate(batch[1]):
            mean = torch.tensor(batch[0][i].attrs['acc_mean'], device=device)
            std = torch.tensor(batch[0][i].attrs['acc_std'], device=device)

            trimmed = torch.narrow(output[i], 0, 0, batch[0][i].n_nodes)
            trimmed = torch.div(torch.sub(trimmed, mean), std)
            loss += criterion(gt.float(), trimmed.float())

        loss = torch.div(loss, params['batch_size'])

        loss.backward()
        optimizer.step()

        # print(prof.key_averages().table(sort_by='cuda_memory_usage'))

        if (epoch + 1) % params['decay_interval'] == 0:
            decay.step()

        writer.add_scalar('MSELoss', loss, epoch)
        writer.add_scalar('ExponentialLR', decay.get_last_lr()[0], epoch)

        del loss
