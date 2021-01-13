from .gnn import GraphNetwork
from .data import SimulationDataset, collate_fn
from .hyperparams import params
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('group')
    args = parser.parse_args()

    device = torch.device(params['device'])

    network = GraphNetwork(
        node_dim=15,
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
    criterion = torch.nn.MSELoss()

    for epoch, batch in enumerate(tqdm(loader, total=params['epochs'])):
        if epoch >= params['epochs'] - 1:
            break
        optimizer.zero_grad()

        output = network(batch[0])

        loss = torch.tensor(0, device=device, dtype=torch.float32)

        for i, gt in enumerate(batch[1]):
            trimmed = torch.narrow(output[i], 0, 0, batch[0][i].n_nodes)
            loss += criterion(gt.float(), trimmed)

        loss.backward()
        optimizer.step()
        writer.add_scalar('Training loss', loss, epoch)
