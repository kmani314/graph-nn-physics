from torch.utils.tensorboard import SummaryWriter
from graph_nn_physics.data import SimulationDataset, collate_fn
from torch.utils.data import DataLoader
from graph_nn_physics.hyperparams import params
from graph_nn_physics.gnn import GraphNetwork
import torch.autograd.profiler as profiler
from torchviz import make_dot
from os.path import join
from tqdm import tqdm
import argparse
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('group')
parser.add_argument('save_dir')
parser.add_argument('--ckpt')
parser.add_argument('--run_name')
parser.add_argument('--save_graph')
parser.add_argument('--profile', action='store_true')
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

dataset = SimulationDataset(args.dataset, args.group, params['vel_context'], params['noise_std'])

torch.set_printoptions(precision=12, threshold=64000)

loader = DataLoader(
    dataset,
    batch_size=params['batch_size'],
    collate_fn=collate_fn,
    pin_memory=True,
)

optimizer = torch.optim.Adam(network.parameters(), lr=params['lr'])
decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, params['gamma'])
criterion = torch.nn.MSELoss()

logging = args.run_name is not None

if logging:
    writer = SummaryWriter(join('runs', args.run_name), flush_secs=1)

min_loss = -1
epoch = 0

if args.ckpt is not None:
    checkpoint = torch.load(args.ckpt)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    network.train()

pbar = tqdm(loader, total=params['epochs'], initial=epoch, dynamic_ncols=True)

last_time = time.time()

with profiler.profile(use_cuda=True, with_stack=True, enabled=args.profile) as prof:
    for batch in pbar:
        if epoch >= params['epochs']:
            break

        optimizer.zero_grad(set_to_none=True)

        batch[0].to(device=device)
        output = network(batch[0])

        if args.save_graph is not None:
            make_dot(output, params=dict(network.named_parameters())).render(args.save_graph)
            args.save_graph = None

        gt = batch[1].to(device=device)
        loss = criterion(output, gt)
        if min_loss < 0 or loss < min_loss:
            min_loss = loss

        norm = torch.mean(torch.linalg.norm(output, keepdims=True, dim=1))
        gt_norm = torch.mean(torch.linalg.norm(gt, keepdims=True, dim=1))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % params['decay_interval'] == 0:
            decay.step()

        if logging:
            dt = time.time() - last_time
            last_time = time.time()
            writer.add_scalar('stats/predicted', norm, epoch)
            writer.add_scalar('stats/it/s', 1 / dt, epoch)
            writer.add_scalar('stats/gt', gt_norm, epoch)
            writer.add_scalar('loss/mse', loss, epoch)
            writer.add_scalar('loss/minimum', min_loss, epoch)
            writer.add_scalar('stats/learning rate', decay.get_last_lr()[0], epoch)

        if (epoch + 1) % params['model_save_interval'] == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': network.cpu().state_dict(),
                    'loss': loss
                },
                join(args.save_dir, f'{args.run_name}-{epoch + 1}.pt')
            )
            network.to(device=device)
        epoch += 1

if args.profile:
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    prof.export_chrome_trace('trace.json')
