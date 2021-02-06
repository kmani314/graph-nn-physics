import torch

def gen_noise(pos, noise_std):
    vels = pos[1:] - pos[:-1]

    # print(torch.full_like(vels, noise_std / num_vels ** 0.5))
    noise = torch.normal(
        torch.zeros_like(vels),
        torch.full_like(vels, noise_std / vels.size(0) ** 0.5)
    )

    # random walk over velocities
    noise = torch.cumsum(noise, dim=0)
    noise = torch.cat([torch.zeros_like(noise[:1]), torch.cumsum(noise, dim=0)], dim=0)
    # print(noise)
    return noise
