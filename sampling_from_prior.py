import torch
import numpy as np
import math
from mbt2018 import JointAutoregressiveHierarchicalPriors

def bottleneck_sampling(model: JointAutoregressiveHierarchicalPriors, task_id):

    medians = model.entropy_bottlenecks[task_id].quantiles[:, 0, 1]

    minima = medians - model.entropy_bottlenecks[task_id].quantiles[:, 0, 0]
    minima = torch.ceil(minima).int()
    minima = torch.clamp(minima, min=0)

    maxima = model.entropy_bottlenecks[task_id].quantiles[:, 0, 2] - medians
    maxima = torch.ceil(maxima).int()
    maxima = torch.clamp(maxima, min=0)

    pmf_start = medians - minima
    pmf_length = maxima + minima + 1

    max_length = pmf_length.max().item()
    device = pmf_start.device
    samples = torch.arange(max_length, device=device)

    samples = samples[None, :] + pmf_start[:, None, None]

    half = float(0.5)

    lower = model.entropy_bottlenecks[task_id]._logits_cumulative(samples - half, stop_gradient=True)
    upper = model.entropy_bottlenecks[task_id]._logits_cumulative(samples + half, stop_gradient=True)
    sign = - torch.sign(lower + upper)
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

    samples = samples[:, 0, :].cpu().detach().numpy()
    pmf = pmf[:, 0, :].cpu().detach().numpy()

    return samples, pmf

def gaussian_sampling(model: JointAutoregressiveHierarchicalPriors, h_s_samples):
    scales_samples, means_samples = h_s_samples.chunk(2, 1)
    gaussian_samples = torch.randn_like(means_samples)

    gaussian_samples = gaussian_samples * scales_samples + means_samples

    fake_y_hat, fake_y_likelihoods = model.gaussian_conditional(gaussian_samples, scales_samples, means=means_samples)

    return fake_y_hat
    
def generate_samples(model: JointAutoregressiveHierarchicalPriors, task_id, init_samples, shape):

    device = next(model.parameters()).device

    samples, pmf = bottleneck_sampling(model, task_id)
    bottleneck_samples = []
    # In case that the sum(pmf) != 1
    for i in range(samples.shape[0]):
        if pmf[i].sum() < 1:
            _samples = np.float32(np.append(samples[i], [samples[i][0]-1, samples[i][-1]+1]))
            _pmf = np.float32(np.append(pmf[i], [(1-pmf[i].sum())/2, (1-pmf[i].sum())/2]))
        else:
            _samples = samples[i]
            _pmf = pmf[i]
        bottleneck_samples.append(np.random.choice(_samples, size=(shape[0], 1, shape[2]//64, shape[3]//64), p=_pmf))

    bottleneck_samples = np.concatenate(bottleneck_samples, 1)
    bottleneck_samples = torch.from_numpy(bottleneck_samples).to(device)
    fake_z_hat, fake_z_likelihoods = model.entropy_bottlenecks[task_id](bottleneck_samples)
    params = model.h_s(fake_z_hat)
    ctx_params = model.context_prediction(init_samples)
    gaussian_params = model.entropy_parameters(
        torch.cat((params, ctx_params), dim=1)
    )
    fake_y_hat = gaussian_sampling(model, gaussian_params)

    return fake_y_hat.detach()
