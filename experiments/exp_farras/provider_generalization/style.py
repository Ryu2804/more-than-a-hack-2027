"""Training-batch-only MixStyle on pooled player features, not raw tracking."""

import torch


def apply(x, rng, probability=0.5):
    """Call only during training; donors come solely from this source batch."""
    if rng.random() >= probability or x.shape[0] < 2:
        return x

    players = x[:, :23, :].to(torch.float64)
    mean = players.mean(dim=-1, keepdim=True).detach()
    std = (players.var(dim=-1, unbiased=False, keepdim=True) + 1e-6).sqrt().detach()
    normalized = (players - mean) / std
    permutation = torch.as_tensor(
        rng.permutation(x.shape[0]), device=x.device, dtype=torch.long
    )
    coefficients = torch.as_tensor(
        rng.beta(0.5, 0.5, size=(x.shape[0], 1, 1)),
        device=x.device,
        dtype=players.dtype,
    )
    mixed_mean = coefficients * mean + (1 - coefficients) * mean[permutation]
    mixed_std = coefficients * std + (1 - coefficients) * std[permutation]
    result = x.clone()
    result[:, :23, :] = normalized * mixed_std + mixed_mean
    return result
