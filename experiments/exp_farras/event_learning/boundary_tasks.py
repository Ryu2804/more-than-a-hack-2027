from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

import bridge as b


def targets(labels, blocks):
    """Prepare full-match targets before cropping; each half resets adjacency."""
    values = b.training._labels(labels, (1,))
    normalized_blocks = b.original._normalize_blocks(blocks, len(values))
    target = np.zeros(len(values), dtype=np.float32)
    valid = np.zeros(len(values), dtype=np.bool_)
    for start, end in normalized_blocks:
        target[start:end], valid[start:end] = b.training.boundary_targets(values[start:end])
    return target, valid


def loss(logits, y, views, weights, family, auxiliary=None):
    if family not in ('boundary', 'ranking'):
        raise ValueError(f'Unknown boundary training family: {family!r}')
    if family == 'boundary':
        if logits.shape[1] != 3:
            raise ValueError('boundary training requires three logit heads')
        if auxiliary is None or len(auxiliary) != 2:
            raise ValueError('boundary training requires full-match auxiliary targets and validity')
        target, valid = auxiliary
        if target.shape != y.shape or valid.shape != y.shape:
            raise ValueError('boundary target and validity must match label shape')

    state = b.old_weights.state_loss(logits[:, :2], y, views, weights, family='event_tcn')
    if family == 'boundary':
        boundary = F.binary_cross_entropy_with_logits(
            logits[:, 2], target.to(logits.dtype),
            pos_weight=logits.new_tensor(5.0), reduction='none',
        )
        mask = valid.to(logits.dtype)
        return state + .2 * (boundary * mask).sum() / mask.sum().clamp_min(1.0)

    live = (y == 0) | (y == 1)
    eligible = live[:, 1:] & live[:, :-1] & (y[:, 1:] != y[:, :-1])
    sign = torch.where(y[:, 1:] == views[:, None], 1.0, -1.0).to(logits.dtype)
    jump = logits[:, 1, 1:] - logits[:, 1, :-1]
    pair_loss = F.softplus(1.0 - sign * jump)
    mask = eligible.to(logits.dtype)
    return state + .1 * (pair_loss * mask).sum() / mask.sum().clamp_min(1.0)
