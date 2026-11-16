from __future__ import annotations

import torch
from torch.nn import functional as F

import bridge as b


def loss(logits, y, views, weights, family):
    if family == 'balanced':
        return b.old_weights.state_loss(logits, y, views, weights, family='spell_balanced_tcn')
    if family != 'focal':
        raise ValueError(f'Unknown objective family: {family!r}')

    known = (y >= 0).to(logits.dtype)
    live = ((y == 0) | (y == 1)).to(logits.dtype)
    target_dead = (y == 2).to(logits.dtype)
    target_team = (y == views[:, None]).to(logits.dtype)
    dead = F.binary_cross_entropy_with_logits(logits[:, 0], target_dead, reduction='none')
    team = F.binary_cross_entropy_with_logits(logits[:, 1], target_team, reduction='none')
    probability_correct = torch.exp(-team)
    focal_team = (1.0 - probability_correct).square() * team
    return (
        (dead * known * weights).sum() / known.sum().clamp_min(1.0)
        + (focal_team * live * weights).sum() / live.sum().clamp_min(1.0)
    )
