from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

import bridge as b


def _complete_runs(labels, blocks):
    for block_start, block_end in blocks:
        part = labels[block_start:block_end]
        edges = np.r_[0, np.flatnonzero(part[1:] != part[:-1]) + 1, len(part)]
        for start, end in zip(edges[:-1], edges[1:]):
            if part[start] not in (0, 1):
                continue
            if start == 0 or end == len(part):
                continue
            if part[start - 1] == -1 or part[end] == -1:
                continue
            yield block_start, block_end, block_start + int(start), block_start + int(end)


def spell_weights(labels, blocks):
    values = b.training._labels(labels, (1,))
    normalized_blocks = b.original._normalize_blocks(blocks, len(values))
    result = np.ones(len(values), dtype=np.float32)
    result[values == -1] = 0.0
    for _, _, start, end in _complete_runs(values, normalized_blocks):
        result[start:end] = np.clip(np.sqrt(20.0 / (end - start)), 0.25, 4.0)
    return result


def state_loss(logits, y, views, weights, family):
    known = (y >= 0).to(logits.dtype)
    live = ((y == 0) | (y == 1)).to(logits.dtype)
    target_dead = (y == 2).to(logits.dtype)
    target_team = (y == views[:, None]).to(logits.dtype)
    dead = F.binary_cross_entropy_with_logits(logits[:, 0], target_dead, reduction='none')
    team = F.binary_cross_entropy_with_logits(logits[:, 1], target_team, reduction='none')
    if family == 'spell_balanced_tcn':
        team_weight = weights * live
        denominator = team_weight.sum()
        denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        return (dead * known).sum() / known.sum().clamp_min(1.0) + (team * team_weight).sum() / denominator
    if family in ('event_tcn', 'dual_scale_tcn'):
        return (dead * known * weights).sum() / known.sum().clamp_min(1.0) + (team * live * weights).sum() / live.sum().clamp_min(1.0)
    raise ValueError(f'Unknown training family: {family!r}')
