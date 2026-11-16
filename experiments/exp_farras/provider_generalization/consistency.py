from __future__ import annotations

import torch
from torch.nn import functional as F

import bridge as b


def log_joint(logits):
    """Return log probabilities in view-team, other-team, dead order."""
    dead, team = logits.unbind(dim=1)
    live = F.logsigmoid(-dead)
    return torch.stack((
        live + F.logsigmoid(team),
        live + F.logsigmoid(-team),
        F.logsigmoid(dead),
    ), dim=1)


def symmetric_kl(first, second, y):
    first_log = log_joint(first)
    second_log = log_joint(second)
    divergence = 0.5 * (
        (first_log.exp() - second_log.exp()) * (first_log - second_log)
    ).sum(dim=1)
    known = y >= 0
    return divergence.masked_select(known).sum() / known.sum().clamp_min(1)


def loss(first, second, y, views, weights):
    """Combine two stochastic forwards of the same batch in one update."""
    first_state = b.old_weights.state_loss(first, y, views, weights, family='event_tcn')
    second_state = b.old_weights.state_loss(second, y, views, weights, family='event_tcn')
    return 0.5 * (first_state + second_state) + 0.5 * symmetric_kl(first, second, y)
