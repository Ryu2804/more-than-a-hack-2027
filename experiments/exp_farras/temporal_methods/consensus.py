import math
from collections.abc import Mapping

import numpy as np

import bridge as b


CONFIGS = [
    dict(name=f'alpha{alpha:g}_contact{beta:g}', alpha=alpha, contact_beta=beta)
    for alpha in (0.25, 0.50, 0.75)
    for beta in (0.0, 0.50)
]


def agreement_weights(bundle, observations, config):
    """Discount entering-frame switch costs from half-local directional agreement."""
    if not isinstance(bundle, Mapping):
        raise ValueError('bundle must contain short and setseq predictions')
    if not isinstance(observations, Mapping):
        raise ValueError('observations must be an observation dictionary')
    if not isinstance(config, Mapping):
        raise ValueError('config must contain alpha and contact_beta')
    try:
        alpha = float(config['alpha'])
        beta = float(config['contact_beta'])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('config must contain numeric alpha and contact_beta') from error
    if not math.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError('alpha must be finite and in [0, 1)')
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError('contact_beta must be finite and nonnegative')

    decoder = b.decoder
    control = decoder._as_numeric(observations.get('contact_control'), 'contact_control')
    if control.ndim != 2 or control.shape[1] != 7:
        raise ValueError('contact_control must have shape (N, 7)')
    if not np.isfinite(control).all():
        raise ValueError('contact_control must be finite')
    n = len(control)
    blocks = decoder._normalize_blocks(observations.get('blocks'), n)
    changes = []
    for source in ('short', 'setseq'):
        if source not in bundle:
            raise ValueError('bundle must contain short and setseq predictions')
        _, p_team = decoder._prediction_components(bundle[source], n)
        logits = decoder._logit(p_team)
        difference = np.zeros(n, dtype=np.float64)
        for start, end in blocks:
            difference[start + 1:end] = np.diff(logits[start:end])
        changes.append(difference)

    first, second = changes
    agreement = np.where(
        first * second > 0.0,
        np.clip(np.minimum(np.abs(first), np.abs(second)) / 2.0, 0.0, 1.0),
        0.0,
    )
    contact = np.clip(control[:, 2] / 3.0, 0.0, 1.0) * (np.abs(control[:, 3]) > 0.0)
    influence = np.clip(agreement + beta * contact, 0.0, 1.0)
    return 1.0 - alpha * influence


def decode(prediction, bundle, observations, stats, config):
    """Keep dead45 evidence, live resets, merge, and full-grid rate calibration."""
    weights = agreement_weights(bundle, observations, config)
    decoder = b.decoder
    X = decoder._as_numeric(observations.get('X'), 'observations X')
    if X.shape != (len(weights), 16):
        raise ValueError('observations X must have shape (N, 16)')
    if not isinstance(stats, Mapping):
        raise ValueError('stats must be a dictionary')
    try:
        rate = float(stats['rate'])
        dead_share = float(stats['dead_share'])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('stats must contain numeric rate and dead_share') from error
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError('training rate must be finite and nonnegative')
    if not math.isfinite(dead_share) or not 0.0 <= dead_share <= 1.0:
        raise ValueError('training dead_share must be in [0, 1]')

    blocks = decoder._normalize_blocks(observations.get('blocks'), len(X))
    p_dead, p_team = decoder._prediction_components(prediction, len(X))
    spec = b.fair.SPEC
    dead = decoder.dead_mask(p_dead, blocks, threshold=spec['dead_threshold'],
                             min_s=spec['min_dead_s'], shift=0.0)
    home, away = decoder._team_evidence(p_team, X, observations, spec)
    target = rate * sum(end - start for start, end in blocks) / (decoder.FPS * 60.0)
    target *= spec['rate_scale']
    min_frames = int(math.ceil(spec['merge_s'] * decoder.FPS))
    transition_costs = np.empty(len(X), dtype=np.float64)

    def make_labels(penalty):
        np.multiply(weights, penalty, out=transition_costs)
        labels = decoder._decode_live_team(home, away, blocks, dead, transition_costs, penalty)
        return decoder._merge_short_runs(labels, blocks, min_frames)

    penalty = decoder.calibrate_lambda(make_labels, blocks, target)
    return make_labels(penalty).astype(np.int8, copy=False)
