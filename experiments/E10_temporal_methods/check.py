import sys
import time
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import run as campaign
import bridge as b


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(b.HERE), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    heldout = b.MATCHES[0]
    names = b.fair.split_names(heldout)
    training = [b.c.load(name, 'r0', light=True) for name in names]
    stats = b.fair.statistics(training)
    prior = campaign.duration.fit(training, cap=80)
    if tuple(prior['training_names']) != names:
        raise ValueError('Smoke prior membership mismatch')
    data = b.c.load(heldout, 'r0')
    observed = b.observations(data)
    indices = np.concatenate([np.arange(start, min(start + 48, end)) for start, end in observed['blocks']])
    sizes = [min(48, end - start) for start, end in observed['blocks']]
    edges = np.r_[0, np.cumsum(sizes)]
    small = dict(_name=heldout, blocks=np.column_stack((edges[:-1], edges[1:])),
                 **{key: observed[key][indices] for key in ('X', 'distances', 'contact_control')})
    bundle = {key: b.result(value['probability'][indices]) for key, value in b.bundle(heldout).items()}
    prediction = b.fuse(bundle)
    control = b.baseline(prediction, small, stats)
    checked = []
    for spec in campaign.settings():
        codes = campaign.decoded(spec, prediction, bundle, small, stats, prior, control)
        if codes.shape != (len(indices),) or codes.dtype != np.int8 or not np.isin(codes, (0, 1, 2)).all():
            raise ValueError('Real-input decoder smoke failed: ' + spec['key'])
        checked.append(spec['key'])
    b.c.write_json(b.OUT / 'checks.json', dict(tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), smoke_settings=checked, heldout=heldout, training_names=list(names),
        observation_samples=len(indices), halves=len(sizes), metric_evaluation=False,
        new_neural_seed_fits=0, seconds=time.monotonic() - started))
    print('CHECKS PASS', result.testsRun, 'SMOKE', len(checked), 'NO METRIC EVALUATION', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
