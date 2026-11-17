import argparse
import hashlib
import json
import subprocess
import sys
import time
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bridge as b
import models


def smoke_paths(family):
    return b.OUT / 'smoke' / family


def smoke(family):
    heldout = b.MATCHES[0]
    names = b.fair.split_names(heldout)
    train = [b.c.load(name, 'r0', with_features=True) for name in names]
    fitted = models.fit(train, family, steps=1)
    if tuple(fitted['training_names']) != names or [(row['seed'], row['updates']) for row in fitted['training']['seed_fits']] != [(0, 1), (1, 1), (2, 1)]:
        raise ValueError('Smoke training did not perform expected disjoint three-seed fits')
    observed = b.fair.observations(b.c.load(heldout, 'r0', with_features=True))
    output = models.predict(fitted, observed)
    if output['probability'].shape != (len(observed['X']), 3):
        raise ValueError('Smoke held-out prediction timeline differs')
    path = smoke_paths(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, path.with_suffix('.pt'))
    np.savez_compressed(path.with_suffix('.npz'), probability=output['probability'])
    metadata = dict(family=family, heldout=heldout, training_names=list(names), steps=1, seeds=[0, 1, 2],
        parameters=fitted['training']['seed_fits'][0]['parameters'], training=fitted['training'],
        probability_sha256=b.c.digest(path.with_suffix('.npz')), checkpoint_sha256=b.c.digest(path.with_suffix('.pt')),
        metric_evaluation=False)
    b.c.write_json(path.with_suffix('.json'), metadata)
    print('TRAINING SMOKE PASS', family, '3seeds/1update', flush=True)


def replay(family):
    path = smoke_paths(family)
    fitted = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
    observed = b.fair.observations(b.c.load(b.MATCHES[0], 'r0', with_features=True))
    output = models.predict(fitted, observed)
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
        if not np.array_equal(output['probability'], archive['probability']):
            raise ValueError('Isolated saved smoke model inference differs')
    b.c.write_json(path.with_name(path.name + '_replay').with_suffix('.json'), dict(family=family,
        checkpoint_sha256=b.c.digest(path.with_suffix('.pt')), probability_sha256=b.c.digest(path.with_suffix('.npz')),
        identical_full_probabilities=True))
    print('SMOKE INFERENCE IDENTICAL', family, flush=True)


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(b.HERE), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    records = []
    for family in models.FAMILIES:
        subprocess.run([sys.executable, '-B', str(b.HERE / 'check.py'), '--family', family], check=True)
        subprocess.run([sys.executable, '-B', str(b.HERE / 'check.py'), '--replay', family], check=True)
        path = smoke_paths(family)
        metadata = json.loads(path.with_suffix('.json').read_text())
        proof = json.loads(path.with_name(path.name + '_replay').with_suffix('.json').read_text())
        if proof['probability_sha256'] != metadata['probability_sha256'] or proof['checkpoint_sha256'] != metadata['checkpoint_sha256'] or not proof['identical_full_probabilities']:
            raise ValueError('Smoke inference proof differs')
        records.append(dict(**metadata, isolated_inference_identical=True))
    b.c.write_json(b.OUT / 'checks.json', dict(tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), smoke=records, new_smoke_neural_seed_fits=9, updates_per_smoke_fit=1,
        metric_evaluation=False, seconds=time.monotonic() - started))
    print('CHECKS PASS', result.testsRun, 'REAL TRAINING FITS', 9, 'ISOLATED INFERENCE REPLAYS', 3, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=models.FAMILIES)
    parser.add_argument('--replay', choices=models.FAMILIES)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.family: smoke(args.family)
        elif args.replay: replay(args.replay)
        else: main()
