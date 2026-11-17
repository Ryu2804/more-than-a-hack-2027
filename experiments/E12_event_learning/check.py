import argparse
import json
import subprocess
import sys
import time
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import audit
import bridge as b
import models


def smoke_path(family):
    return b.OUT / 'smoke' / family


def smoke(family):
    heldout = b.MATCHES[0]
    names = b.fair.split_names(heldout)
    fitted = models.fit([b.c.load(name, 'r0', with_features=True) for name in names], family, steps=1)
    count = fitted['training']['seed_fits'][0]['parameters']
    audit.training(heldout, family, fitted['training_names'], fitted['training']['seed_fits'], 1, count)
    observed = b.fair.observations(b.c.load(heldout, 'r0', with_features=True))
    prediction = models.predict(fitted, observed)
    audit.probability(prediction['probability'], len(observed['X']))
    path = smoke_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, path.with_suffix('.pt'))
    np.savez_compressed(path.with_suffix('.npz'), probability=prediction['probability'])
    b.c.write_json(path.with_suffix('.json'), dict(family=family, heldout=heldout, training_names=list(names),
        parameters=count, training=fitted['training'], checkpoint_sha256=b.c.digest(path.with_suffix('.pt')),
        probability_sha256=b.c.digest(path.with_suffix('.npz')), steps=1, seeds=[0, 1, 2], metric_evaluation=False))
    print('REAL TRAINING SMOKE', family, '3seeds/1update', flush=True)


def replay(family):
    path = smoke_path(family)
    fitted = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
    observed = b.fair.observations(b.c.load(b.MATCHES[0], 'r0', with_features=True))
    actual = models.predict(fitted, observed)['probability']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
        audit.replay(actual, archive['probability'])
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
        path = smoke_path(family)
        metadata = json.loads(path.with_suffix('.json').read_text())
        proof = json.loads(path.with_name(path.name + '_replay').with_suffix('.json').read_text())
        if proof != dict(family=family, checkpoint_sha256=metadata['checkpoint_sha256'],
                         probability_sha256=metadata['probability_sha256'], identical_full_probabilities=True):
            raise ValueError('Smoke model replay provenance mismatch')
        records.append(dict(**metadata, isolated_inference_identical=True))
    b.c.write_json(b.OUT / 'checks.json', dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        smoke=records, new_smoke_neural_seed_fits=18, updates_per_smoke_fit=1, metric_evaluation=False,
        seconds=time.monotonic() - started))
    print('CHECKS PASS', result.testsRun, 'REAL SMOKE SEED FITS', 18, 'SAVED MODEL REPLAYS', 6, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=models.FAMILIES)
    parser.add_argument('--replay', choices=models.FAMILIES)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.family: smoke(args.family)
        elif args.replay: replay(args.replay)
        else: main()
