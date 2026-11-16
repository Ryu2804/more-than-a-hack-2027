import argparse
import json
import subprocess
import sys
import time
import unittest

import access
access.initialize()

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bridge as b
import models

SOURCE = 'idsse'


def smoke_path(family):
    return b.OUT / 'smoke' / family


def fit(family):
    access.restrict('smoke-fit', SOURCE)
    data = [b.c.load(name, b.regime(name), with_features=True) for name in b.train_names(SOURCE)]
    fitted = models.fit(data, SOURCE, family, steps=1)
    path = smoke_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, path.with_suffix('.pt'))
    b.c.write_json(path.with_suffix('.json'), dict(family=family, training_names=list(fitted['training_names']),
        parameters=models.PARAMETERS, training=fitted['training'], steps=1, metric_evaluation=False,
        checkpoint_sha256=b.c.digest(path.with_suffix('.pt')), guard=access.report()))
    print('GUARDED REAL FIT', family, '7source matches/3seeds/1update', flush=True)


def infer(family, replay=False):
    access.restrict('smoke-replay' if replay else 'smoke-predict', SOURCE)
    path = smoke_path(family)
    fitted = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
    name = b.test_names(SOURCE)[0]
    keys = []
    for condition in b.CONDITIONS:
        observed = b.load_observations(name, condition)
        prediction = models.predict(fitted, observed)['probability']
        codes = b.decode(b.result(prediction), b.decoder_observations(observed), fitted['training_stats'])
        output = path.with_name(path.name + '_' + condition).with_suffix('.npz')
        if replay:
            with np.load(output, allow_pickle=False) as archive:
                if not np.array_equal(prediction, archive['probability']) or not np.array_equal(codes, archive['codes']):
                    raise ValueError('Guarded smoke model probability/code replay differs')
        else:
            np.savez_compressed(output, probability=prediction, codes=codes)
        keys.append(condition)
    b.c.write_json(path.with_name(path.name + ('_replay' if replay else '_predict')).with_suffix('.json'),
                   dict(family=family, match=name, conditions=keys, guard=access.report(), metric_evaluation=False,
                        probability_and_codes_identical=replay))
    print('GUARDED SMOKE REPLAY IDENTICAL' if replay else 'GUARDED SMOKE PREDICT', family, 'NO TEST LABEL ACCESS', flush=True)


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(b.HERE), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    records = []
    for family in models.FAMILIES:
        for stage in ('fit', 'predict', 'replay'):
            subprocess.run([sys.executable, '-B', str(b.HERE / 'check.py'), '--family', family, '--stage', stage], check=True)
        path = smoke_path(family)
        metadata = json.loads(path.with_suffix('.json').read_text())
        predicted = json.loads(path.with_name(path.name + '_predict').with_suffix('.json').read_text())
        replayed = json.loads(path.with_name(path.name + '_replay').with_suffix('.json').read_text())
        if not replayed['probability_and_codes_identical'] or predicted['match'] != replayed['match'] or replayed['conditions'] != list(b.CONDITIONS):
            raise ValueError('Smoke target inference replay incomplete')
        if any(value['guard']['nonprobe_blocked_attempts'] for value in (metadata, predicted, replayed)):
            raise ValueError('Smoke attempted forbidden read')
        records.append(dict(family=family, parameters=metadata['parameters'], fit=metadata, predict=predicted, replay=replayed))
    b.c.write_json(b.OUT / 'checks.json', dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        smoke=records, new_smoke_neural_seed_fits=15, updates_per_smoke_fit=1, replayed_condition_arrays=10,
        metric_evaluation=False, target_game_used=False, seconds=time.monotonic() - started, guard=access.report()))
    print('CHECKS PASS', result.testsRun, 'REAL SEED SMOKES', 15, 'REPLAYED TARGET CONDITIONS', 10, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=models.FAMILIES)
    parser.add_argument('--stage', choices=('fit', 'predict', 'replay'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == 'fit': fit(args.family)
        elif args.stage == 'predict': infer(args.family)
        elif args.stage == 'replay': infer(args.family, True)
        else: main()
