import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wave5'))
import support as c
import training_models as training

original = training._WAVE4
sys.path.append(str(HERE.parent / 'wave3'))
sequence = c.import_path('fair_setseq_source', HERE.parent / 'wave3/models.py')
tree = c.import_path('fair_tree_source', HERE.parent / 'wave2/support.py')
decoder = c.import_path('fair_shared_decoder', HERE.parent / 'wave5/segment_decoder.py')
SPEC = dict(next(row for row in decoder.CONFIGS if row['name'] == 'dead45'))
MATCHES = tuple(sorted((*c.IDSSE_DEV, *c.IDSSE_HOLD)))

import numpy as np


def probability_result(probability):
    joint = np.array(probability, dtype=np.float64, copy=True)
    if joint.ndim != 2 or joint.shape[1] != 3 or not np.isfinite(joint).all() or np.any(joint < 0) or np.any(joint.sum(1) <= 0):
        raise ValueError('Invalid joint probabilities')
    joint /= joint.sum(1, keepdims=True)
    live = joint[:, :2].sum(1)
    team = np.divide(joint[:, 0], live, out=np.full(len(joint), 0.5), where=live > 0)
    return dict(probability=joint, p_dead=joint[:, 2], p_team=team, has_boundary=False)


def split_names(heldout):
    if heldout not in MATCHES:
        raise ValueError('Unknown held-out match')
    return tuple(name for name in MATCHES if name != heldout)


def assert_fold(train_names, test_name):
    if len(train_names) != 6 or len(set(train_names)) != 6 or set(train_names) != set(split_names(test_name)):
        raise ValueError('Fold must train exactly six other unique matches')


def observations(data):
    return {key: value for key, value in data.items() if key not in ('y', 'y25')}


def statistics(datasets):
    rates, shares = [], []
    for data in datasets:
        valid = dead = spells = 0
        for start, end in data['blocks25']:
            labels = data['y25'][start:end]
            valid += int(np.sum(labels >= 0))
            dead += int(np.sum(labels == 2))
            for state in (0, 1):
                active = labels == state
                spells += int(np.sum(active & ~np.r_[False, active[:-1]]))
        if not valid:
            raise ValueError('No training labels')
        rates.append(spells / (valid / 25 / 60))
        shares.append(dead / valid)
    return dict(rate=float(np.mean(rates)), dead_share=float(np.mean(shares)))


def score(codes, data):
    values = np.asarray(codes)
    if values.shape != data['y'].shape or not np.isin(values, (0, 1, 2)).all():
        raise ValueError('Invalid prediction grid')
    for (a, b), (start, end) in zip(data['blocks'], data['blocks25']):
        if b - a != (end - start + 4) // 5:
            raise ValueError('5fps/25fps half alignment mismatch')
    if len(data['blocks']) != len(data['blocks25']) or len(data['y25']) != int(data['n_frames']):
        raise ValueError('Timeline mismatch')
    return c.metrics(values.astype(np.int8), data)
