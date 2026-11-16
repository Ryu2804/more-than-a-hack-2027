import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT, CACHE = HERE / 'out', HERE / 'cache'
PREVIOUS = HERE.parent / 'fair_idsse'
sys.path.append(str(PREVIOUS))
import importlib.util

_spec = importlib.util.spec_from_file_location('cross_fair_bridge', PREVIOUS / 'bridge.py')
fair = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fair
_spec.loader.exec_module(fair)
c, original = fair.c, fair.original
probability_result, tree = fair.probability_result, fair.tree
result = probability_result
MATCHES = fair.MATCHES
classic = c.import_path('cross_classic_source', PREVIOUS / 'classic.py')
reference = c.import_path('cross_observation_reference', HERE.parent / 'wave3/reference/jeremy_data.py')
reference.ROOT = HERE.parent / 'wave3/cache/idsse_parser'

import numpy as np


def load_features(name):
    with np.load(CACHE / f'{name}.npz', allow_pickle=False) as archive:
        return archive['control']


def attach_controls(data, family):
    if family not in ('contact_tcn', 'memory_tcn', 'combined_tcn'):
        raise ValueError('Unknown contextual TCN family')
    control = load_features(data['_name']).copy()
    if family == 'contact_tcn':
        control[:, :, 4:] = 0
    elif family == 'memory_tcn':
        control[:, :, :4] = 0
    features = {**data['_features'], 'control': control}
    return {**data, '_features': features}


def validate_names(train_names, heldout):
    fair.assert_fold(tuple(train_names), heldout)


def read_old_prediction(family, heldout):
    path = PREVIOUS / 'out/jobs' / f'{family}_{heldout}.npz'
    metadata = __import__('json').loads(path.with_suffix('.json').read_text())
    identity = metadata['identity']
    old_hash = c.digest(PREVIOUS / 'out/preregistered.json')
    if identity['family'] != family or identity['heldout'] != heldout or identity['manifest_sha256'] != old_hash:
        raise ValueError('Original OOF cache belongs to another model/fold/protocol')
    names = metadata['identity']['training_names']
    validate_names(names, heldout)
    if metadata['sha256'] != c.digest(path):
        raise ValueError('Original OOF probability cache changed')
    with np.load(path, allow_pickle=False) as archive:
        return result(archive['probability'])
