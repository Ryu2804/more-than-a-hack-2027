import importlib.util
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
PREVIOUS = HERE.parent / 'E09_cross_branch'
FAIR = HERE.parent / 'E08_fair_idsse'
_spec = importlib.util.spec_from_file_location('temporal_fair_bridge', FAIR / 'bridge.py')
fair = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fair
_spec.loader.exec_module(fair)
c = fair.c
result = probability_result = fair.probability_result
decoder = fair.decoder
normalize_blocks = decoder._normalize_blocks
MATCHES = fair.MATCHES

import numpy as np


def baseline(prediction, observations, stats):
    return decoder.decode(prediction, observations, fair.SPEC, stats)['codes']


def load_context(name):
    with np.load(PREVIOUS / 'cache' / f'{name}.npz', allow_pickle=False) as archive:
        return archive['control']


def read_prediction(family, heldout):
    path = FAIR / 'out/jobs' / f'{family}_{heldout}.npz'
    metadata = json.loads(path.with_suffix('.json').read_text())
    identity = metadata['identity']
    if identity['family'] != family or identity['heldout'] != heldout or identity['manifest_sha256'] != c.digest(FAIR / 'out/preregistered.json'):
        raise ValueError('OOF prediction identity mismatch')
    fair.assert_fold(tuple(identity['training_names']), heldout)
    if metadata['sha256'] != c.digest(path):
        raise ValueError('OOF prediction bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        return result(archive['probability'])


def bundle(heldout):
    return {key: read_prediction(family, heldout) for key, family in
            (('short', 'short_tcn'), ('setseq', 'setseq'))}


def fuse(values):
    return result(.75 * values['short']['probability'] + .25 * values['setseq']['probability'])


def observations(data):
    observed = c.base.decoder_observations(fair.observations(data))
    context = np.asarray(load_context(data['_name'])[0], dtype=np.float64)
    if context.shape != (len(observed['X']), 7) or not np.isfinite(context).all():
        raise ValueError('Context timeline mismatch')
    observed['contact_control'] = context
    if 'y' in observed or 'y25' in observed:
        raise ValueError('Labels must not enter decoder observations')
    return observed


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
