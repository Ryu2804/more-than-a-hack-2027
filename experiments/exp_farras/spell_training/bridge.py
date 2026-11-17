import importlib.util
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
PREVIOUS = HERE.parent / 'temporal_methods'
FAIR = HERE.parent / 'fair_idsse'
_spec = importlib.util.spec_from_file_location('spell_training_fair_bridge', FAIR / 'bridge.py')
fair = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fair
_spec.loader.exec_module(fair)
c, original, training = fair.c, fair.original, fair.training
result = fair.probability_result
MATCHES = fair.MATCHES
contextual = HERE.parent / 'cross_branch'

import numpy as np


def baseline(prediction, observations, stats):
    return fair.decoder.decode(prediction, observations, fair.SPEC, stats)['codes']


def observations(data):
    observed = c.base.decoder_observations(fair.observations(data))
    with np.load(contextual / 'cache' / f"{data['_name']}.npz", allow_pickle=False) as archive:
        observed['contact_control'] = archive['control'][0].astype(np.float64)
    if observed['contact_control'].shape != (len(data['X']), 7) or not np.isfinite(observed['contact_control']).all():
        raise ValueError('Contact features on incorrect timeline')
    return observed


def read_old(family, heldout):
    path = FAIR / 'out/jobs' / f'{family}_{heldout}.npz'
    metadata = json.loads(path.with_suffix('.json').read_text())
    identity = metadata['identity']
    if identity['family'] != family or identity['heldout'] != heldout or identity['manifest_sha256'] != c.digest(FAIR / 'out/preregistered.json'):
        raise ValueError('OOF probability identity mismatch')
    fair.assert_fold(tuple(identity['training_names']), heldout)
    if metadata['sha256'] != c.digest(path):
        raise ValueError('OOF probability bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        return result(archive['probability'])


def old_bundle(heldout):
    return dict(short=read_old('short_tcn', heldout), setseq=read_old('setseq', heldout))


def control_probability(bundle):
    return result(.75 * bundle['short']['probability'] + .25 * bundle['setseq']['probability'])


def consensus_decode(prediction, bundle, observed, stats):
    # Frozen consensus module requires only decoder/fair bridge aliases.
    return consensus.decode(prediction, bundle, observed, stats, dict(alpha=.5, contact_beta=.5))


decoder = fair.decoder
consensus = c.import_path('spell_training_frozen_consensus', PREVIOUS / 'consensus.py')
