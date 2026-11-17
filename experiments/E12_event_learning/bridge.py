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
PREVIOUS = HERE.parent / 'E11_spell_training'
FAIR = HERE.parent / 'E08_fair_idsse'
CONTEXT = HERE.parent / 'E09_cross_branch'
_spec = importlib.util.spec_from_file_location('event_learning_fair_bridge', FAIR / 'bridge.py')
fair = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fair
_spec.loader.exec_module(fair)
c, original, training = fair.c, fair.original, fair.training
result = fair.probability_result
MATCHES = fair.MATCHES
decoder = fair.decoder

import numpy as np

old_weights = c.import_path('event_learning_previous_weights', PREVIOUS / 'weights.py')
_saved_weights = sys.modules.get('weights')
sys.modules['weights'] = old_weights
try:
    old_sampling = c.import_path('event_learning_previous_sampling', PREVIOUS / 'sampling.py')
finally:
    if _saved_weights is None:
        sys.modules.pop('weights', None)
    else:
        sys.modules['weights'] = _saved_weights
consensus = c.import_path('event_learning_frozen_consensus', HERE.parent / 'E10_temporal_methods/consensus.py')


def load_context(name):
    with np.load(CONTEXT / 'cache' / f'{name}.npz', allow_pickle=False) as archive:
        return archive['control']


def observations(data):
    observed = c.base.decoder_observations(fair.observations(data))
    observed['contact_control'] = np.asarray(load_context(data['_name'])[0], dtype=np.float64)
    if observed['contact_control'].shape != (len(observed['X']), 7) or not np.isfinite(observed['contact_control']).all():
        raise ValueError('Contact observation timeline mismatch')
    return observed


def baseline(prediction, observed, stats):
    return decoder.decode(prediction, observed, fair.SPEC, stats)['codes']


def decode(prediction, bundle, observed, stats, mode):
    if mode == 'dead45':
        return baseline(prediction, observed, stats)
    if mode != 'consensus':
        raise ValueError('Unknown frozen decoder')
    return consensus.decode(prediction, bundle, observed, stats, dict(alpha=.5, contact_beta=.5))


def read_old(family, heldout):
    path = FAIR / 'out/jobs' / f'{family}_{heldout}.npz'
    metadata = json.loads(path.with_suffix('.json').read_text())
    identity = metadata['identity']
    if identity['family'] != family or identity['heldout'] != heldout or identity['manifest_sha256'] != c.digest(FAIR / 'out/preregistered.json'):
        raise ValueError('Original OOF probability identity mismatch')
    fair.assert_fold(tuple(identity['training_names']), heldout)
    if metadata['sha256'] != c.digest(path):
        raise ValueError('Original OOF probability bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        return result(archive['probability'])


def read_event(heldout):
    path = PREVIOUS / 'out/jobs' / f'event_tcn_{heldout}.npz'
    metadata = json.loads(path.with_suffix('.json').read_text())
    expected = dict(family='event_tcn', heldout=heldout, training_names=list(fair.split_names(heldout)),
                    seeds=[0, 1, 2], steps=500, manifest_sha256=c.digest(PREVIOUS / 'out/preregistered.json'))
    if metadata['identity'] != expected or metadata['actual_fit_names'] != expected['training_names']:
        raise ValueError('Previous event OOF training identity mismatch')
    if metadata['sha256'] != c.digest(path):
        raise ValueError('Previous event OOF probability bytes changed')
    proof = json.loads((PREVIOUS / 'out/replays' / f'event_tcn_{heldout}.json').read_text())
    if proof != dict(identity=expected, checkpoint_sha256=metadata['checkpoint_sha256'],
                     probability_sha256=metadata['sha256'], identical_full_probabilities=True):
        raise ValueError('Previous event saved-model replay proof mismatch')
    with np.load(path, allow_pickle=False) as archive:
        return result(archive['probability'])


def old_bundle(heldout):
    return dict(short=read_old('short_tcn', heldout), setseq=read_old('setseq', heldout),
                old_event=read_event(heldout))


def legacy_probability(bundle):
    return result(.75 * bundle['short']['probability'] + .25 * bundle['setseq']['probability'])


def research_probability(bundle):
    return result(.5 * bundle['short']['probability'] + .25 * bundle['setseq']['probability'] + .25 * bundle['old_event']['probability'])
