import importlib.util
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
CACHE = HERE / 'cache'
FAIR = HERE.parent / 'fair_idsse'
_spec = importlib.util.spec_from_file_location('provider_transfer_fair_bridge', FAIR / 'bridge.py')
fair = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fair
_spec.loader.exec_module(fair)
c, original, training = fair.c, fair.original, fair.training
result = fair.probability_result
IDSSE = fair.MATCHES
SC = tuple(sorted((*c.SC_DEV, *c.SC_HOLD)))
PROVIDERS = dict(idsse=IDSSE, skillcorner=SC)
CONDITIONS = ('clean', 'gap')
feature_builder = c.import_path('provider_transfer_features', c.PREVIOUS / 'features.py')
wave1 = c.import_path('provider_transfer_x16', HERE.parent / 'wave1/data.py')
old_weights = c.import_path('provider_transfer_weights', HERE.parent / 'spell_training/weights.py')
_previous_weights = sys.modules.get('weights')
sys.modules['weights'] = old_weights
try:
    old_sampling = c.import_path('provider_transfer_sampling', HERE.parent / 'spell_training/sampling.py')
finally:
    if _previous_weights is None:
        sys.modules.pop('weights', None)
    else:
        sys.modules['weights'] = _previous_weights

import numpy as np


def train_names(source):
    if source not in PROVIDERS:
        raise ValueError('Unknown source provider')
    return PROVIDERS[source]


def test_names(source):
    if source not in PROVIDERS:
        raise ValueError('Unknown source provider')
    return PROVIDERS['skillcorner' if source == 'idsse' else 'idsse']


def provider(name):
    if name in IDSSE:
        return 'idsse'
    if name in SC:
        return 'skillcorner'
    raise ValueError('Only declared external matches are permitted')


def regime(name):
    return 'r0' if provider(name) == 'idsse' else 'clean'


def original_pack(name):
    return c.base.pack_path(name, regime(name))


def original_features(name):
    return c.PREVIOUS / 'cache/features' / f'{name}_{regime(name)}.npz'


def case_paths(name, condition):
    provider(name)
    if condition not in CONDITIONS:
        raise ValueError('Unknown predeclared observation condition')
    return CACHE / 'observations' / f'{name}_{condition}.npz', CACHE / 'features' / f'{name}_{condition}.npz'


def load_observations(name, condition):
    path, feature_path = case_paths(name, condition)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) & {'y', 'y25'}:
            raise ValueError('Label leakage in observation-only carrier')
        data = {key: archive[key] for key in archive.files}
    with np.load(feature_path, allow_pickle=False) as archive:
        data['_features'] = {key: archive[key] for key in archive.files}
    data.update(_name=name, _source=provider(name), _regime=condition)
    return data


def decoder_observations(data):
    if 'y' in data or 'y25' in data:
        raise ValueError('Decoder received test labels')
    return c.base.decoder_observations(data)


def decode(prediction, observed, stats):
    return fair.decoder.decode(prediction, observed, fair.SPEC, stats)['codes']


def statistics(datasets):
    return fair.statistics(datasets)


def score(codes, observed, labels):
    values = np.asarray(codes)
    if values.shape != labels['y'].shape or not np.isin(values, (0, 1, 2)).all():
        raise ValueError('Prediction grid or label codes differ')
    for key in ('blocks', 'blocks25', 'n_frames', 'fps', 'sampled'):
        if not np.array_equal(observed[key], labels[key]):
            raise ValueError('Sealed observation and label timelines differ: ' + key)
    return fair.score(values, {**observed, **labels})
