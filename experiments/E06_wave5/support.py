import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'E05_wave4'
REPO = HERE.parents[1]
CACHE = HERE / 'cache'
OUT = HERE / 'out'
if str(PREVIOUS) not in sys.path:
    sys.path.append(str(PREVIOUS))
import campaign as base

IDSSE_DEV, IDSSE_HOLD = base.IDSSE_DEV, base.IDSSE_HOLD
SC_DEV, SC_HOLD = base.SC_DEV, base.SC_HOLD
DEV, HOLD, FOLD = base.DEV, base.HOLD, base.FOLD
import_path, digest, write_json = base.import_path, base.digest, base.write_json
source, observations = base.source, base.observations
training, training_stats = base.training, base.training_stats
metrics, expand, pm = base.metrics, base.expand, base.pm
dev_cases = base.dev_cases


def pack_path(name, regime):
    return CACHE / 'data' / f'{name}_stress.npz' if regime == 'stress' else base.pack_path(name, regime)


def load(name, regime, with_features=False, light=False):
    if regime != 'stress':
        return base.load(name, regime, with_features=with_features, light=light)
    if name not in SC_HOLD:
        raise ValueError('Stress is reserved for SkillCorner holdout, after freeze')
    with np.load(pack_path(name, regime), allow_pickle=False) as archive:
        keys = ('y', 'y25', 'blocks', 'blocks25', 'n_frames', 'fps', 'sampled') if light else archive.files
        data = {key: archive[key] for key in keys}
    data.update(_name=name, _source=source(name), _regime=regime)
    if with_features:
        with np.load(CACHE / 'features' / f'{name}_stress.npz', allow_pickle=False) as archive:
            data['_features'] = {key: archive[key] for key in archive.files}
    return data


def holdout_cases():
    return [*base.holdout_cases(), *((name, 'stress') for name in SC_HOLD)]
