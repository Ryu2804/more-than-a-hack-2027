import sys

sys.dont_write_bytecode = True

import numpy as np

import campaign as c
import features


def cases():
    return [(name, f'r{round_index}') for name in (*c.IDSSE_DEV, *c.IDSSE_HOLD) for round_index in range(3)] + [
        (name, 'clean') for name in (*c.SC_DEV, *c.SC_HOLD)] + [
        (f'metrica{number}', regime) for number in (1, 2, 3) for regime in ('clean', 'r0')] + [('target', 'clean')]


def prepare():
    directory = c.CACHE / 'features'
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = c.CACHE / 'feature_audit.json'
    if audit_path.exists():
        import json
        old = json.loads(audit_path.read_text())
    else:
        old = {'feature_code_sha256': c.digest(c.HERE / 'features.py'), 'packs': {}}
    code_hash = c.digest(c.HERE / 'features.py')
    if old['feature_code_sha256'] != code_hash:
        raise ValueError('Feature code changed; use a new feature-cache directory')
    for name, regime in cases():
        key = f'{name}_{regime}'
        output = directory / (key + '.npz')
        input_hash = c.digest(c.pack_path(name, regime))
        if key in old['packs'] and output.exists():
            previous = old['packs'][key]
            if previous['input_sha256'] != input_hash or previous['output_sha256'] != c.digest(output):
                raise ValueError('Prepared features changed: ' + key)
            continue
        data = c.load(name, regime)
        result = features.build(c.observations(data))
        if result['player'].shape != (2, len(data['X']), 23):
            raise ValueError('Invalid two-view player features')
        for field in ('player', 'ball', 'control', 'pp'):
            if not np.isfinite(result[field]).all():
                raise ValueError('Nonfinite feature field: ' + field)
        result['player'] = np.asarray(result['player'], dtype=np.float32)
        result['ball'] = np.asarray(result['ball'], dtype=np.float32)
        result['control'] = np.asarray(result['control'], dtype=np.float32)
        result['pp'] = np.asarray(result['pp'], dtype=np.float16)
        result['blocks'] = np.asarray(result['blocks'], dtype=np.int64)
        np.savez_compressed(output, **result)
        old['packs'][key] = dict(input_sha256=input_hash, output_sha256=c.digest(output),
            samples=len(data['X']), valid_fraction=float(np.mean(result['valid'])),
            player_shape=list(result['player'].shape), pp_shape=list(result['pp'].shape))
        c.write_json(audit_path, old)
        print('FEATURES', key, 'samples', len(data['X']), flush=True)
    print('FEATURE CACHE READY', len(old['packs']), flush=True)


if __name__ == '__main__':
    prepare()
