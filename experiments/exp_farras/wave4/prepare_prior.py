import argparse
import os
import sys

os.environ['FARRAS_TRAIN_DEVICE'] = 'cpu'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

import numpy as np

import campaign as c


def prepare(phase):
    if phase == 'holdout' and not (c.OUT / 'selection.json').exists():
        raise FileNotFoundError('Freeze wave4 selection before holdout inference')
    sys.path.insert(0, str(c.PREVIOUS))
    previous = c.import_path('wave4_previous_runner', c.PREVIOUS / 'run.py')
    previous.frozen_selection()
    hybrid = previous.load_fitted('hybrid')
    anchor = previous.load_fitted('anchor')
    target_cases = [(name, regime) for name, regime, _ in c.dev_cases()] if phase == 'dev' else [
        *c.holdout_cases(), ('target', 'clean')]
    audit_path = c.CACHE / f'prior_{phase}_audit.json'
    audit = dict(previous_selection_sha256=c.digest(c.PREVIOUS / 'out/selection.json'), phase=phase, packs={})
    directory = c.CACHE / 'prior'
    directory.mkdir(parents=True, exist_ok=True)
    for name, regime in target_cases:
        data = c.load(name, regime)
        if phase == 'dev' and (name in c.IDSSE_DEV or name.startswith('metrica')):
            tag = 'loo_' + name if name in c.IDSSE_DEV else 'final'
            neural = previous.read_prediction(previous.prediction_path('hybrid', tag, name, regime))
            tree = previous.read_prediction(previous.prediction_path('anchor', tag, name, regime))
            provenance = 'Prior campaign held-out IDSSE or excluded-Metrica anchor fold; no current test-match training'
        else:
            neural = previous.predict_model('hybrid', hybrid, data)
            tree = previous.anchor_predict(anchor, data)
            provenance = 'Prior final models, zero-shot on this provider/match within Farras'
        combined = previous.ensemble.combine(neural, tree, previous.observations(data),
            previous.settings(previous.ensemble, 'tree25'))
        output = directory / f'{name}_{regime}.npz'
        np.savez_compressed(output, probability=combined['probability'], boundary=combined['boundary'],
            hybrid_probability=neural['probability'], hybrid_boundary=neural['boundary'],
            anchor_probability=tree['probability'], anchor_boundary=tree['boundary'])
        audit['packs'][f'{name}_{regime}'] = dict(sha256=c.digest(output), source=provenance,
            samples=len(combined['probability']))
        c.write_json(audit_path, audit)
        print('PRIOR', phase, name, regime, flush=True)
    print('PRIOR READY', phase, len(audit['packs']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['dev', 'holdout'])
    prepare(parser.parse_args().phase)
