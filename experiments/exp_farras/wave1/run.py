"""Execute fixed methods, select on external matches, then audit public scores separately."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data import CACHE, HERE, REPO, WORKSPACE, digest, expand, inputs

sys.path.insert(0, str(REPO / 'evaluation'))
import possession_metrics as pm

OUT = HERE / 'out'


def load(name):
    with np.load(CACHE / f'{name}.npz', allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def training(datasets):
    xs, ys, blocks, offset = [], [], [], 0
    for dataset in datasets:
        for key in ('X', 'X_occluded'):
            xs.append(dataset[key])
            ys.append(dataset['y'])
            blocks.extend((int(a) + offset, int(b) + offset) for a, b in dataset['blocks'])
            offset += len(dataset[key])
    return dict(X=np.concatenate(xs), y=np.concatenate(ys), blocks=blocks, fps=5)


def prediction_input(dataset, regime='X'):
    return dict(X=dataset[regime].copy(), blocks=[tuple(map(int, b)) for b in dataset['blocks']], fps=5)


def checked_predict(module, train, test, config):
    with threadpool_limits(limits=1):
        result = np.asarray(module.fit_predict(train, test, dict(config)))
    if result.shape != (len(test['X']),) or not np.isin(result, [0, 1, 2]).all():
        raise ValueError(f'Invalid labels or shape from {module.__name__}: {result.shape}')
    return result.astype(np.int8)


def metrics(gt, pred):
    result = pm.evaluate(np.where(gt < 0, 3, gt).astype(np.int8), pred, bootstrap_n=0)
    return {key: result[key] for key in ('spell_f1_mean', 'macro_f1', 'accuracy',
            'n_gt_spells', 'n_pred_spells', 'mean_spell_duration_s')}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_method(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if len(module.CONFIGS) != 3 or len({c['name'] for c in module.CONFIGS}) != 3:
        raise ValueError('Method must have three distinct preregistered configurations')
    external = [load('external_1'), load('external_2')]
    target = load('target')
    directory = OUT / path.stem
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for config in module.CONFIGS:
        start = time.monotonic()
        folds = []
        for heldout in (0, 1):
            train = training([external[1 - heldout]])
            test = external[heldout]
            for regime in ('X', 'X_occluded'):
                prediction = checked_predict(module, train, prediction_input(test, regime), config)
                full = expand(prediction, test)
                fold = metrics(test['y25'], full)
                fold.update(heldout_game=heldout + 1, regime='clean' if regime == 'X' else 'occluded')
                folds.append(fold)
                np.save(directory / f'{config["name"]}_game{heldout + 1}_{regime}.npy', full)
        prediction = checked_predict(module, training(external), prediction_input(target), config)
        full = expand(prediction, target)
        table = pd.DataFrame({'frame': np.arange(len(full)), 'label': np.array(pm.LABELS)[full]})
        reason = pm.validate_submission(table, 199319)
        if reason:
            raise ValueError(reason)
        submission = directory / f'{config["name"]}.csv'
        table.to_csv(submission, index=False, lineterminator='\n')
        record = dict(method=path.stem, config=config, external_folds=folds,
            external_mean_spellF1=float(np.mean([f['spell_f1_mean'] for f in folds])),
            external_mean_macroF1=float(np.mean([f['macro_f1'] for f in folds])),
            target_label_counts={label: int((full == k).sum()) for k, label in enumerate(pm.LABELS)},
            submission=str(submission.relative_to(HERE)), submission_sha256=digest(submission),
            code_sha256=digest(path), seconds=time.monotonic() - start)
        records.append(record)
        write_json(directory / 'results.json', records)
        print(f'{path.stem}/{config["name"]}: external spellF1={record["external_mean_spellF1"]:.4f} '
              f'macroF1={record["external_mean_macroF1"]:.4f} seconds={record["seconds"]:.1f}', flush=True)
    return records


def execute_all(workers):
    paths = sorted((HERE / 'methods').glob('m*.py'))
    if len(paths) != 8:
        raise ValueError(f'Expected eight methods, found {len(paths)}')
    OUT.mkdir(exist_ok=True)
    write_json(OUT / 'preregistered.json', dict(
        selection='Mean external spellF1 over game1/game2 x clean/occluded; tie mean macroF1 then name.',
        code={path.name: digest(path) for path in paths},
        harness={name: digest(HERE / name) for name in ('data.py', 'run.py')},
        workers=workers, seed=2027, fps=5,
        public_gt_used_for_selection=False))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(subprocess.run, [sys.executable, '-B', str(Path(__file__)),
                   'method', '--path', str(path)], cwd=str(WORKSPACE), check=True): path for path in paths}
        errors = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(f'{futures[future].name}: {error}')
        if errors:
            raise RuntimeError('\n'.join(errors))
    collect()


def collect():
    records = []
    preregistered = json.loads((OUT / 'preregistered.json').read_text())
    for name, expected in preregistered['code'].items():
        if digest(HERE / 'methods' / name) != expected:
            raise ValueError(f'Method changed after preregistration: {name}')
        path = OUT / Path(name).stem / 'results.json'
        result = json.loads(path.read_text())
        if len(result) != 3:
            raise ValueError(f'Incomplete experiment: {name}')
        records.extend(result)
    records.sort(key=lambda r: (-r['external_mean_spellF1'], -r['external_mean_macroF1'],
                               r['method'], r['config']['name']))
    for record in records:
        if digest(HERE / record['submission']) != record['submission_sha256']:
            raise ValueError('Prediction changed before freeze')
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    for path in inputs():
        if digest(path) != manifest[str(path.relative_to(WORKSPACE))]:
            raise ValueError(f'Read-only source changed: {path}')
    selection = dict(selected=records[0]['submission'], reason=preregistered['selection'],
                     public_gt_seen_by_campaign=False, records=records,
                     data_manifest_sha256=digest(CACHE / 'manifest.json'))
    write_json(OUT / 'selection.json', selection)
    pd.DataFrame([dict(method=r['method'], config=r['config']['name'],
         external_spellF1=r['external_mean_spellF1'], external_macroF1=r['external_mean_macroF1'],
         seconds=r['seconds']) for r in records]).to_csv(OUT / 'external_scoreboard.csv', index=False)
    print('FROZEN SELECTION', selection['selected'], flush=True)


def public_audit():
    import run_evaluation as official
    selection_path = OUT / 'selection.json'
    selection = json.loads(selection_path.read_text())
    gt = pm.label_codes(pm.read_label_csv(official.find_gt('public')))
    rows = []
    for record in selection['records']:
        path = HERE / record['submission']
        if digest(path) != record['submission_sha256']:
            raise ValueError(f'Frozen prediction changed: {path}')
        result, reason = official.score_submission(str(path), gt, pm.TAUS, 0)
        if reason:
            raise ValueError(reason)
        rows.append(dict(submission=record['submission'], selected=record['submission'] == selection['selected'],
                         spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'],
                         possessions=result['n_pred_spells']))
    write_json(OUT / 'public_audit.json', dict(selection_sha256=digest(selection_path),
        warning='Post-freeze diagnostic only; not a selection or tuning set.', records=rows))
    pd.DataFrame(rows).to_csv(OUT / 'public_scoreboard.csv', index=False)
    selected = next(row for row in rows if row['selected'])
    print('Public audit complete:', len(rows), 'valid CSVs; selected:', selected, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['all', 'method', 'collect', 'public'])
    parser.add_argument('--path')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'all':
        execute_all(args.workers)
    elif args.action == 'method':
        run_method(args.path)
    elif args.action == 'collect':
        collect()
    else:
        public_audit()


if __name__ == '__main__':
    main()
