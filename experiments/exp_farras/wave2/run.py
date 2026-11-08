"""External selection precedes holdout and public diagnostics; each phase checks frozen hashes."""
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

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / 'wave1'
REPO = HERE.parents[2]
WORKSPACE = REPO.parent
OUT = HERE / 'out'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FIRST))
spec = importlib.util.spec_from_file_location('first_campaign', FIRST / 'run.py')
first = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first)
pm = first.pm


def module_for(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_fit(module, train, config):
    with threadpool_limits(limits=1):
        return module.fit(train, dict(config))


def model_predict(module, model, data, config, regime='X'):
    test = first.prediction_input(data, regime)
    with threadpool_limits(limits=1):
        prediction = np.asarray(module.predict(model, test, dict(config)))
    if prediction.shape != (len(test['X']),) or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError(f'Invalid prediction from {module.__name__}')
    return first.expand(prediction.astype(np.int8), data)


def save_submission(path, prediction):
    frame = pd.DataFrame({'frame': np.arange(len(prediction)), 'label': np.asarray(pm.LABELS)[prediction]})
    reason = pm.validate_submission(frame, 199319)
    if reason:
        raise ValueError(reason)
    frame.to_csv(path, index=False, lineterminator='\n')
    return first.digest(path)


def run_method(path):
    path = Path(path)
    module = module_for(path)
    if len(module.CONFIGS) != 3 or len({c['name'] for c in module.CONFIGS}) != 3:
        raise ValueError('Three unique preregistered configs required')
    external = [first.load('external_1'), first.load('external_2')]
    target = first.load('target')
    directory = OUT / path.stem
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for config in module.CONFIGS:
        started = time.monotonic()
        folds = []
        for heldout in (0, 1):
            fitted = model_fit(module, first.training([external[1 - heldout]]), config)
            for regime in ('X', 'X_occluded'):
                prediction = model_predict(module, fitted, external[heldout], config, regime)
                row = first.metrics(external[heldout]['y25'], prediction)
                row.update(heldout_game=heldout + 1, regime='clean' if regime == 'X' else 'occluded')
                folds.append(row)
                np.save(directory / f'{config["name"]}_game{heldout + 1}_{regime}.npy', prediction)
        fitted = model_fit(module, first.training(external), config)
        prediction = model_predict(module, fitted, target, config)
        csv_path = directory / f'{config["name"]}.csv'
        csv_hash = save_submission(csv_path, prediction)
        record = dict(method=path.stem, config=config, external_folds=folds,
            external_mean_spellF1=float(np.mean([r['spell_f1_mean'] for r in folds])),
            external_mean_macroF1=float(np.mean([r['macro_f1'] for r in folds])),
            seconds=time.monotonic() - started, code_sha256=first.digest(path),
            submission=csv_path.relative_to(HERE).as_posix(), submission_sha256=csv_hash,
            target_label_counts={label: int((prediction == k).sum()) for k, label in enumerate(pm.LABELS)})
        records.append(record)
        first.write_json(directory / 'results.json', records)
        print(f'{path.stem}/{config["name"]}: external spellF1={record["external_mean_spellF1"]:.4f} '
              f'macroF1={record["external_mean_macroF1"]:.4f} seconds={record["seconds"]:.1f}', flush=True)
    return records


def source_manifest():
    paths = [HERE / 'support.py', HERE / 'run.py', HERE / 'prepare_holdout.py',
             FIRST / 'run.py', FIRST / 'data.py', FIRST / 'out/selection.json',
             HERE / 'cache/holdout_audit.json', HERE / 'cache/holdout_3.npz']
    paths += [FIRST / 'cache' / name for name in ['external_1.npz', 'external_2.npz', 'target.npz', 'manifest.json']]
    paths += first.inputs()
    return {path.relative_to(WORKSPACE).as_posix(): first.digest(path) for path in paths}


def verify_sources(manifest):
    for relative, expected in manifest.items():
        if first.digest(WORKSPACE / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')


def execute_all(workers):
    paths = sorted((HERE / 'methods').glob('w*.py'))
    if len(paths) != 16:
        raise ValueError(f'Expected16 methods, found {len(paths)}')
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen; use new campaign directory rather than overwrite')
    registry = dict(selection='Mean spellF1 Game1/Game2 x clean/occluded; ties macroF1,method,config.',
        seed=2027, fps=5, workers=workers, input_sha256=source_manifest(),
        methods={p.name: dict(sha256=first.digest(p), configs=module_for(p).CONFIGS) for p in paths},
        public_gt_used_for_selection=False, game3_used_for_selection=False,
        prior_exposure='Wave1 diagnostics were known before wave2; no target GT or Game3 used to choose configurations.')
    first.write_json(OUT / 'preregistered.json', registry)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(subprocess.run, [sys.executable, '-B', str(Path(__file__)),
                'method', '--path', str(path)], cwd=str(WORKSPACE), check=True): path for path in paths}
        errors = []
        for job in concurrent.futures.as_completed(jobs):
            try:
                job.result()
            except Exception as error:
                errors.append(f'{jobs[job].name}: {error}')
        if errors:
            raise RuntimeError('\n'.join(errors))
    collect()
    first.write_json(OUT / 'execution.json', dict(seconds=time.monotonic() - started,
        methods=16, configurations=48, external_evaluations=192, workers=workers))


def collect():
    registry = json.loads((OUT / 'preregistered.json').read_text())
    verify_sources(registry['input_sha256'])
    records = []
    for filename, expected in registry['methods'].items():
        path = HERE / 'methods' / filename
        if first.digest(path) != expected['sha256']:
            raise ValueError(f'Frozen method changed: {filename}')
        rows = json.loads((OUT / path.stem / 'results.json').read_text())
        if len(rows) != 3 or [row['config'] for row in rows] != expected['configs']:
            raise ValueError(f'Incomplete or altered configurations: {filename}')
        for row in rows:
            if row['code_sha256'] != expected['sha256'] or first.digest(HERE / row['submission']) != row['submission_sha256']:
                raise ValueError(f'Artifact differs from frozen code/CSV: {filename}')
        records.extend(rows)
    records.sort(key=lambda r: (-r['external_mean_spellF1'], -r['external_mean_macroF1'], r['method'], r['config']['name']))
    selection = dict(selected=records[0]['submission'], reason=registry['selection'], records=records,
        preregistered_sha256=first.digest(OUT / 'preregistered.json'),
        holdout_seen_for_selection=False, public_gt_seen_for_selection=False)
    first.write_json(OUT / 'selection.json', selection)
    pd.DataFrame([dict(method=r['method'], config=r['config']['name'],
        external_spellF1=r['external_mean_spellF1'], external_macroF1=r['external_mean_macroF1'],
        seconds=r['seconds']) for r in records]).to_csv(OUT / 'external_scoreboard.csv', index=False)
    print('FROZEN SELECTION', selection['selected'], flush=True)


def frozen_selection():
    registry = json.loads((OUT / 'preregistered.json').read_text())
    verify_sources(registry['input_sha256'])
    selection = json.loads((OUT / 'selection.json').read_text())
    if first.digest(OUT / 'preregistered.json') != selection['preregistered_sha256']:
        raise ValueError('Preregistration changed')
    for filename, expected in registry['methods'].items():
        if first.digest(HERE / 'methods' / filename) != expected['sha256']:
            raise ValueError(f'Method changed: {filename}')
    return selection


def holdout_and_reproduce():
    selection = frozen_selection()
    selected = selection['records'][0]
    module = module_for(HERE / 'methods' / f'{selected["method"]}.py')
    train = first.training([first.load('external_1'), first.load('external_2')])
    fitted = model_fit(module, train, selected['config'])
    prediction = model_predict(module, fitted, first.load('target'), selected['config'])
    temporary = OUT / 'reproduced.csv'
    regenerated = save_submission(temporary, prediction)
    temporary.unlink()
    if regenerated != selected['submission_sha256']:
        raise ValueError('Selected candidate regeneration differs byte-for-byte')
    with np.load(HERE / 'cache/holdout_3.npz', allow_pickle=False) as archive:
        holdout = {key: archive[key] for key in archive.files}
    baseline_selection = json.loads((FIRST / 'out/selection.json').read_text())
    baseline_record = baseline_selection['records'][0]
    baseline = module_for(FIRST / 'methods' / f'{baseline_record["method"]}.py')
    results = []
    for regime in ('X', 'X_occluded'):
        new_prediction = model_predict(module, fitted, holdout, selected['config'], regime)
        old_sampled = first.checked_predict(baseline, train, first.prediction_input(holdout, regime), baseline_record['config'])
        old_prediction = first.expand(old_sampled, holdout)
        for name, pred in [('wave2_selected', new_prediction), ('wave1_selected', old_prediction)]:
            row = first.metrics(holdout['y25'], pred)
            row.update(candidate=name, regime='clean' if regime == 'X' else 'occluded')
            results.append(row)
            np.save(OUT / f'holdout3_{name}_{regime}.npy', pred)
    report = dict(selection_sha256=first.digest(OUT / 'selection.json'), results=results,
        reproduced_sha256=regenerated, identical_target_rows=len(prediction),
        selection_changed=False, warning='Two frozen candidates only; no selection or tuning on Game3.')
    first.write_json(OUT / 'holdout_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def public_audit():
    import run_evaluation as official
    selection = frozen_selection()
    if not (OUT / 'holdout_report.json').exists():
        raise FileNotFoundError('Evaluate frozen holdout before public audit')
    gt = pm.label_codes(pm.read_label_csv(official.find_gt('public')))
    rows = []
    for record in selection['records']:
        path = HERE / record['submission']
        if first.digest(path) != record['submission_sha256']:
            raise ValueError('Frozen prediction changed')
        result, reason = official.score_submission(str(path), gt, pm.TAUS, 0)
        if reason:
            raise ValueError(reason)
        rows.append(dict(submission=record['submission'], selected=record['submission'] == selection['selected'],
            spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'], possessions=result['n_pred_spells']))
    first.write_json(OUT / 'public_audit.json', dict(selection_sha256=first.digest(OUT / 'selection.json'),
        records=rows, warning='Post-freeze diagnostic only; no parameter or selection updates.'))
    pd.DataFrame(rows).to_csv(OUT / 'public_scoreboard.csv', index=False)
    print('Public audit:', len(rows), 'valid submissions;', next(r for r in rows if r['selected']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['all', 'method', 'collect', 'holdout', 'public'])
    parser.add_argument('--path')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'all':
        execute_all(args.workers)
    elif args.action == 'method':
        run_method(args.path)
    elif args.action == 'collect':
        collect()
    elif args.action == 'holdout':
        holdout_and_reproduce()
    else:
        public_audit()


if __name__ == '__main__':
    main()
