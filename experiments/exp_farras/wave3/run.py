"""Fixed matched-data selection; locked holdout and public audit follow the freeze."""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('FARRAS_TRAIN_DEVICE', 'cpu')
sys.dont_write_bytecode = True

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / 'wave1'
SECOND = HERE.parent / 'wave2'
REPO = HERE.parents[2]
CACHE = HERE / 'cache'
OUT = HERE / 'out'
DATA = CACHE / 'data'
for directory in (FIRST, SECOND, HERE):
    sys.path.insert(0, str(directory))


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


first = import_path('wave3_first_campaign', FIRST / 'run.py')
support = import_path('wave3_prior_support', SECOND / 'support.py')
import decode
import ensemble
import models
import trees

DEV = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY')
HOLDOUT = ('J03WQQ', 'J03WR9')
NEURAL = ('setseq', 'hybrid', 'feature_gru')
TREE = ('control', 'control_symmetric', 'all_features')
FUSION = ('tree25', 'tree50', 'tree75', 'visibility')
FAMILIES = (*NEURAL, *TREE, 'anchor', *('hybrid_' + name for name in FUSION))


def write_json(path, value):
    first.write_json(path, value)


def load(name, regime='r0'):
    with np.load(DATA / f'{name}_{regime}.npz', allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def observations(data):
    return {key: value for key, value in data.items() if key not in ('y', 'y25')}


def train_data(names):
    return [load(name, f'r{round_index}') for name in names for round_index in range(3)]


def settings(module, name):
    return dict(next(config for config in module.CONFIGS if config['name'] == name))


def manifest():
    paths = list(HERE.glob('*.py')) + list((HERE / 'reference').glob('*.py')) + [HERE / 'PLAN.md']
    paths += list(DATA.glob('*.npz'))
    paths += [CACHE / 'idsse_audit.json', CACHE / 'local_audit.json']
    paths += [FIRST / 'run.py', FIRST / 'data.py', SECOND / 'support.py',
              SECOND / 'methods/w14_feature_ablation.py', SECOND / 'methods/w08_symmetric_boost.py']
    paths += [FIRST / 'cache' / name for name in ('external_1.npz', 'external_2.npz', 'target.npz', 'manifest.json')]
    return {path.relative_to(REPO).as_posix(): first.digest(path) for path in sorted(set(paths))}


def verify_registry():
    registry = json.loads((OUT / 'preregistered.json').read_text())
    for relative, expected in registry['sha256'].items():
        if first.digest(REPO / relative) != expected:
            raise ValueError('Frozen input or code changed: ' + relative)
    return registry


def register():
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection is frozen; start a new campaign instead of overwriting')
    if (OUT / 'preregistered.json').exists():
        return verify_registry()
    for name in (*DEV, *HOLDOUT):
        for round_index in range(3):
            if not (DATA / f'{name}_r{round_index}.npz').is_file():
                raise FileNotFoundError(f'Missing prepared IDSSE pack: {name}, round {round_index}')
    for name in ('metrica1', 'metrica2', 'metrica3'):
        for regime in ('clean', 'r0'):
            if not (DATA / f'{name}_{regime}.npz').is_file():
                raise FileNotFoundError(f'Missing prepared Metrica pack: {name}, {regime}')
    if not (DATA / 'target_clean.npz').is_file():
        raise FileNotFoundError('Missing prepared target observations')
    if {c['name'] for c in models.CONFIGS} != set(NEURAL) or {c['name'] for c in trees.CONFIGS} != set(TREE):
        raise ValueError('Model configuration registry differs from the plan')
    if len(decode.CONFIGS) != 6 or {c['name'] for c in ensemble.CONFIGS} != set(FUSION):
        raise ValueError('Decoder/fusion registry differs from the plan')
    value = dict(sha256=manifest(), development=list(DEV), locked_holdout=list(HOLDOUT),
        models=models.CONFIGS, trees=trees.CONFIGS, decoders=decode.CONFIGS, fusion=ensemble.CONFIGS,
        probability_families=list(FAMILIES), configurations=66,
        runtime=dict(train_device=os.environ['FARRAS_TRAIN_DEVICE'], cpu_threads=1,
            torch=models.torch.__version__, numpy=np.__version__),
        selection='Mean of five IDSSE and two Metrica match spellF1; average clean/r0 inside each Metrica match. Ties macroF1,family,decoder.',
        public_gt_used_for_selection=False, holdout_used_for_selection=False,
        warning='IDSSE holdout was used in Jeremy research; Metrica3 and historical public diagnostics were previously exposed.')
    write_json(OUT / 'preregistered.json', value)
    return value


def prediction_path(family, tag, name, regime):
    return CACHE / 'predictions' / family / tag / f'{name}_{regime}.npz'


def save_prediction(path, result):
    p = np.asarray(result['probability'], dtype=np.float32)
    b = np.asarray(result['boundary'], dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or b.shape != (len(p),):
        raise ValueError('Invalid probability/boundary shape')
    if not np.isfinite(p).all() or not np.isfinite(b).all() or (p < 0).any() or ((b < 0) | (b > 1)).any():
        raise ValueError('Invalid probability/boundary values')
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('State probabilities do not sum to one')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probability=p, boundary=b)


def read_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_path(family):
    return OUT / 'models' / (family + ('.pt' if family in NEURAL else '.joblib'))


def fit_model(family, datasets):
    module = models if family in NEURAL else trees
    with threadpool_limits(limits=1):
        return module.fit(datasets, settings(module, family))


def predict_model(family, fitted, data):
    module = models if family in NEURAL else trees
    with threadpool_limits(limits=1):
        return module.predict(fitted, observations(data), settings(module, family))


def fit_job(family, names, tests, tag, final=False):
    paths = [prediction_path(family, tag, name, regime) for name, regime in tests]
    if all(path.exists() for path in paths) and (not final or checkpoint_path(family).exists()):
        print('REUSE', family, tag, flush=True)
        return
    start = time.monotonic()
    datasets = train_data(names)
    fitted = fit_model(family, datasets)
    for (name, regime), path in zip(tests, paths):
        save_prediction(path, predict_model(family, fitted, load(name, regime)))
    if final:
        checkpoint_path(family).parent.mkdir(parents=True, exist_ok=True)
        if family in NEURAL:
            import torch
            torch.save(fitted, checkpoint_path(family))
        else:
            joblib.dump(fitted, checkpoint_path(family))
    elapsed = time.monotonic() - start
    print(f'FIT {family} {tag} seconds={elapsed:.1f}', flush=True)
    del fitted, datasets
    release_gpu()


def anchor_fit(exclude=None, symmetric=False):
    packs = [first.load(f'external_{number}') for number in (1, 2) if number != exclude]
    training = first.training(packs)
    if symmetric:
        training = support.symmetrize(training)
    return support.fit_emitter(training, kind='hist', columns=None if symmetric else [0, 1, 2, 3, 7, 14, 15])


def anchor_predict(fitted, data):
    p = support.probabilities(fitted, observations(data))
    return dict(probability=p, boundary=np.full(len(p), .5, dtype=np.float32))


def prepare_anchor():
    fitted = anchor_fit()
    for name in DEV:
        save_prediction(prediction_path('anchor', 'loo_' + name, name, 'r0'), anchor_predict(fitted, load(name)))
    checkpoint_path('anchor').parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, checkpoint_path('anchor'))
    del fitted
    for number in (1, 2):
        fitted = anchor_fit(exclude=number)
        for regime in ('clean', 'r0'):
            name = f'metrica{number}'
            save_prediction(prediction_path('anchor', 'final', name, regime), anchor_predict(fitted, load(name, regime)))
        del fitted


def cases():
    result = [(name, 'r0', 'loo_' + name) for name in DEV]
    result += [(f'metrica{number}', regime, 'final') for number in (1, 2) for regime in ('clean', 'r0')]
    return result


def family_prediction(family, data, get_base):
    if family.startswith('hybrid_'):
        config = settings(ensemble, family[len('hybrid_'):])
        return ensemble.combine(get_base('hybrid'), get_base('anchor'), observations(data), config)
    return get_base(family)


def metrics(data, sampled):
    full = first.expand(sampled, data)
    row = first.metrics(data['y25'], full)
    row['count_error_ratio'] = float((row['n_pred_spells'] - row['n_gt_spells']) / max(1, row['n_gt_spells']))
    return row


def collect():
    verify_registry()
    prepared = {(name, regime): load(name, regime) for name, regime, _ in cases()}
    records = []
    for family in FAMILIES:
        per_config = {config['name']: [] for config in decode.CONFIGS}
        for name, regime, tag in cases():
            data = prepared[(name, regime)]
            result = family_prediction(family, data,
                lambda base: read_prediction(prediction_path(base, tag, name, regime)))
            for config in decode.CONFIGS:
                codes = decode.decode(result['probability'], result['boundary'], data['blocks'], config)
                row = metrics(data, codes)
                row.update(match=name, regime=regime)
                per_config[config['name']].append(row)
        for config in decode.CONFIGS:
            rows = per_config[config['name']]
            match_rows = []
            for name in (*DEV, 'metrica1', 'metrica2'):
                group = [row for row in rows if row['match'] == name]
                match_rows.append(dict(match=name,
                    spellF1=float(np.mean([row['spell_f1_mean'] for row in group])),
                    macroF1=float(np.mean([row['macro_f1'] for row in group]))))
            records.append(dict(family=family, decoder=config, external_folds=rows, match_means=match_rows,
                external_mean_spellF1=float(np.mean([row['spellF1'] for row in match_rows])),
                external_mean_macroF1=float(np.mean([row['macroF1'] for row in match_rows])),
                max_abs_count_error=float(max(abs(row['count_error_ratio']) for row in rows))))
    records.sort(key=lambda row: (-row['external_mean_spellF1'], -row['external_mean_macroF1'], row['family'], row['decoder']['name']))
    checkpoints = {family: first.digest(checkpoint_path(family)) for family in (*NEURAL, *TREE, 'anchor')}
    selection = dict(selected=dict(family=records[0]['family'], decoder=records[0]['decoder']),
        records=records, preregistered_sha256=first.digest(OUT / 'preregistered.json'),
        checkpoint_sha256=checkpoints, public_seen_for_selection=False, holdout_seen_for_selection=False)
    write_json(OUT / 'selection.json', selection)
    import pandas as pd
    pd.DataFrame([dict(family=row['family'], decoder=row['decoder']['name'],
        spellF1=row['external_mean_spellF1'], macroF1=row['external_mean_macroF1'],
        max_abs_count_error=row['max_abs_count_error']) for row in records]).to_csv(OUT / 'external_scoreboard.csv', index=False)
    print('FROZEN', selection['selected'], 'spellF1', records[0]['external_mean_spellF1'], flush=True)
    return selection


def development():
    started = time.monotonic()
    register()
    for held in DEV:
        names = [name for name in DEV if name != held]
        for family in (*NEURAL, *TREE):
            fit_job(family, names, [(held, 'r0')], 'loo_' + held)
    tests = [(f'metrica{number}', regime) for number in (1, 2) for regime in ('clean', 'r0')]
    for family in (*NEURAL, *TREE):
        fit_job(family, DEV, tests, 'final', final=True)
    prepare_anchor()
    collect()
    write_json(OUT / 'execution.json', dict(seconds=time.monotonic() - started,
        neural_fits=18, idsse_tree_fits=18, configurations=66, external_evaluations=594,
        development_matches=7, evaluation_scenarios=9, parallel_fits=1,
        train_device=os.environ['FARRAS_TRAIN_DEVICE']))


def frozen_selection():
    verify_registry()
    selection = json.loads((OUT / 'selection.json').read_text())
    if first.digest(OUT / 'preregistered.json') != selection['preregistered_sha256']:
        raise ValueError('Preregistration was changed')
    for family, expected in selection['checkpoint_sha256'].items():
        if first.digest(checkpoint_path(family)) != expected:
            raise ValueError('Fitted checkpoint was changed: ' + family)
    return selection


def load_fitted(family):
    if family in NEURAL:
        import torch
        return torch.load(checkpoint_path(family), weights_only=False)
    return joblib.load(checkpoint_path(family))


def infer_probability(family, name, regime, memo):
    key = (family, name, regime)
    if key in memo:
        return memo[key]
    data = load(name, regime)
    if family.startswith('hybrid_'):
        result = family_prediction(family, data,
            lambda base: infer_probability(base, name, regime, memo))
    elif family == 'prior_symmetric':
        fitted = anchor_fit(symmetric=True)
        result = anchor_predict(fitted, data)
        del fitted
    else:
        fitted = load_fitted(family)
        result = anchor_predict(fitted, data) if family == 'anchor' else predict_model(family, fitted, data)
        del fitted
        release_gpu()
    memo[key] = result
    return result


def candidates(selection):
    return {'selected': selection['selected'],
        'adapted_jeremy': dict(family='setseq', decoder=settings(decode, 'cost05')),
        'prior_control': dict(family='anchor', decoder=settings(decode, 'cost6')),
        'prior_symmetric': dict(family='prior_symmetric', decoder=settings(decode, 'cost6'))}


def decoded(spec, prediction, data):
    return decode.decode(prediction['probability'], prediction['boundary'], data['blocks'], spec['decoder'])


def write_submission(path, sampled, data):
    import pandas as pd
    full = first.expand(sampled, data)
    table = pd.DataFrame(dict(frame=np.arange(len(full)), label=np.asarray(first.pm.LABELS)[full]))
    reason = first.pm.validate_submission(table, 199319)
    if reason:
        raise ValueError(reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator='\n')
    return first.digest(path)


def independent_refit(family, data):
    if family.startswith('hybrid_'):
        neural = independent_refit('hybrid', data)
        anchor = independent_refit('anchor', data)
        return ensemble.combine(neural, anchor, observations(data), settings(ensemble, family[len('hybrid_'):]))
    if family == 'anchor':
        fitted = anchor_fit()
        result = anchor_predict(fitted, data)
    else:
        datasets = train_data(DEV)
        fitted = fit_model(family, datasets)
        result = predict_model(family, fitted, data)
        del datasets
    del fitted
    release_gpu()
    return result


def holdout():
    selection = frozen_selection()
    memo, rows, submissions = {}, [], {}
    specs = candidates(selection)
    for name, regime in [(name, 'r0') for name in HOLDOUT] + [('metrica3', 'clean'), ('metrica3', 'r0')]:
        data = load(name, regime)
        for role, spec in specs.items():
            prediction = infer_probability(spec['family'], name, regime, memo)
            row = metrics(data, decoded(spec, prediction, data))
            row.update(candidate=role, family=spec['family'], decoder=spec['decoder']['name'], match=name, regime=regime)
            rows.append(row)
            print('HOLDOUT', role, name, regime, row['spell_f1_mean'], flush=True)
    data = load('target', 'clean')
    if np.any(data['y'] >= 0) or np.any(data['y25'] >= 0):
        raise ValueError('Target pack must not contain labels')
    for role, spec in specs.items():
        prediction = infer_probability(spec['family'], 'target', 'clean', memo)
        sampled = decoded(spec, prediction, data)
        path = OUT / 'candidates' / (role + '.csv')
        submissions[role] = dict(path=path.relative_to(HERE).as_posix(), sha256=write_submission(path, sampled, data),
            family=spec['family'], decoder=spec['decoder'])
    started = time.monotonic()
    reproduced = independent_refit(selection['selected']['family'], data)
    sampled = decoded(selection['selected'], reproduced, data)
    temporary = OUT / 'independent_refit.csv'
    regenerated = write_submission(temporary, sampled, data)
    temporary.unlink()
    if regenerated != submissions['selected']['sha256']:
        raise ValueError('Independent selected-model refit is not byte-identical')
    report = dict(selection_sha256=first.digest(OUT / 'selection.json'), results=rows,
        submissions=submissions, reproduced_sha256=regenerated, identical_target_rows=199319,
        independent_refit_seconds=time.monotonic() - started, selection_changed=False,
        warning='Only four prespecified candidates; no holdout tuning. Metrica3 was previously exposed.')
    write_json(OUT / 'holdout_report.json', report)
    print('REFIT IDENTICAL', regenerated, flush=True)


def public_audit():
    selection = frozen_selection()
    report = json.loads((OUT / 'holdout_report.json').read_text())
    if report['selection_sha256'] != first.digest(OUT / 'selection.json'):
        raise ValueError('Holdout evaluated a different frozen selection')
    import run_evaluation as official
    gt = first.pm.label_codes(first.pm.read_label_csv(official.find_gt('public')))
    rows = []
    for role, artifact in report['submissions'].items():
        path = HERE / artifact['path']
        if first.digest(path) != artifact['sha256']:
            raise ValueError('Candidate CSV changed after holdout')
        result, reason = official.score_submission(str(path), gt, first.pm.TAUS, 0)
        if reason:
            raise ValueError(reason)
        rows.append(dict(candidate=role, family=artifact['family'], decoder=artifact['decoder']['name'],
            spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'], possessions=result['n_pred_spells']))
    write_json(OUT / 'public_audit.json', dict(selection_sha256=report['selection_sha256'], records=rows,
        warning='Post-freeze diagnostic for four fixed candidates only; no ranking of66 by public scores.'))
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dev', 'collect', 'holdout', 'public'])
    args = parser.parse_args()
    {'dev': development, 'collect': collect, 'holdout': holdout, 'public': public_audit}[args.action]()


if __name__ == '__main__':
    main()
