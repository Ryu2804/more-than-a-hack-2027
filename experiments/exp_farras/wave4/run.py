import argparse
import gc
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import campaign as c
import decoder
import fusion
import models

BASE = ('plain_tcn', 'pp_tcn', 'control_tcn')
FAMILIES = (*BASE, 'pp_gru25', 'pp_gru50', 'pp_gru75', 'pp_tree25', 'prior_combined')
os.environ.setdefault('FARRAS_TRAIN_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
prior_decoder = c.import_path('wave4_previous_decode', c.PREVIOUS / 'decode.py')


def config(module, name):
    return dict(next(value for value in module.CONFIGS if value['name'] == name))


def read_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_prediction(path, value):
    probability = np.asarray(value['probability'])
    if probability.ndim != 2 or probability.shape[1] != 3 or not np.isfinite(probability).all():
        raise ValueError('Prediction must be finite Nx3')
    if (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=1e-5):
        raise ValueError('Prediction must be normalized nonnegative probabilities')
    for field in ('p_dead', 'p_team', 'boundary'):
        if field in value:
            values = np.asarray(value[field])
            if values.shape != (len(probability),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Invalid prediction field: ' + field)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **value)


def prediction_path(family, tag, name, regime):
    return c.CACHE / 'predictions' / family / tag / f'{name}_{regime}.npz'


def checkpoint(family):
    return c.OUT / 'models' / (family + '.pt')


def source_manifest():
    paths = list(c.HERE.glob('*.py')) + list((c.HERE / 'reference').glob('*.py')) + [c.HERE / 'PLAN.md']
    paths += list((c.CACHE / 'features').glob('*.npz')) + list((c.CACHE / 'prior').glob('*.npz'))
    paths += [c.CACHE / name for name in ('feature_audit.json', 'skillcorner_audit.json', 'prior_dev_audit.json')]
    paths += list((c.CACHE / 'ryu_weights').glob('*'))
    paths += [c.PREVIOUS / 'out/selection.json', c.PREVIOUS / 'out/models/hybrid.pt',
              c.PREVIOUS / 'out/models/anchor.joblib', c.PREVIOUS / 'models.py',
              c.PREVIOUS / 'tokens.py', c.PREVIOUS / 'decode.py', c.PREVIOUS / 'ensemble.py']
    paths += [c.pack_path(name, regime) for name in (*c.IDSSE_DEV, *c.IDSSE_HOLD) for regime in ('r0', 'r1', 'r2')]
    paths += [c.pack_path(name, 'clean') for name in (*c.SC_DEV, *c.SC_HOLD)]
    paths += [c.pack_path(f'metrica{number}', regime) for number in (1, 2, 3) for regime in ('clean', 'r0')]
    paths += [c.pack_path('target', 'clean')]
    paths += [c.REPO / 'games.py', c.REPO / 'evaluation/possession_metrics.py',
              c.REPO / 'evaluation/run_evaluation.py', c.PREVIOUS / 'cache/local_audit.json']
    return {path.relative_to(c.REPO).as_posix(): c.digest(path) for path in sorted(set(paths))}


def external_sources():
    from pathlib import Path
    audit = json.loads((c.PREVIOUS / 'cache/local_audit.json').read_text())
    root = Path(os.environ.get('FARRAS_SOURCE_EXPERIMENTS', audit['source_root'])).resolve()
    result = {}
    for key, expected in audit['source_sha256'].items():
        path = root / key.split(':', 1)[1]
        observed = c.digest(path)
        if observed != expected:
            raise ValueError('Raw input differs from prior prepared pack provenance: ' + path.name)
        result[path.relative_to(c.REPO.parent).as_posix()] = observed
    return result


def verify_registry():
    value = json.loads((c.OUT / 'preregistered.json').read_text())
    if value['runtime']['device'] != os.environ['FARRAS_TRAIN_DEVICE']:
        raise ValueError('Training/inference backend differs from preregistration')
    for path, expected in value['sha256'].items():
        if c.digest(c.REPO / path) != expected:
            raise ValueError('Frozen source/input changed: ' + path)
    for path, expected in value['external_sha256'].items():
        if c.digest(c.REPO.parent / path) != expected:
            raise ValueError('Frozen external raw input changed: ' + path)
    return value


def register():
    if (c.OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen; use a new campaign')
    if (c.OUT / 'preregistered.json').exists():
        return verify_registry()
    if {item['name'] for item in models.CONFIGS} != set(BASE) or len(decoder.CONFIGS) != 8:
        raise ValueError('Model/decoder grid differs from plan')
    value = dict(sha256=source_manifest(), external_sha256=external_sources(),
        development=list(c.DEV), holdout=list(c.HOLD),
        fold_of=c.FOLD, models=models.CONFIGS, decoders=decoder.CONFIGS, fusion=fusion.CONFIGS,
        families=list(FAMILIES), configurations=64,
        selection='Equal provider means: IDSSE,SkillCorner,Metrica; clean/r0 averaged within Metrica match. Ties macroF1,family,decoder.',
        runtime=dict(device=os.environ['FARRAS_TRAIN_DEVICE'], torch=torch.__version__, numpy=np.__version__, cpu_threads=1),
        public_used_for_selection=False, holdout_used_for_selection=False,
        exposure='Prior IDSSE/Metrica3 results known; SC holdout new to Farras but used in Ryu research. Shipped Ryu cannot be heldout on IDSSE/SC.')
    c.write_json(c.OUT / 'preregistered.json', value)
    return value


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_job(family, names, tests, tag, final=False):
    paths = [prediction_path(family, tag, name, regime) for name, regime in tests]
    if all(path.exists() for path in paths) and (not final or checkpoint(family).exists()):
        print('REUSE', family, tag, flush=True)
        return
    start = time.monotonic()
    datasets = c.training(names)
    with threadpool_limits(limits=1):
        fitted = models.fit(datasets, config(models, family))
    del datasets
    gc.collect()
    for (name, regime), path in zip(tests, paths):
        data = c.load(name, regime, with_features=True)
        with threadpool_limits(limits=1):
            output = models.predict(fitted, c.observations(data), config(models, family))
        save_prediction(path, output)
        del data, output
    if final:
        checkpoint(family).parent.mkdir(parents=True, exist_ok=True)
        torch.save(fitted, checkpoint(family))
    print('FIT', family, tag, 'seconds', round(time.monotonic() - start, 1), flush=True)
    del fitted
    release()


def prior_parts(name, regime):
    stored = read_prediction(c.CACHE / 'prior' / f'{name}_{regime}.npz')
    return dict(probability=stored['probability'], boundary=stored['boundary']), dict(
        probability=stored['hybrid_probability'], boundary=stored['hybrid_boundary']), dict(
        probability=stored['anchor_probability'], boundary=stored['anchor_boundary'])


def probability(family, name, regime, get_base):
    if family in BASE:
        return get_base(family)
    prior, neural, tree = prior_parts(name, regime)
    if family == 'prior_combined':
        return prior
    mixture = family[len('pp_'):]
    other = tree if mixture == 'tree25' else neural
    return fusion.combine(get_base('pp_tcn'), other, config(fusion, mixture))


def collect():
    verify_registry()
    records = []
    stats_by_tag = {f'fold{fold}': c.training_stats([name for name in c.DEV if c.FOLD[name] != fold])
                    for fold in range(6)}
    stats_by_tag['final'] = c.training_stats(c.DEV)
    for family in FAMILIES:
        by_decoder = {item['name']: [] for item in decoder.CONFIGS}
        for name, regime, tag in c.dev_cases():
            data = c.load(name, regime)
            stats = stats_by_tag[tag]
            output = probability(family, name, regime,
                lambda base: read_prediction(prediction_path(base, tag, name, regime)))
            decoder_input = c.decoder_observations(data)
            for settings in decoder.CONFIGS:
                result = decoder.decode(output, decoder_input, settings, stats)
                row = c.metrics(result['codes'], data)
                row.update(match=name, source=c.source(name), regime=regime,
                    calibrated_lambda=float(result['lambda']), target_count=float(result['target_count']),
                    dead_shift=float(result['dead_shift']))
                by_decoder[settings['name']].append(row)
        for settings in decoder.CONFIGS:
            rows = by_decoder[settings['name']]
            means = []
            for provider in ('idsse', 'skillcorner', 'metrica'):
                matches = sorted({row['match'] for row in rows if row['source'] == provider})
                means.append(dict(source=provider, spellF1=float(np.mean([
                    np.mean([row['spell_f1_mean'] for row in rows if row['match'] == match]) for match in matches])),
                    macroF1=float(np.mean([np.mean([row['macro_f1'] for row in rows if row['match'] == match]) for match in matches]))))
            records.append(dict(family=family, decoder=settings, external_folds=rows, provider_means=means,
                external_mean_spellF1=float(np.mean([row['spellF1'] for row in means])),
                external_mean_macroF1=float(np.mean([row['macroF1'] for row in means])),
                max_abs_count_error=float(max(abs(row['count_error_ratio']) for row in rows))))
    records.sort(key=lambda row: (-row['external_mean_spellF1'], -row['external_mean_macroF1'], row['family'], row['decoder']['name']))
    selection = dict(selected=dict(family=records[0]['family'], decoder=records[0]['decoder']), records=records,
        preregistered_sha256=c.digest(c.OUT / 'preregistered.json'),
        checkpoint_sha256={family: c.digest(checkpoint(family)) for family in BASE},
        public_used_for_selection=False, holdout_used_for_selection=False)
    c.write_json(c.OUT / 'selection.json', selection)
    import pandas as pd
    pd.DataFrame([dict(family=row['family'], decoder=row['decoder']['name'],
        spellF1=row['external_mean_spellF1'], macroF1=row['external_mean_macroF1'],
        max_abs_count_error=row['max_abs_count_error']) for row in records]).to_csv(c.OUT / 'external_scoreboard.csv', index=False)
    print('FROZEN', selection['selected'], 'spellF1', records[0]['external_mean_spellF1'], flush=True)
    return selection


def development():
    started = time.monotonic()
    register()
    for fold in range(6):
        names = [name for name in c.DEV if c.FOLD[name] != fold]
        tests = [(name, 'r0' if name.startswith('J03') else 'clean') for name in c.DEV if c.FOLD[name] == fold]
        for family in BASE:
            fit_job(family, names, tests, f'fold{fold}')
    tests = [(f'metrica{number}', regime) for number in (1, 2) for regime in ('clean', 'r0')]
    for family in BASE:
        fit_job(family, c.DEV, tests, 'final', final=True)
    collect()
    c.write_json(c.OUT / 'execution.json', dict(seconds=time.monotonic() - started,
        model_ensemble_fits=21, seed_fits=63, configurations=64, development_evaluations=1472,
        development_matches=21, evaluation_scenarios=23, device=os.environ['FARRAS_TRAIN_DEVICE']))


def frozen_selection():
    verify_registry()
    selection = json.loads((c.OUT / 'selection.json').read_text())
    if c.digest(c.OUT / 'preregistered.json') != selection['preregistered_sha256']:
        raise ValueError('Preregistration changed')
    for family, expected in selection['checkpoint_sha256'].items():
        if c.digest(checkpoint(family)) != expected:
            raise ValueError('Checkpoint changed: ' + family)
    return selection


def infer_base(family, name, regime, memo):
    key = family, name, regime
    if key not in memo:
        fitted = torch.load(checkpoint(family), weights_only=False)
        data = c.load(name, regime, with_features=True)
        with threadpool_limits(limits=1):
            memo[key] = models.predict(fitted, c.observations(data), config(models, family))
        del fitted, data
        release()
    return memo[key]


def candidates(selection):
    return {'selected': selection['selected'],
        'new_pp_tcn': dict(family='pp_tcn', decoder=config(decoder, 'ryu_rate')),
        'previous_wave3': dict(family='prior_combined', decoder={'name': 'previous_centered2'})}


def candidate_codes(spec, output, data, stats):
    if spec['decoder']['name'] == 'previous_centered2':
        prior_settings = next(value for value in prior_decoder.CONFIGS if value['name'] == 'centered2')
        return prior_decoder.decode(output['probability'], output['boundary'], data['blocks'], prior_settings)
    return decoder.decode(output, c.decoder_observations(data), spec['decoder'], stats)['codes']


def submission(path, codes, data):
    import pandas as pd
    full = c.expand(codes, data)
    table = pd.DataFrame(dict(frame=np.arange(len(full)), label=np.asarray(c.pm.LABELS)[full]))
    reason = c.pm.validate_submission(table, 199319)
    if reason:
        raise ValueError(reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator='\n')
    return c.digest(path)


def independent_refit(family, data):
    if family == 'prior_combined':
        prior, _, _ = prior_parts('target', 'clean')
        return prior
    base = family if family in BASE else 'pp_tcn'
    datasets = c.training(c.DEV)
    with threadpool_limits(limits=1):
        fitted = models.fit(datasets, config(models, base))
    del datasets
    with threadpool_limits(limits=1):
        output = models.predict(fitted, c.observations(data), config(models, base))
    del fitted
    release()
    if family not in BASE:
        _, neural, tree = prior_parts('target', 'clean')
        mixture = family[len('pp_'):]
        output = fusion.combine(output, tree if mixture == 'tree25' else neural, config(fusion, mixture))
    return output


def holdout():
    selection = frozen_selection()
    if not (c.CACHE / 'prior_holdout_audit.json').exists():
        raise FileNotFoundError('Run prepare_prior.py holdout after freeze')
    stats = c.training_stats(c.DEV)
    specs = candidates(selection)
    memo, rows, artifacts = {}, [], {}
    for name, regime in c.holdout_cases():
        data = c.load(name, regime)
        for role, spec in specs.items():
            output = probability(spec['family'], name, regime,
                lambda base: infer_base(base, name, regime, memo))
            codes = candidate_codes(spec, output, data, stats)
            row = c.metrics(codes, data)
            row.update(candidate=role, family=spec['family'], decoder=spec['decoder']['name'], match=name, regime=regime)
            rows.append(row)
            print('HOLDOUT', role, name, regime, row['spell_f1_mean'], flush=True)
    reference = c.import_path('wave4_shipped_reference', c.HERE / 'reference.py')
    reference_audits = {}
    for regime in ('clean', 'r0'):
        data = c.load('metrica3', regime)
        output, audit = reference.predict('metrica3_' + regime, device=os.environ['FARRAS_TRAIN_DEVICE'])
        codes = decoder.decode(output, c.decoder_observations(data), config(decoder, 'ryu_rate'), audit['training_stats'])['codes']
        row = c.metrics(codes, data)
        row.update(candidate='shipped_ryu', family='shipped_no_metrica', decoder='ryu_rate', match='metrica3', regime=regime)
        rows.append(row)
        reference_audits['metrica3_' + regime] = audit
    data = c.load('target', 'clean', with_features=True)
    if np.any(data['y'] >= 0) or np.any(data['y25'] >= 0):
        raise ValueError('Target must not contain labels')
    for role, spec in specs.items():
        output = probability(spec['family'], 'target', 'clean', lambda base: infer_base(base, 'target', 'clean', memo))
        path = c.OUT / 'candidates' / (role + '.csv')
        artifacts[role] = dict(path=path.relative_to(c.HERE).as_posix(),
            sha256=submission(path, candidate_codes(spec, output, data, stats), data), **spec)
    output, audit = reference.predict('target', device=os.environ['FARRAS_TRAIN_DEVICE'])
    codes = decoder.decode(output, c.decoder_observations(data), config(decoder, 'ryu_rate'), audit['training_stats'])['codes']
    path = c.OUT / 'candidates/shipped_ryu.csv'
    artifacts['shipped_ryu'] = dict(path=path.relative_to(c.HERE).as_posix(), sha256=submission(path, codes, data),
        family='shipped_no_metrica', decoder=config(decoder, 'ryu_rate'))
    reference_audits['target'] = audit
    start = time.monotonic()
    output = independent_refit(selection['selected']['family'], data)
    path = c.OUT / 'independent_refit.csv'
    actual = submission(path, candidate_codes(selection['selected'], output, data, stats), data)
    path.unlink()
    if actual != artifacts['selected']['sha256']:
        raise ValueError('Independent selected-component refit differs byte-for-byte')
    previous_hash = json.loads((c.PREVIOUS / 'out/holdout_report.json').read_text())['reproduced_sha256']
    if artifacts['previous_wave3']['sha256'] != previous_hash:
        raise ValueError('Frozen previous wave3 candidate was not reproduced')
    report = dict(selection_sha256=c.digest(c.OUT / 'selection.json'), results=rows, submissions=artifacts,
        reference_audits=reference_audits, reproduced_sha256=actual, identical_target_rows=199319,
        independent_refit_seconds=time.monotonic() - start, selection_changed=False,
        warning='Shipped Ryu never scored as heldout on its IDSSE/SkillCorner training matches; prior wave3 checkpoint refit proof inherited.')
    c.write_json(c.OUT / 'holdout_report.json', report)
    print('REFIT IDENTICAL', actual, flush=True)


def public_audit():
    frozen_selection()
    report = json.loads((c.OUT / 'holdout_report.json').read_text())
    if report['selection_sha256'] != c.digest(c.OUT / 'selection.json'):
        raise ValueError('Different frozen selection in holdout report')
    sys.path.insert(0, str(c.REPO / 'evaluation'))
    import run_evaluation as official
    gt = c.pm.label_codes(c.pm.read_label_csv(official.find_gt('public')))
    rows = []
    for role, artifact in report['submissions'].items():
        path = c.HERE / artifact['path']
        if c.digest(path) != artifact['sha256']:
            raise ValueError('Candidate changed before public audit')
        result, reason = official.score_submission(str(path), gt, c.pm.TAUS, 0)
        if reason:
            raise ValueError(reason)
        rows.append(dict(candidate=role, family=artifact['family'], decoder=artifact['decoder']['name'],
            spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'], possessions=result['n_pred_spells']))
    c.write_json(c.OUT / 'public_audit.json', dict(selection_sha256=report['selection_sha256'], records=rows,
        warning='Four fixed candidates only after freeze; no64-way public sweep. Shipped reference uses cached inputs/adapted decoder, not certified byte reproduction.'))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dev', 'collect', 'holdout', 'public'])
    {'dev': development, 'collect': collect, 'holdout': holdout, 'public': public_audit}[parser.parse_args().action]()
