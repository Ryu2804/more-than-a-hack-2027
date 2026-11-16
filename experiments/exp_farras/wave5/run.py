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

import support as c
import training_models as models
import segment_decoder as decoder
import adaptive_fusion as fusion
import diagnostics

old_runner = c.import_path('wave5_frozen_wave4_runner', c.PREVIOUS / 'run.py')
os.environ.setdefault('FARRAS_TRAIN_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
BASE = ('boundary_tcn', 'short_tcn', 'masked_tcn')
CONTROL = dict(family='old_gru25', decoder='wave4')


def settings(module, name):
    return dict(next(row for row in module.CONFIGS if row['name'] == name))


def read_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if 'has_boundary' in result:
        flag = result['has_boundary']
        if flag.shape != () or flag.dtype.kind != 'b':
            raise ValueError('Stored boundary provenance must be one boolean')
        result['has_boundary'] = bool(flag)
    return result


def save_prediction(path, output):
    old_runner.save_prediction(path, output)


def prediction_path(family, tag, name, regime):
    return c.CACHE / 'predictions' / family / tag / f'{name}_{regime}.npz'


def checkpoint(family):
    return c.OUT / 'models' / (family + '.pt')


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def new_dependencies(family):
    if family in ('new_ensemble', 'new_ensemble_gru25', 'quality_new'):
        return BASE
    if family in BASE:
        return (family,)
    if family in ('boundary_gru25', 'short_gru25', 'masked_gru25'):
        return (family.removesuffix('_gru25') + '_tcn',)
    return ()


def inherited_manifest():
    old_runner.frozen_selection()
    old = json.loads((c.PREVIOUS / 'out/preregistered.json').read_text())
    hashes = dict(old['sha256'])
    paths = list(c.HERE.glob('*.py')) + [c.HERE / 'PLAN.md', c.PREVIOUS / 'out/selection.json',
        c.PREVIOUS / 'out/holdout_report.json', c.PREVIOUS / 'out/models/pp_tcn.pt']
    paths += list((c.base.CACHE / 'prior').glob('*.npz'))
    paths += [c.base.CACHE / 'predictions/pp_tcn' / tag / f'{name}_{regime}.npz'
              for name, regime, tag in c.dev_cases()]
    for path in paths:
        hashes[path.relative_to(c.REPO).as_posix()] = c.digest(path)
    return hashes, old['external_sha256']


def verify_registry():
    value = json.loads((c.OUT / 'preregistered.json').read_text())
    if value['device'] != os.environ['FARRAS_TRAIN_DEVICE']:
        raise ValueError('Backend differs from preregistration')
    for relative, expected in value['sha256'].items():
        if c.digest(c.REPO / relative) != expected:
            raise ValueError('Frozen source/input changed: ' + relative)
    for relative, expected in value['external_sha256'].items():
        if c.digest(c.REPO.parent / relative) != expected:
            raise ValueError('Frozen raw source changed: ' + relative)
    return value


def register():
    if (c.OUT / 'selection.json').exists():
        raise FileExistsError('Selection frozen; create a new campaign')
    if (c.OUT / 'preregistered.json').exists():
        return verify_registry()
    if tuple(row['name'] for row in models.CONFIGS) != BASE:
        raise ValueError('Model grid differs from plan')
    if len(fusion.FAMILIES) != 12 or len(decoder.CONFIGS) != 8:
        raise ValueError('Probability/decoder grid differs from plan')
    hashes, external = inherited_manifest()
    value = dict(sha256=hashes, external_sha256=external, device=os.environ['FARRAS_TRAIN_DEVICE'],
        models=models.CONFIGS, families=list(fusion.FAMILIES), decoders=decoder.CONFIGS,
        development=list(c.DEV), fold_of=c.FOLD, holdout=list(c.HOLD),
        promotion=dict(min_spellF1_gain=0.01, min_winning_providers=2, max_provider_drop=0.03),
        stress_seed=20261113, public_used=False,
        exposure='All original holdout matches previously seen. Stress transformation is new, not fresh matches.')
    c.write_json(c.OUT / 'preregistered.json', value)
    return value


def fit_job(family, names, tests, tag, final=False):
    paths = [prediction_path(family, tag, name, regime) for name, regime in tests]
    if all(path.exists() for path in paths) and (not final or checkpoint(family).exists()):
        print('REUSE', family, tag, flush=True)
        return
    started = time.monotonic()
    datasets = c.training(names)
    with threadpool_limits(limits=1):
        fitted = models.fit(datasets, settings(models, family))
    del datasets
    gc.collect()
    for (name, regime), path in zip(tests, paths):
        data = c.load(name, regime, with_features=True)
        with threadpool_limits(limits=1):
            output = models.predict(fitted, c.observations(data))
        save_prediction(path, output)
        del data, output
    if final:
        checkpoint(family).parent.mkdir(parents=True, exist_ok=True)
        torch.save(fitted, checkpoint(family))
    print('FIT', family, tag, 'seconds', round(time.monotonic() - started, 1), flush=True)
    del fitted
    release()


def prior(name, regime):
    root = c.CACHE if regime == 'stress' else c.base.CACHE
    values = read_prediction(root / 'prior' / f'{name}_{regime}.npz')
    return dict(probability=values['hybrid_probability'], boundary=values['hybrid_boundary'], has_boundary=False)


def old_pp(name, regime, tag, memo):
    if tag != 'holdout':
        return read_prediction(c.base.CACHE / 'predictions/pp_tcn' / tag / f'{name}_{regime}.npz')
    return infer('old_pp', name, regime, memo)


def infer(family, name, regime, memo):
    key = family, name, regime
    if key not in memo:
        path = prediction_path(family, 'holdout', name, regime)
        if path.exists():
            memo[key] = read_prediction(path)
        else:
            import models as original_models
            model_path = c.PREVIOUS / 'out/models/pp_tcn.pt' if family == 'old_pp' else checkpoint(family)
            fitted = torch.load(model_path, weights_only=False)
            data = c.load(name, regime, with_features=True)
            with threadpool_limits(limits=1):
                output = (original_models if family == 'old_pp' else models).predict(fitted, c.observations(data))
            save_prediction(path, output)
            memo[key] = output
            del fitted, data
            release()
    return memo[key]


def probability(family, name, regime, tag, memo=None, replacements=None):
    memo = {} if memo is None else memo
    predictions = {}
    for base in new_dependencies(family):
        if replacements is not None and base in replacements:
            predictions[base] = replacements[base]
        elif tag == 'holdout':
            predictions[base] = infer(base, name, regime, memo)
        else:
            predictions[base] = read_prediction(prediction_path(base, tag, name, regime))
    return fusion.combine(family, predictions, old_pp(name, regime, tag, memo), prior(name, regime),
        c.observations(c.load(name, regime)))


def summarize(rows):
    means = []
    for provider in ('idsse', 'skillcorner', 'metrica'):
        matches = sorted({row['match'] for row in rows if row['source'] == provider})
        means.append(dict(source=provider, **{key: float(np.mean([
            np.mean([row[field] for row in rows if row['match'] == match]) for match in matches]))
            for key, field in (('spellF1', 'spell_f1_mean'), ('macroF1', 'macro_f1'))}))
    return dict(provider_means=means, external_mean_spellF1=float(np.mean([r['spellF1'] for r in means])),
        external_mean_macroF1=float(np.mean([r['macroF1'] for r in means])),
        max_abs_count_error=float(max(abs(row['count_error_ratio']) for row in rows)))


def collect():
    verify_registry()
    if (c.OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen')
    stats_by_tag = {f'fold{fold}': c.training_stats([n for n in c.DEV if c.FOLD[n] != fold]) for fold in range(6)}
    stats_by_tag['final'] = c.training_stats(c.DEV)
    records = []
    for family in fusion.FAMILIES:
        rows_by_decoder = {s['name']: [] for s in decoder.CONFIGS}
        for name, regime, tag in c.dev_cases():
            data = c.load(name, regime)
            output = probability(family, name, regime, tag)
            observations = c.base.decoder_observations(data)
            for spec in decoder.CONFIGS:
                result = decoder.decode(output, observations, spec, stats_by_tag[tag])
                row = c.metrics(result['codes'], data)
                row.update(match=name, source=c.source(name), regime=regime,
                    calibrated_lambda=float(result['lambda']), target_count=float(result['target_count']))
                rows_by_decoder[spec['name']].append(row)
        for spec in decoder.CONFIGS:
            rows = rows_by_decoder[spec['name']]
            records.append(dict(family=family, decoder=spec, external_folds=rows, **summarize(rows)))
    records.sort(key=lambda r: (-r['external_mean_spellF1'], -r['external_mean_macroF1'], r['family'], r['decoder']['name']))
    control = next(r for r in records if r['family'] == CONTROL['family'] and r['decoder']['name'] == CONTROL['decoder'])
    best_new = next(r for r in records if r is not control)
    eligible = []
    for row in records:
        delta = [p['spellF1'] - b['spellF1'] for p, b in zip(row['provider_means'], control['provider_means'])]
        row['gain_vs_control'] = row['external_mean_spellF1'] - control['external_mean_spellF1']
        row['eligible'] = row['gain_vs_control'] >= 0.01 and sum(d > 0 for d in delta) >= 2 and min(delta) >= -0.03
        if row['eligible']:
            eligible.append(row)
    selected = eligible[0] if eligible else control
    spec_of = lambda row: dict(family=row['family'], decoder=row['decoder'])
    result = dict(selected=spec_of(selected), best_new=spec_of(best_new), control=spec_of(control),
        promoted=bool(eligible), records=records, preregistered_sha256=c.digest(c.OUT / 'preregistered.json'),
        checkpoint_sha256={family: c.digest(checkpoint(family)) for family in BASE},
        public_used_for_selection=False, holdout_used_for_selection=False)
    c.write_json(c.OUT / 'selection.json', result)
    import pandas as pd
    pd.DataFrame([dict(family=r['family'], decoder=r['decoder']['name'], spellF1=r['external_mean_spellF1'],
        macroF1=r['external_mean_macroF1'], gain=r['gain_vs_control'], eligible=r['eligible'],
        max_abs_count_error=r['max_abs_count_error']) for r in records]).to_csv(c.OUT / 'external_scoreboard.csv', index=False)
    print('FROZEN', result['selected'], 'promoted', result['promoted'], 'best_raw', result['best_new'], flush=True)
    return result


def development():
    started = time.monotonic()
    register()
    for fold in range(6):
        names = [name for name in c.DEV if c.FOLD[name] != fold]
        tests = [(n, 'r0' if n.startswith('J03') else 'clean') for n in c.DEV if c.FOLD[n] == fold]
        for family in BASE:
            fit_job(family, names, tests, f'fold{fold}')
    tests = [(f'metrica{n}', regime) for n in (1, 2) for regime in ('clean', 'r0')]
    for family in BASE:
        fit_job(family, c.DEV, tests, 'final', final=True)
    collect()
    c.write_json(c.OUT / 'execution.json', dict(seconds=time.monotonic() - started,
        ensemble_fits=21, seed_fits=63, configurations=96, development_evaluations=2208,
        development_matches=21, evaluation_scenarios=23, device=os.environ['FARRAS_TRAIN_DEVICE']))


def frozen_selection():
    verify_registry()
    selection = json.loads((c.OUT / 'selection.json').read_text())
    if selection['preregistered_sha256'] != c.digest(c.OUT / 'preregistered.json'):
        raise ValueError('Registry changed')
    for family, expected in selection['checkpoint_sha256'].items():
        if c.digest(checkpoint(family)) != expected:
            raise ValueError('Model checkpoint changed: ' + family)
    return selection


def candidate_specs(selection):
    return dict(selected=selection['selected'], best_new=selection['best_new'], previous_wave4=selection['control'],
        pure_boundary=dict(family='boundary_tcn', decoder=settings(decoder, 'wave4')))


def codes_for(spec, output, data, stats):
    return decoder.decode(output, c.base.decoder_observations(data), spec['decoder'], stats)['codes']


def independent_rebuild(spec, data, stats, memo):
    replacements = {}
    for family in new_dependencies(spec['family']):
        datasets = c.training(c.DEV)
        with threadpool_limits(limits=1):
            fitted = models.fit(datasets, settings(models, family))
        del datasets
        with threadpool_limits(limits=1):
            replacements[family] = models.predict(fitted, c.observations(data))
        del fitted
        release()
    output = probability(spec['family'], 'target', 'clean', 'holdout', memo, replacements)
    path = c.OUT / 'independent_rebuild.csv'
    actual = old_runner.submission(path, codes_for(spec, output, data, stats), data)
    path.unlink()
    return actual, len(replacements) * 3


def verify_postfreeze_inputs():
    stress_path = c.CACHE / 'stress_audit.json'
    reference_path = c.CACHE / 'reference_audit.json'
    stress = json.loads(stress_path.read_text())
    reference = json.loads(reference_path.read_text())
    selection_hash = c.digest(c.OUT / 'selection.json')
    if any(audit['selection_sha256'] != selection_hash for audit in (stress, reference)):
        raise ValueError('Postfreeze inputs belong to a different selection')
    if reference['stress_manifest_sha256'] != c.digest(stress_path):
        raise ValueError('Stress inputs changed after reference inference')
    expected_keys = {f'{name}_stress' for name in c.SC_HOLD}
    if set(stress['packs']) != expected_keys or set(reference['packs']) != expected_keys:
        raise ValueError('Postfreeze preparation must cover exactly six holdout matches')
    paths = [stress_path, reference_path]
    for name in c.SC_HOLD:
        entry = stress['packs'][f'{name}_stress']
        prior_entry = reference['packs'][f'{name}_stress']
        for path, field, prior_field in (
            (c.base.pack_path(name, 'clean'), 'source_sha256', 'source_sha256'),
            (c.pack_path(name, 'stress'), 'pack_sha256', 'input_pack_sha256'),
            (c.CACHE / 'features' / f'{name}_stress.npz', 'feature_sha256', 'input_feature_sha256')):
            if c.digest(path) != entry[field] or prior_entry[prior_field] != entry[field]:
                raise ValueError('Stress/reference provenance mismatch: ' + name)
            paths.append(path)
        path = c.CACHE / 'prior' / f'{name}_stress.npz'
        if c.digest(path) != prior_entry['sha256']:
            raise ValueError('Frozen GRU stress prediction changed: ' + name)
        paths.append(path)
    return {path.relative_to(c.REPO).as_posix(): c.digest(path) for path in paths}


def holdout():
    selection = frozen_selection()
    prepared_hashes = verify_postfreeze_inputs()
    stats = c.training_stats(c.DEV)
    specs = candidate_specs(selection)
    memo, rows, duration_rows, artifacts = {}, [], [], {}
    for name, regime in c.holdout_cases():
        data = c.load(name, regime)
        for role, spec in specs.items():
            output = probability(spec['family'], name, regime, 'holdout', memo)
            codes = codes_for(spec, output, data, stats)
            row = c.metrics(codes, data)
            row.update(candidate=role, family=spec['family'], decoder=spec['decoder']['name'], match=name,
                regime=regime, source=c.source(name))
            rows.append(row)
            duration_rows.append(dict(candidate=role, match=name, regime=regime, **diagnostics.summarize(codes, data)))
            print('HOLDOUT', role, name, regime, row['spell_f1_mean'], flush=True)
    for name, regime, tag in c.dev_cases():
        data = c.load(name, regime)
        fold_stats = c.training_stats(c.DEV if tag == 'final' else [n for n in c.DEV if c.FOLD[n] != c.FOLD[name]])
        for role in ('selected', 'previous_wave4'):
            output = probability(specs[role]['family'], name, regime, tag)
            codes = codes_for(specs[role], output, data, fold_stats)
            duration_rows.append(dict(candidate=role, match=name, regime=regime, phase='development',
                **diagnostics.summarize(codes, data)))
    data = c.load('target', 'clean', with_features=True)
    if np.any(data['y'] >= 0) or np.any(data['y25'] >= 0):
        raise ValueError('Target must not contain labels')
    for role, spec in specs.items():
        output = probability(spec['family'], 'target', 'clean', 'holdout', memo)
        path = c.OUT / 'candidates' / (role + '.csv')
        artifacts[role] = dict(path=path.relative_to(c.HERE).as_posix(),
            sha256=old_runner.submission(path, codes_for(spec, output, data, stats), data), **spec)
    expected = json.loads((c.PREVIOUS / 'out/holdout_report.json').read_text())['reproduced_sha256']
    if artifacts['previous_wave4']['sha256'] != expected:
        raise ValueError('Previous wave4 candidate not reproduced byte-for-byte')
    started = time.monotonic()
    reproduced, seed_fits = independent_rebuild(specs['selected'], data, stats, memo)
    if reproduced != artifacts['selected']['sha256']:
        raise ValueError('Independent selected-component refit/rebuild differs')
    report = dict(selection_sha256=c.digest(c.OUT / 'selection.json'), results=rows, submissions=artifacts,
        reproduced_sha256=reproduced, identical_target_rows=199319, new_component_refit_seed_fits=seed_fits,
        independent_rebuild_seconds=time.monotonic() - started,
        stress_audit_sha256=c.digest(c.CACHE / 'stress_audit.json'),
        reference_audit_sha256=c.digest(c.CACHE / 'reference_audit.json'),
        postfreeze_sha256=prepared_hashes,
        selection_changed=False, exposure='Adaptive old holdouts; paired stress is not additional independent matches.')
    c.write_json(c.OUT / 'duration_diagnostics.json', dict(records=duration_rows))
    c.write_json(c.OUT / 'holdout_report.json', report)
    print('REBUILD IDENTICAL', reproduced, 'new seed refits', seed_fits, flush=True)


def public_audit():
    frozen_selection()
    report = json.loads((c.OUT / 'holdout_report.json').read_text())
    if report['selection_sha256'] != c.digest(c.OUT / 'selection.json'):
        raise ValueError('Different selection in holdout')
    if verify_postfreeze_inputs() != report['postfreeze_sha256']:
        raise ValueError('Prepared inputs changed after holdout')
    sys.path.insert(0, str(c.REPO / 'evaluation'))
    import run_evaluation as official
    gt = c.pm.label_codes(c.pm.read_label_csv(official.find_gt('public')))
    rows = []
    for role, artifact in report['submissions'].items():
        path = c.HERE / artifact['path']
        if c.digest(path) != artifact['sha256']:
            raise ValueError('Candidate changed: ' + role)
        result, reason = official.score_submission(str(path), gt, c.pm.TAUS, 0)
        if reason:
            raise ValueError(reason)
        rows.append(dict(candidate=role, family=artifact['family'], decoder=artifact['decoder']['name'],
            spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'], possessions=result['n_pred_spells'],
            sha256=artifact['sha256']))
    c.write_json(c.OUT / 'public_audit.json', dict(selection_sha256=report['selection_sha256'], records=rows,
        distinct_submission_hashes=len({r['sha256'] for r in rows}), public_used_for_selection=False))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dev', 'collect', 'holdout', 'public'])
    {'dev': development, 'collect': collect, 'holdout': holdout, 'public': public_audit}[parser.parse_args().action]()
