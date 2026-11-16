import argparse
import json
import os
import sys
import time
from functools import cache

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

import core

c = core.c
OUT = core.OUT
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261114

@cache

def grid():
    methods = core.modules()
    if len(methods) != 6 or any(len(method.CONFIGS) != 6 for method in methods):
        raise ValueError('Expected six methods with six settings each')
    values = [('control', None, {})]
    for method in methods:
        stem = method.__file__.split('/')[-1].split('\\')[-1].removesuffix('.py')
        values.extend((stem + '/' + setting['name'], method, setting) for setting in method.CONFIGS)
    if len({name for name, _, _ in values}) != 37:
        raise ValueError('Duplicate experiment identifiers')
    return values


def register():
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection frozen; start another campaign')
    if (OUT / 'preregistered.json').exists():
        return verify()
    core.previous.frozen_selection()
    core.previous.verify_postfreeze_inputs()
    paths = list(core.HERE.glob('*.py')) + [core.HERE / 'PLAN.md', core.PREVIOUS / 'out/selection.json',
        core.PREVIOUS / 'out/preregistered.json', core.PREVIOUS / 'out/holdout_report.json']
    paths += list((core.PREVIOUS / 'cache/predictions').rglob('*.npz'))
    paths += list((c.base.CACHE / 'predictions/pp_tcn').rglob('*.npz'))
    paths += list((core.PREVIOUS / 'cache/prior').glob('*.npz')) + list((c.base.CACHE / 'prior').glob('*.npz'))
    inherited = json.loads((core.PREVIOUS / 'out/preregistered.json').read_text())
    hashes = dict(inherited['sha256'])
    hashes.update(core.previous.verify_postfreeze_inputs())
    for path in paths:
        hashes[path.relative_to(c.REPO).as_posix()] = c.digest(path)
    value = dict(sha256=hashes, external_sha256=inherited['external_sha256'],
        grid=[dict(name=name, config=setting) for name, _, setting in grid()],
        promotion=dict(min_spellF1_gain=0.005, min_winning_providers=2, max_provider_drop=0.01,
            paired_bootstrap_lower_strictly_positive=True),
        bootstrap=dict(n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, unit='match within provider',
            interval='unadjusted exploratory percentile 95%; not family-wise confidence'),
        reused_wave5_selection_sha256=c.digest(core.PREVIOUS / 'out/selection.json'),
        training_seed_fits=0, runtime='CPU decoding, frozen probability caches', public_used=False,
        exposure='All original data and ball-drop stress previously exposed; no fresh blind matches.')
    c.write_json(OUT / 'preregistered.json', value)
    return value


def verify():
    value = json.loads((OUT / 'preregistered.json').read_text())
    for relative, expected in value['sha256'].items():
        if c.digest(c.REPO / relative) != expected:
            raise ValueError('Frozen input/source changed: ' + relative)
    for relative, expected in value['external_sha256'].items():
        if c.digest(c.REPO.parent / relative) != expected:
            raise ValueError('Frozen raw source changed: ' + relative)
    return value


def prediction(identifier, bundle, observations, stats):
    if identifier == 'control':
        return core.decode(bundle['short'], observations, stats)
    name, module, config = next(row for row in grid() if row[0] == identifier)
    output = np.asarray(module.predict(bundle, observations, stats, config))
    if output.shape != (len(observations['X']),) or not np.isin(output, (0, 1, 2)).all():
        raise ValueError('Invalid output: ' + name)
    return output.astype(np.int8, copy=False)


def paired_interval(rows, reference):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.zeros(BOOTSTRAP_N)
    provider_deltas = []
    for provider in ('idsse', 'skillcorner', 'metrica'):
        names = sorted({row['match'] for row in rows if row['source'] == provider})
        delta = np.array([np.mean([row['spell_f1_mean'] for row in rows if row['match'] == name]) -
            np.mean([row['spell_f1_mean'] for row in reference if row['match'] == name]) for name in names])
        provider_deltas.append(float(delta.mean()))
        indices = rng.integers(0, len(names), size=(BOOTSTRAP_N, len(names)))
        samples += delta[indices].mean(axis=1) / 3
    low, high = np.quantile(samples, (0.025, 0.975))
    return dict(lower=float(low), upper=float(high), provider_deltas=provider_deltas)


def development():
    started = time.monotonic()
    register()
    configs = grid()
    records = {name: dict(name=name, config=config, external_folds=[]) for name, _, config in configs}
    statistics = {f'fold{fold}': c.training_stats([name for name in c.DEV if c.FOLD[name] != fold]) for fold in range(6)}
    statistics['final'] = c.training_stats(c.DEV)
    with threadpool_limits(limits=1):
        for name, regime, tag in c.dev_cases():
            data = c.load(name, regime)
            observations = c.base.decoder_observations(data)
            bundle = core.load_bundle(name, regime, tag)
            for identifier, method, setting in configs:
                codes = core.decode(bundle['short'], observations, statistics[tag]) if method is None else method.predict(bundle, observations, statistics[tag], setting)
                codes = np.asarray(codes)
                if codes.shape != (len(data['X']),) or not np.isin(codes, (0, 1, 2)).all():
                    raise ValueError('Invalid labels: ' + identifier)
                row = c.metrics(codes, data)
                row.update(match=name, source=c.source(name), regime=regime)
                records[identifier]['external_folds'].append(row)
            print('DEV', name, regime, 'configurations', len(configs), flush=True)
    control = records['control']
    for row in records.values():
        row.update(core.previous.summarize(row['external_folds']))
        row['paired_bootstrap'] = paired_interval(row['external_folds'], control['external_folds'])
        row['gain_vs_control'] = row['external_mean_spellF1'] - core.previous.summarize(control['external_folds'])['external_mean_spellF1']
        deltas = row['paired_bootstrap']['provider_deltas']
        row['eligible'] = row['name'] != 'control' and row['gain_vs_control'] >= 0.005 and sum(delta > 0 for delta in deltas) >= 2 and min(deltas) >= -0.01 and row['paired_bootstrap']['lower'] > 0
    if abs(control['external_mean_spellF1'] - 0.670099305497763) > 1e-12:
        raise ValueError('Frozen wave5 control differs from historical result')
    ranked = sorted(records.values(), key=lambda row: (-row['external_mean_spellF1'], -row['external_mean_macroF1'], row['name']))
    alternatives = [row for row in ranked if row['name'] != 'control']
    eligible = [row for row in ranked if row['eligible']]
    report = dict(selected=(eligible[0] if eligible else control)['name'], best_new=alternatives[0]['name'],
        promoted=bool(eligible), records=ranked, preregistered_sha256=c.digest(OUT / 'preregistered.json'),
        public_used_for_selection=False, holdout_used_for_selection=False)
    c.write_json(OUT / 'selection.json', report)
    c.write_json(OUT / 'execution.json', dict(seconds=time.monotonic() - started, configurations=37,
        new_configurations=36, development_evaluations=37 * len(c.dev_cases()), training_seed_fits=0,
        development_scenarios=len(c.dev_cases()), development_unique_matches=21, runtime='CPU cached-probability decoding'))
    print('FROZEN', report['selected'], 'promoted', report['promoted'], 'best_new', report['best_new'], flush=True)
    for row in ranked[:8]:
        print(row['name'], row['external_mean_spellF1'], row['gain_vs_control'], row['eligible'], row['paired_bootstrap'], flush=True)


def selection():
    verify()
    report = json.loads((OUT / 'selection.json').read_text())
    if report['preregistered_sha256'] != c.digest(OUT / 'preregistered.json'):
        raise ValueError('Registry changed after selection')
    return report


def holdout():
    report = selection()
    roles = dict(selected=report['selected'], best_new=report['best_new'], previous_wave5='control')
    stats = c.training_stats(c.DEV)
    rows, duration, artifacts = [], [], {}
    with threadpool_limits(limits=1):
        for name, regime in c.holdout_cases():
            data = c.load(name, regime)
            observations = c.base.decoder_observations(data)
            bundle = core.load_bundle(name, regime, 'holdout')
            for role, identifier in roles.items():
                codes = prediction(identifier, bundle, observations, stats)
                row = c.metrics(codes, data)
                row.update(candidate=role, variant=identifier, match=name, source=c.source(name), regime=regime)
                rows.append(row)
                duration.append(dict(candidate=role, match=name, source=c.source(name), regime=regime,
                    **core.previous.diagnostics.summarize(codes, data)))
                print('HOLDOUT', role, name, regime, row['spell_f1_mean'], flush=True)
        data = c.load('target', 'clean')
        if np.any(data['y'] >= 0) or np.any(data['y25'] >= 0):
            raise ValueError('Target contains labels')
        observations = c.base.decoder_observations(data)
        bundle = core.load_bundle('target', 'clean', 'holdout')
        for role, identifier in roles.items():
            path = OUT / 'candidates' / (role + '.csv')
            sha = core.previous.old_runner.submission(path, prediction(identifier, bundle, observations, stats), data)
            artifacts[role] = dict(path=path.relative_to(core.HERE).as_posix(), sha256=sha, variant=identifier)
    expected = json.loads((core.PREVIOUS / 'out/holdout_report.json').read_text())['submissions']['selected']['sha256']
    if artifacts['previous_wave5']['sha256'] != expected:
        raise ValueError('Wave5 target control not reproduced byte-for-byte')
    c.write_json(OUT / 'holdout_report.json', dict(selection_sha256=c.digest(OUT / 'selection.json'),
        results=rows, submissions=artifacts, distinct_submissions=len({item['sha256'] for item in artifacts.values()}),
        exposure='Previously exposed nine matches; six paired stresses are not new matches.'))
    c.write_json(OUT / 'duration_diagnostics.json', dict(records=duration))


def rebuild():
    report = selection()
    held = json.loads((OUT / 'holdout_report.json').read_text())
    if held['selection_sha256'] != c.digest(OUT / 'selection.json'):
        raise ValueError('Holdout used different selection')
    data = c.load('target', 'clean')
    bundle = core.load_bundle('target', 'clean', 'holdout')
    codes = prediction(report['selected'], bundle, c.base.decoder_observations(data), c.training_stats(c.DEV))
    path = OUT / 'rebuild.csv'
    actual = core.previous.old_runner.submission(path, codes, data)
    path.unlink()
    expected = held['submissions']['selected']['sha256']
    if actual != expected:
        raise ValueError('Independent process rebuild differs')
    c.write_json(OUT / 'verification.json', dict(selection_sha256=c.digest(OUT / 'selection.json'),
        reproduced_sha256=actual, identical_target_rows=int(data['n_frames']), new_training_seed_fits=0,
        rebuild='Independent process; reloaded frozen probabilities and reran selected decoding, not a refit.'))
    print('REBUILD IDENTICAL', actual, int(data['n_frames']), flush=True)


def public():
    selection()
    held = json.loads((OUT / 'holdout_report.json').read_text())
    proof = json.loads((OUT / 'verification.json').read_text())
    if held['selection_sha256'] != c.digest(OUT / 'selection.json') or proof['selection_sha256'] != held['selection_sha256']:
        raise ValueError('Selection changed since holdout/rebuild')
    if proof['reproduced_sha256'] != held['submissions']['selected']['sha256']:
        raise ValueError('Selected rebuild proof differs')
    sys.path.insert(0, str(c.REPO / 'evaluation'))
    import run_evaluation as official
    gt = c.pm.label_codes(c.pm.read_label_csv(official.find_gt('public')))
    rows = []
    for role, artifact in held['submissions'].items():
        path = core.HERE / artifact['path']
        if c.digest(path) != artifact['sha256']:
            raise ValueError('Submission changed: ' + role)
        metrics, reason = official.score_submission(str(path), gt, c.pm.TAUS, 0)
        if reason:
            raise ValueError(reason)
        rows.append(dict(candidate=role, variant=artifact['variant'], spellF1=metrics['spell_f1_mean'],
            macroF1=metrics['macro_f1'], possessions=metrics['n_pred_spells'], sha256=artifact['sha256']))
    c.write_json(OUT / 'public_audit.json', dict(records=rows, selection_sha256=held['selection_sha256'], public_used_for_selection=False))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dev', 'holdout', 'rebuild', 'public'])
    with threadpool_limits(limits=1):
        dict(dev=development, holdout=holdout, rebuild=rebuild, public=public)[parser.parse_args().action]()
