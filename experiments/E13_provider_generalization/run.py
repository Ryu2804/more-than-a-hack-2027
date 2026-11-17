import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time

import access
access.initialize()

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bridge as b
import models
import seal

PRIMARY = 'modality_dropout'


def jobs():
    return [(source, family) for source in b.PROVIDERS for family in models.FAMILIES]


def prediction_id(source, family, name, condition):
    return f'prediction/{source}/{family}/{name}/{condition}'


def fit_paths(source, family):
    base = b.OUT / 'jobs' / f'{source}_{family}'
    return base.with_suffix('.pt'), base.with_suffix('.json'), b.OUT / 'replays' / f'{source}_{family}.json'


def prediction_path(source, family, name, condition):
    return b.OUT / 'predictions' / source / family / f'{name}_{condition}.npz'


def artifact_paths():
    paths = {}
    for source, family in jobs():
        checkpoint, metadata, proof = fit_paths(source, family)
        for kind, path in (('checkpoint', checkpoint), ('metadata', metadata), ('replay', proof)):
            paths[f'{kind}/{source}/{family}'] = path
        for name in b.test_names(source):
            for condition in b.CONDITIONS:
                paths[prediction_id(source, family, name, condition)] = prediction_path(source, family, name, condition)
    if len(paths) != 300:
        raise ValueError('Protocol requires exactly 300 committed artifacts')
    return paths


def verify_inputs(role, source=None):
    path = b.OUT / 'preregistered.json'
    registry = json.loads(path.read_text())
    for relative, expected in registry['code_sha256'].items():
        if b.c.digest(b.c.REPO / relative) != expected:
            raise ValueError('Frozen code/preparation changed: ' + relative)
    for row in registry['inputs']:
        if role == 'fit':
            if row['provider'] != source or row['kind'] not in ('original_pack', 'original_features'):
                continue
        elif role in ('predict', 'replay'):
            if row['kind'] not in ('observations', 'features'):
                continue
        elif role == 'score':
            if row['kind'] not in ('observations', 'features', 'label'):
                continue
        if b.c.digest(b.c.REPO / row['path']) != row['sha256']:
            raise ValueError('Frozen role input changed: ' + row['path'])
    if registry['device'] != str(b.original._device()):
        raise ValueError('Training/inference device changed')
    return b.c.digest(path)


def register():
    path = b.OUT / 'preregistered.json'
    if path.exists():
        return verify_inputs('register')
    prep = json.loads((b.OUT / 'preparation.json').read_text())
    checks = json.loads((b.OUT / 'checks.json').read_text())
    if checks['failures'] or checks['errors'] or {row['family'] for row in checks['smoke']} != set(models.FAMILIES):
        raise ValueError('Complete semantic and five-family real smoke checks required')
    if {row['match'] for row in prep['records']} != set((*b.IDSSE, *b.SC)):
        raise ValueError('Preparation must cover all 27 external matches')
    inputs = []
    for record in prep['records']:
        for field in ('original_pack', 'original_features'):
            inputs.append(dict(kind=field, provider=record['provider'], path=record[field], sha256=record[field + '_sha256']))
        inputs.append(dict(kind='label', provider=record['provider'], path=record['label_path'], sha256=record['label_sha256']))
        for case in record['cases']:
            for field in ('observations', 'features'):
                inputs.append(dict(kind=field, provider=record['provider'], path=case[field], sha256=case[field + '_sha256']))
    source_paths = set(b.HERE.glob('*.py')) | {b.HERE / 'PLAN.md', b.OUT / 'preparation.json', b.OUT / 'checks.json',
                                               b.c.HERE / 'prepare_stress.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            candidate = __import__('pathlib').Path(filename).resolve()
            if candidate.suffix == '.py' and b.c.REPO in candidate.parents and candidate.is_file():
                source_paths.add(candidate)
    code = {item.relative_to(b.c.REPO).as_posix(): b.c.digest(item) for item in sorted(source_paths)}
    value = dict(code_sha256=code, inputs=inputs, primary=PRIMARY, control='control',
        train={source: list(b.train_names(source)) for source in b.PROVIDERS},
        test={source: list(b.test_names(source)) for source in b.PROVIDERS},
        families=list(models.FAMILIES), conditions=list(b.CONDITIONS), seeds=[0, 1, 2], steps=500,
        batch=16, crop=384, parameters=models.PARAMETERS, device=str(b.original._device()),
        new_neural_seed_fits=30, new_optimizer_updates=15000, expected_prediction_arrays=270, expected_sealed_artifacts=300,
        bootstrap=dict(n=5000, seed=20261120, unit='unique match within target provider', provider_weight='equal', multiple_comparison_adjusted=False),
        promotion=dict(min_gain=.01, min_provider_delta=-.02, lower_bound_positive=True),
        competition_public_access_denied=True, target_game_used=False, new_public_evaluation=False,
        exposure='All 27 matches historically exposed. Source-provider disjoint fits and sealed predictions, not fresh blind validation.',
        guard=access.report())
    b.c.write_json(path, value)
    return verify_inputs('register')


def identity(source, family, manifest, steps=500):
    return dict(source=source, family=family, training_names=list(b.train_names(source)), test_names=list(b.test_names(source)),
                seeds=[0, 1, 2], steps=steps, manifest_sha256=manifest)


def validate_fit(fitted, source, family, steps):
    if fitted['source'] != source or fitted['family'] != family or tuple(fitted['training_names']) != b.train_names(source):
        raise ValueError('Source-only checkpoint identity mismatch')
    expected = [(seed, steps, models.PARAMETERS) for seed in (0, 1, 2)]
    actual = [(row['seed'], row['updates'], row['parameters']) for row in fitted['training']['seed_fits']]
    if actual != expected or set(fitted['training_names']) & set(b.test_names(source)):
        raise ValueError('Actual training budget or membership differs')


def fit_job(source, family):
    access.restrict('fit', source)
    manifest = verify_inputs('fit', source)
    checkpoint, metadata, _ = fit_paths(source, family)
    if checkpoint.exists() or metadata.exists():
        read_metadata(source, family, manifest)
        return
    started = time.monotonic()
    datasets = [b.c.load(name, b.regime(name), with_features=True) for name in b.train_names(source)]
    fitted = models.fit(datasets, source, family)
    validate_fit(fitted, source, family, 500)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, checkpoint)
    b.c.write_json(metadata, dict(identity=identity(source, family, manifest), checkpoint_sha256=b.c.digest(checkpoint),
        training=fitted['training'], training_stats=fitted['training_stats'], seconds=time.monotonic() - started, guard=access.report()))
    print('FIT SOURCE ONLY', source, family, '3seeds/500updates', 'seconds', round(time.monotonic() - started, 1), flush=True)


def read_metadata(source, family, manifest):
    checkpoint, metadata, _ = fit_paths(source, family)
    value = json.loads(metadata.read_text())
    if value['identity'] != identity(source, family, manifest) or value['checkpoint_sha256'] != b.c.digest(checkpoint):
        raise ValueError('Checkpoint provenance mismatch')
    if value['guard']['nonprobe_blocked_attempts']:
        raise ValueError('Fit attempted forbidden data access')
    if value['training']['names'] != list(b.train_names(source)):
        raise ValueError('Actual source training names mismatch')
    actual = [(row['seed'], row['updates'], row['parameters']) for row in value['training']['seed_fits']]
    if actual != [(seed, 500, models.PARAMETERS) for seed in (0, 1, 2)]:
        raise ValueError('Actual seed budget mismatch')
    return value


def save_prediction(path, probability, codes, key, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive['identity']) != key or str(archive['manifest_sha256']) != manifest or not np.array_equal(probability, archive['probability']) or not np.array_equal(codes, archive['codes']):
                raise ValueError('Existing prediction differs; no overwrite permitted')
        return
    np.savez_compressed(path, probability=probability, codes=codes.astype(np.int8), identity=key, manifest_sha256=manifest)


def infer(source, family, replay=False):
    access.restrict('replay' if replay else 'predict', source)
    manifest = verify_inputs('replay' if replay else 'predict', source)
    metadata = read_metadata(source, family, manifest)
    fitted = torch.load(fit_paths(source, family)[0], map_location='cpu', weights_only=False)
    validate_fit(fitted, source, family, 500)
    records = []
    for name in b.test_names(source):
        for condition in b.CONDITIONS:
            observed = b.load_observations(name, condition)
            output = models.predict(fitted, observed)['probability']
            codes = b.decode(b.result(output), b.decoder_observations(observed), fitted['training_stats'])
            path = prediction_path(source, family, name, condition)
            key = prediction_id(source, family, name, condition)
            if replay:
                with np.load(path, allow_pickle=False) as archive:
                    if str(archive['identity']) != key or str(archive['manifest_sha256']) != manifest or not np.array_equal(output, archive['probability']) or not np.array_equal(codes, archive['codes']):
                        raise ValueError('Saved model probability/decoder replay differs')
            else:
                save_prediction(path, output, codes, key, manifest)
            records.append(dict(id=key, sha256=b.c.digest(path)))
            del observed
        print('REPLAY' if replay else 'PREDICT', source, family, name, 'NO TEST LABEL ACCESS', flush=True)
        gc.collect()
    guard_report = access.report()
    if replay:
        prediction_access = json.loads((b.OUT / 'access' / f'predict_{source}_{family}.json').read_text())
        if prediction_access['identity'] != identity(source, family, manifest) or prediction_access['guard']['nonprobe_blocked_attempts']:
            raise ValueError('Prediction read policy proof mismatch')
        b.c.write_json(fit_paths(source, family)[2], dict(identity=identity(source, family, manifest),
            checkpoint_sha256=metadata['checkpoint_sha256'], records=records, probability_and_codes_identical=True,
            guard=guard_report, prediction_guard=prediction_access['guard']))
    else:
        b.c.write_json(b.OUT / 'access' / f'predict_{source}_{family}.json', dict(identity=identity(source, family, manifest), guard=guard_report))
    print('ALL INFERENCE IDENTICAL' if replay else 'PREDICTIONS READY', source, family, len(records), flush=True)


def all_job(source, family):
    for action in ('job', 'predict', 'replay'):
        subprocess.run([sys.executable, '-B', str(b.HERE / 'run.py'), action, '--source', source, '--family', family], check=True)


def seal_predictions():
    manifest = verify_inputs('predict')
    expected = artifact_paths()
    for source, family in jobs():
        metadata = read_metadata(source, family, manifest)
        proof = json.loads(fit_paths(source, family)[2].read_text())
        expected_ids = {prediction_id(source, family, name, condition) for name in b.test_names(source) for condition in b.CONDITIONS}
        if proof['identity'] != identity(source, family, manifest) or proof['checkpoint_sha256'] != metadata['checkpoint_sha256'] or not proof['probability_and_codes_identical']:
            raise ValueError('Full saved-model replay proof mismatch')
        if {row['id'] for row in proof['records']} != expected_ids or len(proof['records']) != len(expected_ids):
            raise ValueError('Replay proof omits target predictions')
        for row in proof['records']:
            if b.c.digest(expected[row['id']]) != row['sha256']:
                raise ValueError('Prediction changed after inference replay')
        if proof['guard']['nonprobe_blocked_attempts'] or proof['prediction_guard']['nonprobe_blocked_attempts']:
            raise ValueError('Forbidden test-label read attempted')
    path = b.OUT / 'predictions_seal.json'
    if path.exists():
        seal.verify(path, manifest, expected)
    else:
        seal.create(path, manifest, [dict(id=key, path=str(value.resolve()), sha256=b.c.digest(value)) for key, value in expected.items()], expected)
    print('SEALED ALL', len(expected), 'ARTIFACTS BEFORE TEST SCORING', flush=True)


def paired(record, control, condition):
    rng = np.random.default_rng(20261120)
    means, delta_by_provider, wins = [], {}, 0
    for target in ('idsse', 'skillcorner'):
        left = sorted((row for row in record['rows'] if row['condition'] == condition and row['target_provider'] == target), key=lambda row: row['match'])
        right = sorted((row for row in control['rows'] if row['condition'] == condition and row['target_provider'] == target), key=lambda row: row['match'])
        if [row['match'] for row in left] != [row['match'] for row in right] or len(left) != len(b.PROVIDERS[target]):
            raise ValueError('Paired target-provider coverage mismatch')
        delta = np.array([a['spell_f1_mean'] - z['spell_f1_mean'] for a, z in zip(left, right)])
        draws = rng.integers(0, len(delta), size=(5000, len(delta)))
        means.append(delta[draws].mean(1))
        delta_by_provider[target] = float(delta.mean())
        wins += int((delta > 0).sum())
    low, high = np.quantile(np.mean(means, axis=0), (.025, .975))
    return dict(mean_delta=float(np.mean(list(delta_by_provider.values()))), provider_delta=delta_by_provider,
                lower=float(low), upper=float(high), wins=wins, matches=27)


def compute_scores(manifest):
    path = b.OUT / 'predictions_seal.json'
    seal.verify(path, manifest, artifact_paths())
    verify_inputs('score')
    records = {family: dict(family=family, rows=[]) for family in models.FAMILIES}
    for source, family in jobs():
        for name in b.test_names(source):
            with np.load(b.CACHE / 'labels' / f'{name}.npz', allow_pickle=False) as archive:
                labels = {key: archive[key] for key in archive.files}
            for condition in b.CONDITIONS:
                observed = b.load_observations(name, condition)
                with np.load(prediction_path(source, family, name, condition), allow_pickle=False) as archive:
                    codes = archive['codes']
                    if str(archive['manifest_sha256']) != manifest or str(archive['identity']) != prediction_id(source, family, name, condition):
                        raise ValueError('Sealed prediction identity mismatch')
                row = dict(match=name, source_provider=source, target_provider=b.provider(name), condition=condition,
                           **b.score(codes, observed, labels))
                records[family]['rows'].append(row)
            print('SCORE SEALED', source, family, name, flush=True)
        gc.collect()
    control = records['control']
    for record in records.values():
        record['summary'] = {}
        for condition in b.CONDITIONS:
            provider_means = {}
            for target in b.PROVIDERS:
                rows = [row for row in record['rows'] if row['target_provider'] == target and row['condition'] == condition]
                provider_means[target] = dict(spellF1=float(np.mean([row['spell_f1_mean'] for row in rows])),
                    macroF1=float(np.mean([row['macro_f1'] for row in rows])), matches=len(rows))
            record['summary'][condition] = dict(provider=provider_means,
                equal_provider_spellF1=float(np.mean([value['spellF1'] for value in provider_means.values()])),
                equal_provider_macroF1=float(np.mean([value['macroF1'] for value in provider_means.values()])),
                paired_vs_control=paired(record, control, condition))
    primary = records[PRIMARY]
    effect = primary['summary']['clean']['paired_vs_control']
    promoted = effect['mean_delta'] >= .01 and min(effect['provider_delta'].values()) >= -.02 and effect['lower'] > 0
    ranked = sorted(records.values(), key=lambda row: (-row['summary']['clean']['equal_provider_spellF1'], row['family']))
    report = dict(manifest_sha256=manifest, seal_sha256=b.c.digest(path), records=ranked, primary=primary, control=control,
        primary_promoted_for_research=promoted, selected_for_research=PRIMARY if promoted else 'control',
        best_raw=ranked[0]['family'], unique_target_matches=27, paired_conditions=54, metric_rows=270,
        new_neural_seed_fits=30, new_optimizer_updates=15000, scorer_labels_opened_only_after_complete_seal=True,
        competition_public_read_denied=True, target_game_used=False, new_public_evaluation=False,
        fresh_blind_test=False, overfit_absence_established=False)
    access.report()
    return report


def score_all():
    if (b.OUT / 'results.json').exists():
        raise FileExistsError('Completed scored campaign immutable; do not tune/re-run to choose variants')
    manifest = verify_inputs('predict')
    report = compute_scores(manifest)
    b.c.write_json(b.OUT / 'results.json', report)
    b.c.write_json(b.OUT / 'score_access.json', access.report())
    print('PRIMARY', report['primary']['summary']['clean'], 'PROMOTED', report['primary_promoted_for_research'], flush=True)
    for row in report['records']:
        print(row['family'], 'CLEAN', row['summary']['clean']['equal_provider_spellF1'],
              'GAP', row['summary']['gap']['equal_provider_spellF1'], flush=True)


def verify_scores():
    manifest = verify_inputs('predict')
    expected = json.loads((b.OUT / 'results.json').read_text())
    actual = compute_scores(manifest)
    if actual != expected:
        raise ValueError('Independent sealed metric reproduction differs')
    b.c.write_json(b.OUT / 'verification.json', dict(manifest_sha256=manifest,
        seal_sha256=expected['seal_sha256'], identical_metric_rows=270, identical_probability_and_decoded_replays=270,
        saved_model_replays=10, sealed_artifacts=300, full_training_refit=False,
        scorer_labels_opened_only_after_complete_seal=True, competition_public_read_denied=True,
        fresh_blind_test=False, overfit_absence_established=False, guard=access.report()))
    print('INDEPENDENT METRICS IDENTICAL', 270, 'SEALED ARTIFACTS', 300, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'job', 'predict', 'replay', 'all-job', 'seal', 'score', 'verify', 'all'))
    parser.add_argument('--source', choices=tuple(b.PROVIDERS))
    parser.add_argument('--family', choices=models.FAMILIES)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'register': register()
        elif args.action == 'job': fit_job(args.source, args.family)
        elif args.action == 'predict': infer(args.source, args.family)
        elif args.action == 'replay': infer(args.source, args.family, True)
        elif args.action == 'all-job': all_job(args.source, args.family)
        elif args.action == 'seal': seal_predictions()
        elif args.action == 'score': score_all()
        elif args.action == 'verify': verify_scores()
        else:
            register()
            for source, family in jobs(): all_job(source, family)
            seal_predictions()
            score_all()
