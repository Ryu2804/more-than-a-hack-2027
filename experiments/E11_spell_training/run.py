import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bridge as b
import models

CONTROL = 'control/dead45'
PRIMARY = 'event_replace25/dead45'
KEYS = dict(spell='spell_balanced_tcn', event='event_tcn', dual='dual_scale_tcn')
DECODERS = ('dead45', 'consensus')


def families():
    return ('control', *(key + '_' + mode for key in KEYS for mode in ('pure', 'base75', 'replace25')))


def settings():
    return [(family, decoder) for family in families() for decoder in DECODERS]


def verify():
    registry = json.loads((b.OUT / 'preregistered.json').read_text())
    for relative, expected in registry['sha256'].items():
        if b.c.digest(b.c.REPO / relative) != expected:
            raise ValueError('Frozen source/input changed: ' + relative)
    if registry['device'] != str(b.original._device()):
        raise ValueError('Training/inference backend changed')
    return b.c.digest(b.OUT / 'preregistered.json')


def register():
    path = b.OUT / 'preregistered.json'
    if path.exists():
        return verify()
    old_path = b.PREVIOUS / 'out/preregistered.json'
    old = json.loads(old_path.read_text())
    old_hash = b.c.digest(old_path)
    old_result = json.loads((b.PREVIOUS / 'out/results.json').read_text())
    proof = json.loads((b.PREVIOUS / 'out/verification.json').read_text())
    if old_result['manifest_sha256'] != old_hash or proof['manifest_sha256'] != old_hash or proof['identical_decoded_arrays'] != 133:
        raise ValueError('Predecessor verification incomplete')
    if old_result['selected_for_research'] != CONTROL or abs(old_result['control']['mean_spellF1'] - .6154575861548992) > 1e-12:
        raise ValueError('Unexpected predecessor control')
    checks = json.loads((b.OUT / 'checks.json').read_text())
    if checks['failures'] or checks['errors'] or len(checks['smoke']) != 3:
        raise ValueError('Semantic tests and three-family real training smoke required')
    counts = {record['parameters'] for record in checks['smoke']}
    if len(counts) != 1:
        raise ValueError('Training families have unequal parameter counts')
    hashes = dict(old['sha256'])
    paths = list(b.HERE.glob('*.py')) + [b.HERE / 'PLAN.md', old_path,
        b.PREVIOUS / 'out/results.json', b.PREVIOUS / 'out/verification.json', b.OUT / 'checks.json',
        b.PREVIOUS / 'consensus.py', b.FAIR / 'bridge.py', b.c.HERE / 'training_models.py', b.c.HERE / 'diagnostics.py']
    for name in b.MATCHES:
        b.old_bundle(name)
        for family in ('short_tcn', 'setseq'):
            path_job = b.FAIR / 'out/jobs' / f'{family}_{name}.npz'
            paths.extend((path_job, path_job.with_suffix('.json')))
    for item in paths:
        hashes[item.relative_to(b.c.REPO).as_posix()] = b.c.digest(item)
    registry = dict(sha256=hashes, primary=PRIMARY, control=CONTROL, families=list(families()),
        decoders=list(DECODERS), matches=list(b.MATCHES), folds={name: list(b.fair.split_names(name)) for name in b.MATCHES},
        models=list(models.FAMILIES), seeds=[0, 1, 2], steps=500, batch=16, crop=384, input_channels=29,
        parameters=next(iter(counts)), device=str(b.original._device()), old_manifest_sha256=old_hash,
        bootstrap=dict(n=5000, seed=20261117, unit='unique match', multiple_comparison_adjusted=False),
        new_neural_seed_fits=63, new_optimizer_updates=31500, configuration_rows=20, development_evaluations=140,
        new_public_evaluation=False, outer_label_meta_fit=False,
        exposure='Repeatedly exposed research matches. Not fresh blind data. Equal updates and parameters, not convolution compute.')
    b.c.write_json(path, registry)
    return verify()


def paths(family, heldout):
    probability = b.OUT / 'jobs' / f'{family}_{heldout}.npz'
    return probability, probability.with_suffix('.json'), probability.with_suffix('.pt')


def identity(family, heldout, manifest):
    return dict(family=family, heldout=heldout, training_names=list(b.fair.split_names(heldout)),
                seeds=[0, 1, 2], steps=500, manifest_sha256=manifest)


def read_new(family, heldout, manifest):
    probability, metadata, checkpoint = paths(family, heldout)
    value = json.loads(metadata.read_text())
    if value['identity'] != identity(family, heldout, manifest):
        raise ValueError('Fresh model identity mismatch')
    if value['actual_fit_names'] != list(b.fair.split_names(heldout)):
        raise ValueError('Incorrect actual training membership')
    if value['sha256'] != b.c.digest(probability) or value['checkpoint_sha256'] != b.c.digest(checkpoint):
        raise ValueError('Model/prediction bytes changed')
    audit = value['training']['seed_fits']
    expected_parameters = json.loads((b.OUT / 'preregistered.json').read_text())['parameters']
    if [(row['seed'], row['updates'], row['parameters']) for row in audit] != [(seed, 500, expected_parameters) for seed in (0, 1, 2)]:
        raise ValueError('Actual fit budget or parameter count mismatch')
    with np.load(probability, allow_pickle=False) as archive:
        output = b.result(archive['probability'])
    return output, value


def fit_job(family, heldout):
    manifest = verify()
    probability, metadata, checkpoint = paths(family, heldout)
    if any(path.exists() for path in (probability, metadata, checkpoint)):
        read_new(family, heldout, manifest)
        return
    names = b.fair.split_names(heldout)
    b.fair.assert_fold(names, heldout)
    started = time.monotonic()
    train = [b.c.load(name, 'r0', with_features=True) for name in names]
    fitted = models.fit(train, family)
    if tuple(fitted['training_names']) != names or fitted['training']['matches'] != 6:
        raise ValueError('Fresh fit membership mismatch')
    del train
    gc.collect()
    observed = b.fair.observations(b.c.load(heldout, 'r0', with_features=True))
    output = models.predict(fitted, observed)
    probability.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, checkpoint)
    np.savez_compressed(probability, probability=output['probability'])
    b.c.write_json(metadata, dict(identity=identity(family, heldout, manifest), actual_fit_names=list(names),
        training=fitted['training'], sha256=b.c.digest(probability), checkpoint_sha256=b.c.digest(checkpoint),
        seconds=time.monotonic() - started))
    print('FIT', family, heldout, '3seeds/500updates', 'seconds', round(time.monotonic() - started, 1), flush=True)


def replay_path(family, heldout):
    return b.OUT / 'replays' / f'{family}_{heldout}.json'


def verify_replay(family, heldout, manifest, metadata):
    proof = json.loads(replay_path(family, heldout).read_text())
    if proof != dict(identity=identity(family, heldout, manifest), checkpoint_sha256=metadata['checkpoint_sha256'],
                     probability_sha256=metadata['sha256'], identical_full_probabilities=True):
        raise ValueError('Saved-model inference replay proof mismatch')


def replay_job(family, heldout):
    manifest = verify()
    _, metadata = read_new(family, heldout, manifest)
    if replay_path(family, heldout).exists():
        verify_replay(family, heldout, manifest, metadata)
        return
    fitted = torch.load(paths(family, heldout)[2], map_location='cpu', weights_only=False)
    if tuple(fitted['training_names']) != b.fair.split_names(heldout) or fitted['family'] != family:
        raise ValueError('Checkpoint training identity mismatch')
    observed = b.fair.observations(b.c.load(heldout, 'r0', with_features=True))
    rebuilt = models.predict(fitted, observed)
    with np.load(paths(family, heldout)[0], allow_pickle=False) as archive:
        if not np.array_equal(rebuilt['probability'], archive['probability']):
            raise ValueError('Independent saved-model inference differs from cached probabilities')
    b.c.write_json(replay_path(family, heldout), dict(identity=identity(family, heldout, manifest),
        checkpoint_sha256=metadata['checkpoint_sha256'], probability_sha256=metadata['sha256'], identical_full_probabilities=True))
    print('INFERENCE IDENTICAL', family, heldout, flush=True)


def fit_fold(heldout):
    manifest = verify()
    for family in models.FAMILIES:
        probability, metadata, checkpoint = paths(family, heldout)
        if any(path.exists() for path in (probability, metadata, checkpoint)):
            read_new(family, heldout, manifest)
        else:
            subprocess.run([sys.executable, '-B', str(b.HERE / 'run.py'), 'job', '--family', family, '--match', heldout], check=True)
        subprocess.run([sys.executable, '-B', str(b.HERE / 'run.py'), 'replay-job', '--family', family, '--match', heldout], check=True)
        _, meta = read_new(family, heldout, manifest)
        verify_replay(family, heldout, manifest, meta)
    print('FOLD COMPLETE', heldout, flush=True)


def load_bundle(heldout, manifest):
    bundle = b.old_bundle(heldout)
    for key, family in KEYS.items():
        output, metadata = read_new(family, heldout, manifest)
        verify_replay(family, heldout, manifest, metadata)
        bundle[key] = output
    return bundle


def fuse(family, bundle):
    control = b.control_probability(bundle)['probability']
    if family == 'control':
        return b.result(control)
    key, mode = family.split('_', 1)
    new = bundle[key]['probability']
    if mode == 'pure':
        return b.result(new)
    if mode == 'base75':
        return b.result(.75 * control + .25 * new)
    if mode == 'replace25':
        return b.result(.5 * bundle['short']['probability'] + .25 * bundle['setseq']['probability'] + .25 * new)
    raise ValueError('Unknown probability family')


def decode(prediction, bundle, observations, stats, mode):
    if mode == 'dead45':
        return b.baseline(prediction, observations, stats)
    return b.consensus_decode(prediction, bundle, observations, stats)


def interval(delta):
    delta = np.asarray(delta, dtype=np.float64)
    draws = np.random.default_rng(20261117).integers(0, 7, size=(5000, 7))
    low, high = np.quantile(delta[draws].mean(1), (.025, .975))
    return dict(mean_delta=float(delta.mean()), lower=float(low), upper=float(high),
                wins=int(np.sum(delta > 0)), ties=int(np.sum(delta == 0)))


def code_identity(heldout, family, decoder, codes):
    return dict(match=heldout, family=family, decoder=decoder,
                sha256=hashlib.sha256(np.asarray(codes, dtype=np.int8).tobytes()).hexdigest())


def generate(manifest, scoring):
    training = {name: b.c.load(name, 'r0', light=True) for name in b.MATCHES}
    records = {family + '/' + decoder: dict(key=family + '/' + decoder, family=family, decoder=decoder, rows=[])
               for family, decoder in settings()}
    diagnostic = b.c.import_path('spell_training_diagnostics', b.c.HERE / 'diagnostics.py') if scoring else None
    hashes = []
    predecessor = json.loads((b.PREVIOUS / 'out/results.json').read_text())
    old_controls = {row['match']: row['sha256'] for row in predecessor['decoded_sha256'] if row['key'] == CONTROL}
    for heldout in b.MATCHES:
        stats = b.fair.statistics([training[name] for name in b.fair.split_names(heldout)])
        data = b.c.load(heldout, 'r0')
        observations = b.observations(data)
        bundle = load_bundle(heldout, manifest)
        for family in families():
            probability = fuse(family, bundle)
            for mode in DECODERS:
                codes = decode(probability, bundle, observations, stats, mode)
                item = code_identity(heldout, family, mode, codes)
                if family == 'control' and mode == 'dead45' and item['sha256'] != old_controls[heldout]:
                    raise ValueError('Frozen predecessor control bytes differ')
                hashes.append(item)
                if scoring:
                    row = dict(match=heldout, training_names=list(b.fair.split_names(heldout)), **b.fair.score(codes, data))
                    row['duration_diagnostic'] = diagnostic.summarize(codes, data)
                    records[family + '/' + mode]['rows'].append(row)
        print('EVAL' if scoring else 'REBUILD', heldout, '20settings', flush=True)
        del data, observations, bundle
        gc.collect()
    return records, hashes


def evaluate():
    manifest = verify()
    if (b.OUT / 'results.json').exists():
        raise FileExistsError('Completed campaign immutable; use a new folder')
    records, hashes = generate(manifest, True)
    control = records[CONTROL]
    for record in records.values():
        record['mean_spellF1'] = float(np.mean([row['spell_f1_mean'] for row in record['rows']]))
        record['mean_macroF1'] = float(np.mean([row['macro_f1'] for row in record['rows']]))
        record['mean_count_error_ratio'] = float(np.mean([row['count_error_ratio'] for row in record['rows']]))
        record['paired_vs_control'] = interval([row['spell_f1_mean'] - old['spell_f1_mean']
                                              for row, old in zip(record['rows'], control['rows'])])
    if abs(control['mean_spellF1'] - .6154575861548992) > 1e-12:
        raise ValueError('Predecessor control metric differs')
    if abs(records['control/consensus']['mean_spellF1'] - .6169215054811713) > 1e-12:
        raise ValueError('Frozen consensus control metric differs')
    primary = records[PRIMARY]
    effect = primary['paired_vs_control']
    promoted = effect['mean_delta'] >= .005 and effect['wins'] >= 5 and effect['lower'] > 0
    ranked = sorted(records.values(), key=lambda row: (-row['mean_spellF1'], -row['mean_macroF1'], row['key']))
    metadata = [read_new(family, name, manifest)[1] for name in b.MATCHES for family in models.FAMILIES]
    report = dict(manifest_sha256=manifest, primary=primary, control=control, records=ranked,
        primary_promoted_for_research=promoted, selected_for_research=PRIMARY if promoted else CONTROL,
        best_raw=ranked[0]['key'], decoded_sha256=hashes, configuration_rows=20, development_evaluations=140,
        new_neural_seed_fits=63, new_optimizer_updates=31500, verified_saved_model_inference_replays=21,
        summed_training_job_seconds=sum(value['seconds'] for value in metadata),
        new_public_evaluation=False, selection_from_public=False, outer_label_meta_fit=False)
    if verify() != manifest:
        raise ValueError('Registry changed during evaluation')
    b.c.write_json(b.OUT / 'results.json', report)
    print('PRIMARY', primary['mean_spellF1'], effect, 'PROMOTED', promoted, flush=True)
    for record in ranked[:8]:
        print(record['key'], record['mean_spellF1'], record['paired_vs_control'], flush=True)


def rebuild():
    manifest = verify()
    report = json.loads((b.OUT / 'results.json').read_text())
    if report['manifest_sha256'] != manifest:
        raise ValueError('Results provenance differs')
    _, hashes = generate(manifest, False)
    if hashes != report['decoded_sha256'] or verify() != manifest:
        raise ValueError('Independent decoded array rebuild differs')
    b.c.write_json(b.OUT / 'verification.json', dict(manifest_sha256=manifest, identical_decoded_arrays=len(hashes),
        verified_saved_model_inference_replays=21, identical_predecessor_control=True,
        new_public_evaluation=False, full_training_refit=False))
    print('REBUILD IDENTICAL', len(hashes), 'SAVED MODEL INFERENCE REPLAYS', 21, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'job', 'replay-job', 'fit-fold', 'all', 'evaluate', 'rebuild'))
    parser.add_argument('--family', choices=models.FAMILIES)
    parser.add_argument('--match', choices=b.MATCHES)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'register': register()
        elif args.action == 'job': fit_job(args.family, args.match)
        elif args.action == 'replay-job': replay_job(args.family, args.match)
        elif args.action == 'fit-fold': fit_fold(args.match)
        elif args.action == 'evaluate': evaluate()
        elif args.action == 'rebuild': rebuild()
        else:
            register()
            for name in b.MATCHES: fit_fold(name)
            evaluate()
