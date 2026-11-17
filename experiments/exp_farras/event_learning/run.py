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

import audit
import bridge as b
import models

CONTROL = 'event_control/consensus'
LEGACY = 'legacy/dead45'
PRIMARY = 'balanced_replace/consensus'
DECODERS = ('dead45', 'consensus')
GROUPS = dict(first=models.FAMILIES[:3], second=models.FAMILIES[3:], all=models.FAMILIES)


def families():
    return ('legacy', 'event_control', *(family + '_' + mode for family in models.FAMILIES for mode in ('pure', 'replace', 'split')))


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
    previous = json.loads((b.PREVIOUS / 'out/results.json').read_text())
    proof = json.loads((b.PREVIOUS / 'out/verification.json').read_text())
    if previous['manifest_sha256'] != old_hash or proof['manifest_sha256'] != old_hash or proof['identical_decoded_arrays'] != 140 or proof['verified_saved_model_inference_replays'] != 21:
        raise ValueError('Predecessor evidence incomplete')
    previous_control = next(row for row in previous['records'] if row['key'] == 'event_replace25/consensus')
    if abs(previous_control['mean_spellF1'] - .619562844991005) > 1e-12:
        raise ValueError('Unexpected stronger research control')
    checks = json.loads((b.OUT / 'checks.json').read_text())
    if checks['failures'] or checks['errors'] or {row['family'] for row in checks['smoke']} != set(models.FAMILIES):
        raise ValueError('All semantic tests and six-family real training smoke required')
    counts = {row['family']: row['parameters'] for row in checks['smoke']}
    hashes = dict(old['sha256'])
    paths = list(b.HERE.glob('*.py')) + [b.HERE / 'PLAN.md', b.OUT / 'checks.json', old_path,
        b.PREVIOUS / 'out/results.json', b.PREVIOUS / 'out/verification.json',
        b.PREVIOUS / 'weights.py', b.PREVIOUS / 'sampling.py', b.HERE.parent / 'temporal_methods/consensus.py',
        b.FAIR / 'bridge.py', b.c.HERE / 'training_models.py', b.c.HERE / 'diagnostics.py']
    for heldout in b.MATCHES:
        b.old_bundle(heldout)
        for family in ('short_tcn', 'setseq'):
            cached = b.FAIR / 'out/jobs' / f'{family}_{heldout}.npz'
            paths.extend((cached, cached.with_suffix('.json')))
        cached = b.PREVIOUS / 'out/jobs' / f'event_tcn_{heldout}.npz'
        metadata = json.loads(cached.with_suffix('.json').read_text())
        if metadata['checkpoint_sha256'] != b.c.digest(cached.with_suffix('.pt')):
            raise ValueError('Predecessor event checkpoint changed')
        paths.extend((cached, cached.with_suffix('.json'), cached.with_suffix('.pt'),
                      b.PREVIOUS / 'out/replays' / f'event_tcn_{heldout}.json'))
    for item in paths:
        hashes[item.relative_to(b.c.REPO).as_posix()] = b.c.digest(item)
    registry = dict(sha256=hashes, primary=PRIMARY, control=CONTROL, legacy=LEGACY,
        families=list(families()), models=list(models.FAMILIES), decoders=list(DECODERS),
        matches=list(b.MATCHES), folds={name: list(b.fair.split_names(name)) for name in b.MATCHES},
        seeds=[0, 1, 2], steps=500, batch=16, crop=384, parameters=counts, device=str(b.original._device()),
        new_neural_seed_fits=126, new_optimizer_updates=63000, configuration_rows=40, development_evaluations=280,
        old_manifest_sha256=old_hash, research_control_mean_spellF1=previous_control['mean_spellF1'],
        bootstrap=dict(n=5000, seed=20261118, unit='unique match', multiple_comparison_adjusted=False),
        promotion=dict(min_gain=.005, min_wins=5, lower_bound_positive=True),
        new_public_evaluation=False, outer_label_meta_fit=False,
        exposure='Previously exposed research matches and prior exploratory control. Not fresh blind data. Input/parameter counts differ for contact and boundary.')
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
    if value['identity'] != identity(family, heldout, manifest) or value['actual_fit_names'] != list(b.fair.split_names(heldout)):
        raise ValueError('Fresh model training identity mismatch')
    if value['sha256'] != b.c.digest(probability) or value['checkpoint_sha256'] != b.c.digest(checkpoint):
        raise ValueError('Fresh model/prediction bytes changed')
    registry = json.loads((b.OUT / 'preregistered.json').read_text())
    audit.training(heldout, family, value['actual_fit_names'], value['training']['seed_fits'], 500, registry['parameters'][family])
    expected_averages = 500 if family == 'ema' else 0
    if any(row['averaged_updates'] != expected_averages for row in value['training']['seed_fits']):
        raise ValueError('Unexpected EMA update count')
    with np.load(probability, allow_pickle=False) as archive:
        output = b.result(archive['probability'])
    audit.probability(output['probability'], len(output['probability']))
    return output, value


def fit_job(family, heldout):
    manifest = verify()
    probability, metadata, checkpoint = paths(family, heldout)
    if any(path.exists() for path in (probability, metadata, checkpoint)):
        read_new(family, heldout, manifest)
        return
    started = time.monotonic()
    names = b.fair.split_names(heldout)
    fitted = models.fit([b.c.load(name, 'r0', with_features=True) for name in names], family)
    registry = json.loads((b.OUT / 'preregistered.json').read_text())
    audit.training(heldout, family, fitted['training_names'], fitted['training']['seed_fits'], 500, registry['parameters'][family])
    observed = b.fair.observations(b.c.load(heldout, 'r0', with_features=True))
    output = models.predict(fitted, observed)
    probability.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, checkpoint)
    np.savez_compressed(probability, probability=output['probability'])
    b.c.write_json(metadata, dict(identity=identity(family, heldout, manifest), actual_fit_names=list(fitted['training_names']),
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
    registry = json.loads((b.OUT / 'preregistered.json').read_text())
    if fitted['family'] != family:
        raise ValueError('Checkpoint family mismatch')
    audit.training(heldout, family, fitted['training_names'], fitted['training']['seed_fits'], 500, registry['parameters'][family])
    observed = b.fair.observations(b.c.load(heldout, 'r0', with_features=True))
    actual = models.predict(fitted, observed)['probability']
    with np.load(paths(family, heldout)[0], allow_pickle=False) as archive:
        audit.replay(actual, archive['probability'])
    b.c.write_json(replay_path(family, heldout), dict(identity=identity(family, heldout, manifest),
        checkpoint_sha256=metadata['checkpoint_sha256'], probability_sha256=metadata['sha256'], identical_full_probabilities=True))
    print('INFERENCE IDENTICAL', family, heldout, flush=True)


def fit_fold(heldout, group='all'):
    manifest = verify()
    for family in GROUPS[group]:
        probability, metadata, checkpoint = paths(family, heldout)
        if any(path.exists() for path in (probability, metadata, checkpoint)):
            read_new(family, heldout, manifest)
        else:
            subprocess.run([sys.executable, '-B', str(b.HERE / 'run.py'), 'job', '--family', family, '--match', heldout], check=True)
        subprocess.run([sys.executable, '-B', str(b.HERE / 'run.py'), 'replay-job', '--family', family, '--match', heldout], check=True)
        _, value = read_new(family, heldout, manifest)
        verify_replay(family, heldout, manifest, value)
    print('GROUP COMPLETE', heldout, group, flush=True)


def load_bundle(heldout, manifest):
    bundle = b.old_bundle(heldout)
    for family in models.FAMILIES:
        output, metadata = read_new(family, heldout, manifest)
        verify_replay(family, heldout, manifest, metadata)
        bundle[family] = output
    return bundle


def fuse(family, bundle):
    if family == 'legacy':
        return b.legacy_probability(bundle)
    if family == 'event_control':
        return b.research_probability(bundle)
    key, mode = family.split('_', 1)
    new = bundle[key]['probability']
    if mode == 'pure':
        return b.result(new)
    base = .5 * bundle['short']['probability'] + .25 * bundle['setseq']['probability']
    if mode == 'replace':
        return b.result(base + .25 * new)
    if mode == 'split':
        return b.result(base + .125 * bundle['old_event']['probability'] + .125 * new)
    raise ValueError('Unknown fixed fusion family')


def interval(delta):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (7,) or not np.isfinite(delta).all():
        raise ValueError('Paired comparison needs seven finite match differences')
    draws = np.random.default_rng(20261118).integers(0, 7, size=(5000, 7))
    low, high = np.quantile(delta[draws].mean(1), (.025, .975))
    return dict(mean_delta=float(delta.mean()), lower=float(low), upper=float(high),
                wins=int(np.sum(delta > 0)), ties=int(np.sum(delta == 0)))


def code_identity(heldout, family, decoder, codes):
    return dict(match=heldout, family=family, decoder=decoder,
                sha256=hashlib.sha256(np.asarray(codes, dtype=np.int8).tobytes()).hexdigest())


def generate(manifest, scoring):
    training = {name: b.c.load(name, 'r0', light=True) for name in b.MATCHES}
    records = {family + '/' + mode: dict(key=family + '/' + mode, family=family, decoder=mode, rows=[])
               for family in families() for mode in DECODERS}
    diagnostic = b.c.import_path('event_learning_diagnostics', b.c.HERE / 'diagnostics.py') if scoring else None
    predecessor = json.loads((b.PREVIOUS / 'out/results.json').read_text())
    old_hashes = {(row['match'], row['family'], row['decoder']): row['sha256'] for row in predecessor['decoded_sha256']}
    hashes = []
    for heldout in b.MATCHES:
        stats = b.fair.statistics([training[name] for name in b.fair.split_names(heldout)])
        data = b.c.load(heldout, 'r0')
        observed = b.observations(data)
        bundle = load_bundle(heldout, manifest)
        for family in families():
            probability = fuse(family, bundle)
            for mode in DECODERS:
                codes = b.decode(probability, bundle, observed, stats, mode)
                item = code_identity(heldout, family, mode, codes)
                if family in ('legacy', 'event_control'):
                    old_family = 'control' if family == 'legacy' else 'event_replace25'
                    if item['sha256'] != old_hashes[heldout, old_family, mode]:
                        raise ValueError('Predecessor control decoded bytes differ')
                hashes.append(item)
                if scoring:
                    row = dict(match=heldout, training_names=list(b.fair.split_names(heldout)), **b.fair.score(codes, data))
                    row['duration_diagnostic'] = diagnostic.summarize(codes, data)
                    records[family + '/' + mode]['rows'].append(row)
        print('EVAL' if scoring else 'REBUILD', heldout, '40settings', flush=True)
        del data, observed, bundle
        gc.collect()
    return records, hashes


def evaluate():
    manifest = verify()
    if (b.OUT / 'results.json').exists():
        raise FileExistsError('Completed campaign immutable; use a new folder')
    records, hashes = generate(manifest, True)
    control, legacy = records[CONTROL], records[LEGACY]
    for record in records.values():
        record['mean_spellF1'] = float(np.mean([row['spell_f1_mean'] for row in record['rows']]))
        record['mean_macroF1'] = float(np.mean([row['macro_f1'] for row in record['rows']]))
        record['mean_count_error_ratio'] = float(np.mean([row['count_error_ratio'] for row in record['rows']]))
        for label, baseline in (('paired_vs_control', control), ('paired_vs_legacy', legacy)):
            record[label] = interval([row['spell_f1_mean'] - old['spell_f1_mean']
                                      for row, old in zip(record['rows'], baseline['rows'])])
    if abs(control['mean_spellF1'] - .619562844991005) > 1e-12 or abs(legacy['mean_spellF1'] - .6154575861548992) > 1e-12:
        raise ValueError('Predecessor control metrics differ')
    primary = records[PRIMARY]
    effect = primary['paired_vs_control']
    promoted = effect['mean_delta'] >= .005 and effect['wins'] >= 5 and effect['lower'] > 0
    ranked = sorted(records.values(), key=lambda row: (-row['mean_spellF1'], -row['mean_macroF1'], row['key']))
    metadata = [read_new(family, name, manifest)[1] for name in b.MATCHES for family in models.FAMILIES]
    report = dict(manifest_sha256=manifest, primary=primary, control=control, legacy=legacy, records=ranked,
        primary_promoted_for_research=promoted, selected_for_research=PRIMARY if promoted else CONTROL,
        best_raw=ranked[0]['key'], decoded_sha256=hashes, configuration_rows=40, development_evaluations=280,
        new_neural_seed_fits=126, new_optimizer_updates=63000, verified_saved_model_inference_replays=42,
        summed_training_job_seconds=sum(value['seconds'] for value in metadata),
        new_public_evaluation=False, selection_from_public=False, outer_label_meta_fit=False)
    if verify() != manifest:
        raise ValueError('Registry changed during evaluation')
    b.c.write_json(b.OUT / 'results.json', report)
    print('PRIMARY', primary['mean_spellF1'], effect, 'PROMOTED', promoted, flush=True)
    for record in ranked[:10]:
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
        verified_saved_model_inference_replays=42, identical_predecessor_controls=True,
        new_public_evaluation=False, full_training_refit=False))
    print('REBUILD IDENTICAL', len(hashes), 'SAVED MODEL INFERENCE REPLAYS', 42, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'job', 'replay-job', 'fit-fold', 'all', 'evaluate', 'rebuild'))
    parser.add_argument('--family', choices=models.FAMILIES)
    parser.add_argument('--match', choices=b.MATCHES)
    parser.add_argument('--group', choices=tuple(GROUPS), default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'register': register()
        elif args.action == 'job': fit_job(args.family, args.match)
        elif args.action == 'replay-job': replay_job(args.family, args.match)
        elif args.action == 'fit-fold': fit_fold(args.match, args.group)
        elif args.action == 'evaluate': evaluate()
        elif args.action == 'rebuild': rebuild()
        else:
            register()
            for name in b.MATCHES: fit_fold(name)
            evaluate()
