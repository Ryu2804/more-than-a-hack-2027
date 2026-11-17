import argparse
import gc
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import bridge as b
import boundaries
import consensus
import duration

METHODS = {'duration': duration, 'consensus': consensus, 'boundaries': boundaries}
CONTROL = 'control/dead45'


def settings():
    rows = [dict(key=CONTROL, method='control', config=dict(name='dead45'))]
    for method, module in METHODS.items():
        rows.extend(dict(key=method + '/' + config['name'], method=method, config=dict(config))
                    for config in module.CONFIGS)
    if len(rows) != 19 or len({row['key'] for row in rows}) != 19:
        raise ValueError('Protocol requires 19 unique settings')
    return rows


def primary_key():
    return next(row['key'] for row in settings() if row['method'] == 'duration'
                and row['config']['strength'] == .5 and row['config']['switch_bias'] == 0)


def verify():
    registry = json.loads((b.OUT / 'preregistered.json').read_text())
    if registry['settings'] != settings() or registry['primary'] != primary_key():
        raise ValueError('Preregistered method configurations changed')
    for relative, expected in registry['sha256'].items():
        if b.c.digest(b.c.REPO / relative) != expected:
            raise ValueError('Frozen source/input changed: ' + relative)
    return b.c.digest(b.OUT / 'preregistered.json')


def register():
    path = b.OUT / 'preregistered.json'
    if path.exists():
        return verify()
    old_path = b.PREVIOUS / 'out/preregistered.json'
    old = json.loads(old_path.read_text())
    old_hash = b.c.digest(old_path)
    report_path = b.PREVIOUS / 'out/results.json'
    report = json.loads(report_path.read_text())
    proof_path = b.PREVIOUS / 'out/verification.json'
    proof = json.loads(proof_path.read_text())
    if report['manifest_sha256'] != old_hash or proof['manifest_sha256'] != old_hash or proof['identical_decoded_arrays'] != 203:
        raise ValueError('Predecessor needs complete matching verification')
    control = next(row for row in report['records']
                   if row['family'] == 'short_setseq25' and row['decoder'] == 'dead45')
    if abs(control['mean_spellF1'] - .6154575861548992) > 1e-12:
        raise ValueError('Unexpected predecessor control')
    hashes = dict(old['sha256'])
    paths = list(b.HERE.glob('*.py')) + [b.HERE / 'PLAN.md', old_path, report_path, proof_path,
                                       b.FAIR / 'bridge.py', b.c.HERE / 'diagnostics.py', b.OUT / 'checks.json']
    for name in b.MATCHES:
        b.fair.assert_fold(b.fair.split_names(name), name)
        for family in ('short_tcn', 'setseq'):
            b.read_prediction(family, name)
            cached = b.FAIR / 'out/jobs' / f'{family}_{name}.npz'
            paths.extend((cached, cached.with_suffix('.json')))
        paths.extend((b.PREVIOUS / 'cache' / f'{name}.npz', b.c.base.pack_path(name, 'r0')))
    for item in paths:
        hashes[item.relative_to(b.c.REPO).as_posix()] = b.c.digest(item)
    value = dict(sha256=hashes, settings=settings(), primary=primary_key(), control=CONTROL,
                 matches=list(b.MATCHES), folds={name: list(b.fair.split_names(name)) for name in b.MATCHES},
                 old_manifest_sha256=old_hash, baseline_mean_spellF1=control['mean_spellF1'],
                 probability_weights=dict(short=.75, setseq=.25), new_neural_seed_fits=0,
                 training_prior_cap=80, training_prior_fits=7,
                 bootstrap=dict(n=5000, seed=20261116, unit='unique match', multiple_comparison_adjusted=False),
                 promotion=dict(min_gain=.005, min_wins=5, lower_bound_positive=True),
                 new_public_evaluation=False, outer_label_meta_fit=False,
                 exposure='Repeatedly exposed matches; prior exploratory winner used as control. Not fresh blind data.')
    b.c.write_json(path, value)
    return verify()


def interval(delta):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (7,) or not np.isfinite(delta).all():
        raise ValueError('Paired comparison requires seven finite match deltas')
    draws = np.random.default_rng(20261116).integers(0, 7, size=(5000, 7))
    low, high = np.quantile(delta[draws].mean(1), (.025, .975))
    return dict(mean_delta=float(delta.mean()), lower=float(low), upper=float(high),
                wins=int(np.sum(delta > 0)), ties=int(np.sum(delta == 0)))


def prior_digest(prior):
    encoded = json.dumps(b.jsonable(prior), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_fold(heldout, training_data):
    names = b.fair.split_names(heldout)
    b.fair.assert_fold(names, heldout)
    train = [training_data[name] for name in names]
    stats = b.fair.statistics(train)
    prior = duration.fit(train, cap=80)
    if tuple(prior['training_names']) != names:
        raise ValueError('Duration prior fitted on incorrect training membership')
    data = b.c.load(heldout, 'r0')
    observations = b.observations(data)
    bundle = b.bundle(heldout)
    prediction = b.fuse(bundle)
    if len(prediction['probability']) != len(data['y']):
        raise ValueError('Cached predictions are on another timeline')
    control = b.baseline(prediction, observations, stats)
    return data, observations, bundle, prediction, stats, prior, control


def decoded(spec, prediction, bundle, observations, stats, prior, control):
    if spec['method'] == 'control':
        return control
    if spec['method'] == 'duration':
        return duration.decode(prediction, observations, prior, spec['config'])
    if spec['method'] == 'consensus':
        return consensus.decode(prediction, bundle, observations, stats, spec['config'])
    return boundaries.decode(prediction, observations, stats, spec['config'], baseline_codes=control)


def code_identity(heldout, key, codes):
    return dict(match=heldout, key=key,
                sha256=hashlib.sha256(np.asarray(codes, dtype=np.int8).tobytes()).hexdigest())


def evaluate():
    manifest = verify()
    if (b.OUT / 'results.json').exists():
        raise FileExistsError('Completed experiment immutable; use another campaign')
    started = time.monotonic()
    specifications = settings()
    records = {spec['key']: dict(**spec, rows=[]) for spec in specifications}
    training_data = {name: b.c.load(name, 'r0', light=True) for name in b.MATCHES}
    diagnostic = b.c.import_path('temporal_metric_diagnostics', b.c.HERE / 'diagnostics.py')
    previous = json.loads((b.PREVIOUS / 'out/results.json').read_text())
    expected_control = {row['match']: row['sha256'] for row in previous['decoded_sha256']
                        if row['family'] == 'short_setseq25' and row['decoder'] == 'dead45'}
    hashes, priors = [], []
    for heldout in b.MATCHES:
        data, observations, bundle, prediction, stats, prior, control = load_fold(heldout, training_data)
        if code_identity(heldout, CONTROL, control)['sha256'] != expected_control[heldout]:
            raise ValueError('Control differs from predecessor decoded bytes')
        prior_path = b.OUT / 'priors' / f'{heldout}.json'
        b.c.write_json(prior_path, b.jsonable(prior))
        priors.append(dict(match=heldout, training_names=list(b.fair.split_names(heldout)),
                           canonical_sha256=prior_digest(prior), file_sha256=b.c.digest(prior_path), stats=stats))
        for spec in specifications:
            codes = decoded(spec, prediction, bundle, observations, stats, prior, control)
            row = dict(match=heldout, training_names=list(b.fair.split_names(heldout)), **b.fair.score(codes, data))
            row['duration_diagnostic'] = diagnostic.summarize(codes, data)
            records[spec['key']]['rows'].append(row)
            hashes.append(code_identity(heldout, spec['key'], codes))
        print('EVAL', heldout, '19 settings', 'seconds', round(time.monotonic() - started, 1), flush=True)
        del data, observations, bundle, prediction, prior, control
        gc.collect()
    control_record = records[CONTROL]
    for record in records.values():
        record['mean_spellF1'] = float(np.mean([row['spell_f1_mean'] for row in record['rows']]))
        record['mean_macroF1'] = float(np.mean([row['macro_f1'] for row in record['rows']]))
        record['mean_count_error_ratio'] = float(np.mean([row['count_error_ratio'] for row in record['rows']]))
        record['paired_vs_control'] = interval([row['spell_f1_mean'] - old['spell_f1_mean']
                                               for row, old in zip(record['rows'], control_record['rows'])])
    if abs(control_record['mean_spellF1'] - .6154575861548992) > 1e-12:
        raise ValueError('Frozen control metric not reproduced')
    primary = records[primary_key()]
    effect = primary['paired_vs_control']
    promoted = effect['mean_delta'] >= .005 and effect['wins'] >= 5 and effect['lower'] > 0
    ranked = sorted(records.values(), key=lambda row: (-row['mean_spellF1'], -row['mean_macroF1'], row['key']))
    report = dict(manifest_sha256=manifest, primary=primary, control=control_record,
                  primary_promoted_for_research=promoted, selected_for_research=primary['key'] if promoted else CONTROL,
                  best_raw=ranked[0]['key'], records=ranked, decoded_sha256=hashes, prior_fits=priors,
                  configuration_rows=19, development_evaluations=133, new_neural_seed_fits=0,
                  new_duration_prior_fits=7, seconds=time.monotonic() - started,
                  new_public_evaluation=False, selection_from_public=False, outer_label_meta_fit=False)
    if verify() != manifest:
        raise ValueError('Registry changed during evaluation')
    b.c.write_json(b.OUT / 'results.json', report)
    print('PRIMARY', primary['key'], primary['mean_spellF1'], 'PROMOTED', promoted, flush=True)
    print('CONTROL', control_record['mean_spellF1'], flush=True)
    for record in ranked[:8]:
        print(record['key'], record['mean_spellF1'], record['paired_vs_control'], flush=True)


def rebuild():
    manifest = verify()
    report = json.loads((b.OUT / 'results.json').read_text())
    if report['manifest_sha256'] != manifest:
        raise ValueError('Result provenance mismatch')
    training_data = {name: b.c.load(name, 'r0', light=True) for name in b.MATCHES}
    hashes = []
    expected_priors = {row['match']: row for row in report['prior_fits']}
    for heldout in b.MATCHES:
        data, observations, bundle, prediction, stats, prior, control = load_fold(heldout, training_data)
        expected = expected_priors[heldout]
        if prior_digest(prior) != expected['canonical_sha256'] or b.c.digest(b.OUT / 'priors' / f'{heldout}.json') != expected['file_sha256']:
            raise ValueError('Independent training-prior rebuild mismatch')
        for spec in settings():
            codes = decoded(spec, prediction, bundle, observations, stats, prior, control)
            hashes.append(code_identity(heldout, spec['key'], codes))
        print('REBUILD', heldout, flush=True)
        del data, observations, bundle, prediction, prior, control
        gc.collect()
    if hashes != report['decoded_sha256']:
        raise ValueError('Independent decoder rebuild mismatch')
    if verify() != manifest:
        raise ValueError('Registry changed during rebuild')
    proof = dict(manifest_sha256=manifest, identical_decoded_arrays=len(hashes), identical_refitted_duration_priors=7,
                 predecessor_control_identical=True, old_weights_loaded=False, new_public_evaluation=False)
    b.c.write_json(b.OUT / 'verification.json', proof)
    print('REBUILD IDENTICAL', len(hashes), 'PRIORS', 7, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'evaluate', 'all', 'rebuild'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'register':
            register()
        elif args.action == 'evaluate':
            evaluate()
        elif args.action == 'rebuild':
            rebuild()
        else:
            register()
            evaluate()
