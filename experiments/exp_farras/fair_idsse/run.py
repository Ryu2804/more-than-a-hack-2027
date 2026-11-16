import argparse
import gc
import json
import time
import subprocess
import sys
import os
from itertools import combinations

import bridge as b
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import classic
import neural

OUT = b.HERE / 'out'
FAMILIES = ('hgb', 'setseq', 'pp_tcn', 'short_tcn', 'touch_hold')


def registry():
    paths = list(b.HERE.glob('*.py')) + [b.HERE / 'PLAN.md']
    for folder in ('wave1', 'wave2', 'wave3', 'wave4', 'wave5'):
        paths += list((b.HERE.parent / folder).glob('*.py'))
    paths += list((b.HERE.parent / 'wave4/reference').glob('*.py'))
    paths += [b.c.REPO / 'evaluation/possession_metrics.py', b.HERE.parent / 'wave3/cache/idsse_audit.json']
    audit = json.loads((b.HERE.parent / 'wave3/cache/idsse_audit.json').read_text())
    for name in b.MATCHES:
        pack = b.c.base.pack_path(name, 'r0')
        if b.c.digest(pack) != audit['matches'][name]['regimes']['r0']['sha256']:
            raise ValueError('Canonical IDSSE input changed: ' + name)
        paths += [pack, b.c.base.CACHE / 'features' / f'{name}_r0.npz']
    return dict(sha256={p.relative_to(b.c.REPO).as_posix(): b.c.digest(p) for p in paths},
        matches=list(b.MATCHES), families=list(FAMILIES), seeds=[0, 1, 2], steps_per_seed=500,
        folds={name:list(b.split_names(name)) for name in b.MATCHES},
        decoder=b.SPEC, regime='r0', pretrained_checkpoints=False, target_public_access=False,
        training_devices={family:(str(b.original._device()) if family in ('pp_tcn','short_tcn') else 'cpu') for family in FAMILIES},
        runtime_versions=dict(python=__import__('platform').python_version(),numpy=np.__version__,torch=torch.__version__,
            sklearn=__import__('sklearn').__version__,scipy=__import__('scipy').__version__),
        device=str(b.original._device()), exposure='Previously exposed matches; new match-exclusive fits, not fresh blind data.')


def register():
    path = OUT / 'preregistered.json'
    if path.exists():
        verify()
    else:
        b.c.write_json(path, registry())
    return b.c.digest(path)


def verify(family=None):
    value = json.loads((OUT / 'preregistered.json').read_text())
    expected = value['device'] if family is None else value['training_devices'][family]
    if expected != str(b.original._device()):
        raise ValueError('Backend changed')
    for relative, expected in value['sha256'].items():
        if b.c.digest(b.c.REPO / relative) != expected:
            raise ValueError('Frozen source/input changed: ' + relative)
    return b.c.digest(OUT / 'preregistered.json')


def job_paths(family, heldout):
    stem = OUT / 'jobs' / f'{family}_{heldout}'
    return stem.with_suffix('.npz'), stem.with_suffix('.json')


def checked_job(family, heldout, data, train, manifest, child=False):
    names = tuple(item['_name'] for item in train)
    b.assert_fold(names, heldout)
    path, meta_path = job_paths(family, heldout)
    identity = dict(manifest_sha256=manifest, family=family, heldout=heldout, training_names=list(names),
        seeds=[0, 1, 2] if family in ('setseq', 'pp_tcn', 'short_tcn') else [],
        steps=500 if family in ('setseq', 'pp_tcn', 'short_tcn') else None)
    if path.exists() or meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['identity'] != identity or meta['sha256'] != b.c.digest(path):
            raise ValueError('Cached job identity/content mismatch')
        with np.load(path, allow_pickle=False) as z:
            return {key:z[key] for key in z.files}, meta
    if not child:
        subprocess.run([sys.executable, '-B', str(b.HERE / 'run.py'), 'job', '--family', family,
            '--heldout', heldout], check=True)
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError('Worker exited without complete prediction evidence')
        return checked_job(family, heldout, data, train, manifest, child=True)
    started = time.monotonic()
    observations = b.observations(data)
    if family == 'touch_hold':
        output = dict(codes=classic.touch_predict(observations))
        fit_names = []
    elif family == 'hgb':
        fitted = classic.tree_fit(train)
        output = classic.tree_predict(fitted, observations)
        if set(fitted['training_names']) != set(names):
            raise ValueError('HGB training membership mismatch')
        fit_names = list(fitted['training_names'])
        del fitted
    else:
        fitted = neural.fit(train, family, steps=500, seeds=(0, 1, 2))
        if set(fitted['training_names']) != set(names):
            raise ValueError('Model training membership mismatch')
        output = neural.predict(fitted, observations)
        fit_names = list(fitted['training_names'])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fitted, OUT / 'jobs' / f'{family}_{heldout}.pt')
        del fitted
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k:v for k,v in output.items() if k in ('probability', 'codes')})
    meta = dict(identity=identity, actual_fit_names=fit_names, seconds=time.monotonic()-started, sha256=b.c.digest(path))
    b.c.write_json(meta_path, meta)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('FIT', family, heldout, round(meta['seconds'], 2), flush=True)
    return output, meta


def decode(output, observations, stats, family, mode):
    if family == 'touch_hold':
        return np.asarray(output['codes'], dtype=np.int8)
    prediction = b.probability_result(output['probability'])
    if mode == 'argmax':
        return prediction['probability'].argmax(1).astype(np.int8)
    return b.decoder.decode(prediction, observations, b.SPEC, stats)['codes']


def compare(rows):
    primary = [r for r in rows if r['mode'] != 'argmax']
    summary = []
    for family in FAMILIES:
        values = [r for r in primary if r['family'] == family]
        summary.append(dict(family=family, mean_spellF1=float(np.mean([r['spell_f1_mean'] for r in values])),
            mean_macroF1=float(np.mean([r['macro_f1'] for r in values])),
            mean_count_error_ratio=float(np.mean([r['count_error_ratio'] for r in values])),
            panel='native_rule' if family == 'touch_hold' else 'shared_decoder'))
    pairs = []
    rng = np.random.default_rng(20261114)
    draws = rng.integers(0, 7, size=(5000, 7))
    for first, second in combinations(FAMILIES, 2):
        a = {r['match']:r['spell_f1_mean'] for r in primary if r['family']==first}
        z = {r['match']:r['spell_f1_mean'] for r in primary if r['family']==second}
        delta = np.array([a[name]-z[name] for name in b.MATCHES])
        low, high = np.quantile(delta[draws].mean(1), [.025,.975])
        pairs.append(dict(first=first, second=second, mean_delta=float(delta.mean()), lower=float(low),upper=float(high),wins=int(np.sum(delta>0)),ties=int(np.sum(delta==0))))
    return summary, pairs


def all_folds():
    if (OUT / 'results.json').exists():
        raise FileExistsError('Benchmark complete; create new protocol instead of overwriting')
    started = time.monotonic()
    manifest = register()
    rows, jobs, hashes = [], [], {}
    for heldout in b.MATCHES:
        names = b.split_names(heldout)
        b.assert_fold(names, heldout)
        train = [b.c.load(name, 'r0', light=True) for name in names]
        data = b.c.load(heldout, 'r0')
        stats = b.statistics(train)
        observations = b.c.base.decoder_observations(b.observations(data))
        for family in FAMILIES:
            output, meta = checked_job(family, heldout, data, train, manifest)
            jobs.append(dict(family=family, heldout=heldout, **meta))
            for mode in (('native_rule',) if family=='touch_hold' else ('shared_decoder','argmax')):
                codes = decode(output, observations, stats, family, mode)
                row = b.score(codes, data)
                row.update(family=family, match=heldout, mode=mode, training_names=list(names), training_stats=stats)
                rows.append(row)
                hashes[family+'/'+heldout+'/'+mode] = __import__('hashlib').sha256(codes.astype(np.int8).tobytes()).hexdigest()
                print('SCORE', family, heldout, mode, row['spell_f1_mean'], flush=True)
        del train, data, observations, output
        gc.collect()
    summary, pairs = compare(rows)
    if verify() != manifest:
        raise ValueError('Manifest changed during fitting')
    result = dict(manifest_sha256=manifest, rows=rows, summary=summary, paired_comparisons=pairs,
        jobs=jobs, decoded_sha256=hashes, seconds=time.monotonic()-started,
        neural_seed_fits=63, neural_optimizer_updates=31500, hgb_fits=7, old_checkpoints_loaded=0,
        same_input_same_labels=True, target_public_access=False, exact_team_submission_reproduction=False,
        inference_label_access=False)
    b.c.write_json(OUT / 'results.json', result)
    print(json.dumps(summary, indent=2), flush=True)


def rebuild():
    manifest = verify()
    result = json.loads((OUT / 'results.json').read_text())
    if result['manifest_sha256'] != manifest:
        raise ValueError('Manifest mismatch')
    reproduced = {}
    for heldout in b.MATCHES:
        names = b.split_names(heldout)
        train = [b.c.load(name,'r0', light=True) for name in names]
        stats = b.statistics(train)
        data = b.c.load(heldout, 'r0')
        observations = b.c.base.decoder_observations(b.observations(data))
        for family in FAMILIES:
            path, meta_path = job_paths(family, heldout)
            meta = json.loads(meta_path.read_text())
            if meta['identity']['manifest_sha256'] != manifest or meta['sha256'] != b.c.digest(path) or meta['identity']['training_names'] != list(names):
                raise ValueError('Prediction provenance mismatch')
            with np.load(path, allow_pickle=False) as z:
                output = {key:z[key] for key in z.files}
            for mode in (('native_rule',) if family=='touch_hold' else ('shared_decoder','argmax')):
                codes = decode(output, observations, stats, family, mode)
                key = family+'/'+heldout+'/'+mode
                reproduced[key] = __import__('hashlib').sha256(codes.astype(np.int8).tobytes()).hexdigest()
        del train, data, observations
    if reproduced != result['decoded_sha256']:
        raise ValueError('Independent decoding differs')
    b.c.write_json(OUT / 'verification.json', dict(manifest_sha256=manifest, identical_decoded_arrays=len(reproduced),
        model_refit=False, prediction_cache_hashes_verified=True))
    print('REBUILD IDENTICAL', len(reproduced), flush=True)


def worker(family, heldout):
    config = json.loads((OUT / 'preregistered.json').read_text())
    os.environ['FARRAS_TRAIN_DEVICE'] = config['training_devices'][family]
    manifest = verify(family)
    names = b.split_names(heldout)
    need_features = family in ('pp_tcn', 'short_tcn')
    train = [b.c.load(name, 'r0', with_features=need_features) for name in names]
    data = b.c.load(heldout, 'r0', with_features=need_features)
    checked_job(family, heldout, data, train, manifest, child=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['all','rebuild','job'])
    parser.add_argument('--family',choices=FAMILIES)
    parser.add_argument('--heldout',choices=b.MATCHES)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == 'job':
            worker(args.family, args.heldout)
        else:
            dict(all=all_folds,rebuild=rebuild)[args.action]()
