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
import combine
import models

PRIMARY = ('triple_combined','hysteresis')
CONTROL = ('short','dead45')


def verify():
    registry=json.loads((b.OUT/'preregistered.json').read_text())
    for relative,expected in registry['sha256'].items():
        if b.c.digest(b.c.REPO/relative)!=expected:
            raise ValueError('Frozen source/input changed: '+relative)
    if str(b.original._device())!=registry['training_device']:
        raise ValueError('Training backend changed')
    return b.c.digest(b.OUT/'preregistered.json')


def register():
    path=b.OUT/'preregistered.json'
    if path.exists():
        return verify()
    old=json.loads((b.PREVIOUS/'out/preregistered.json').read_text())
    results=json.loads((b.PREVIOUS/'out/results.json').read_text())
    proof=json.loads((b.PREVIOUS/'out/verification.json').read_text())
    old_hash=b.c.digest(b.PREVIOUS/'out/preregistered.json')
    if results['manifest_sha256']!=old_hash or proof['manifest_sha256']!=old_hash or proof['identical_decoded_arrays']!=63:
        raise ValueError('Previous fair benchmark lacks matching full verification')
    hashes=dict(old['sha256'])
    prep=json.loads((b.OUT/'preparation.json').read_text())
    if prep['context_source_sha256']!=b.c.digest(b.HERE/'context.py'):
        raise ValueError('Prepared feature code changed')
    paths=list(b.HERE.glob('*.py'))+[b.HERE/'PLAN.md',b.OUT/'preparation.json',
        b.PREVIOUS/'out/preregistered.json',b.PREVIOUS/'out/results.json',b.PREVIOUS/'out/verification.json',
        b.HERE.parent/'wave3/reference/jeremy_data.py']
    for record in prep['records']:
        name=record['match']
        clean=b.reference.ROOT/'cache'/f'{name}_clean.npz'
        context=b.CACHE/f'{name}.npz'
        if b.c.digest(clean)!=record['source25_sha256'] or b.c.digest(context)!=record['context_sha256'] or b.c.digest(b.c.base.pack_path(name,'r0'))!=record['prepared_sha256']:
            raise ValueError('Context input provenance mismatch')
        paths += [clean,context]
    if {r['match'] for r in prep['records']}!=set(b.MATCHES):
        raise ValueError('Preparation must cover all seven matches')
    for name in b.MATCHES:
        for family in ('short_tcn','pp_tcn','setseq'):
            b.read_old_prediction(family,name)
            path_job=b.PREVIOUS/'out/jobs'/f'{family}_{name}.npz'
            paths += [path_job,path_job.with_suffix('.json')]
    for item in paths:
        hashes[item.relative_to(b.c.REPO).as_posix()]=b.c.digest(item)
    for relative,expected in hashes.items():
        if b.c.digest(b.c.REPO/relative)!=expected:
            raise ValueError('Inherited frozen artifact changed: '+relative)
    value=dict(sha256=hashes,primary=dict(family=PRIMARY[0],decoder=PRIMARY[1]),
        control=dict(family=CONTROL[0],decoder=CONTROL[1]),families=list(combine.FAMILIES),decoders=list(combine.DECODERS),
        matches=list(b.MATCHES),folds={name:list(b.fair.split_names(name)) for name in b.MATCHES},
        models=list(models.FAMILIES),seeds=[0,1,2],steps=500,batch=16,crop=384,
        training_device=str(b.original._device()),source_refs=dict(jeremy='ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1',
        ryu='5eaf3e595e91fa0502aac48037d0cae8c356ec6f',rasya='a13249724b994728b5686d01bc48e6647cd1e40e',
        consolidation='264efa67e1a2e74cce22fea81ed784acf159b258',farras='87f3236d4ed875f4cc868d9f750a0776a7779333',main='ef22d025b0c02bcdbcec53df34973ece0f53a701'),
        old_manifest_sha256=old_hash,new_public_evaluation=False,
        bootstrap=dict(n=5000,seed=20261115,unit='unique match',multiple_comparison_adjusted=False),
        exposure='Previously exposed matches and hypotheses. New disjoint fits, not fresh blind data.')
    b.c.write_json(b.OUT/'preregistered.json',value)
    return verify()


def job_paths(family,heldout):
    path=b.OUT/'jobs'/f'{family}_{heldout}.npz'
    return path,path.with_suffix('.json')


def job_identity(family,heldout,manifest):
    return dict(family=family,heldout=heldout,training_names=list(b.fair.split_names(heldout)),
        seeds=[0,1,2],steps=500,manifest_sha256=manifest)


def read_new(family,heldout,manifest):
    path,metadata=job_paths(family,heldout)
    meta=json.loads(metadata.read_text())
    if meta['identity']!=job_identity(family,heldout,manifest) or meta['sha256']!=b.c.digest(path):
        raise ValueError('Contextual prediction cache identity mismatch')
    if meta['actual_fit_names']!=list(b.fair.split_names(heldout)):
        raise ValueError('Incorrect actual training membership')
    with np.load(path,allow_pickle=False) as archive:
        output=b.result(archive['probability'])
    return output,meta


def fit_job(family,heldout):
    manifest=verify()
    path,metadata=job_paths(family,heldout)
    if path.exists() or metadata.exists():
        read_new(family,heldout,manifest)
        return
    names=b.fair.split_names(heldout)
    b.validate_names(names,heldout)
    started=time.monotonic()
    train=[b.c.load(name,'r0',with_features=True) for name in names]
    fitted=models.fit(train,family)
    if fitted['training_names']!=tuple(names) or fitted['training']['matches']!=6:
        raise ValueError('Fresh model was not trained on six other matches')
    del train
    gc.collect()
    data=b.fair.observations(b.c.load(heldout,'r0',with_features=True))
    output=models.predict(fitted,data)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(fitted,path.with_suffix('.pt'))
    np.savez_compressed(path,probability=output['probability'])
    b.c.write_json(metadata,dict(identity=job_identity(family,heldout,manifest),
        sha256=b.c.digest(path),actual_fit_names=list(fitted['training_names']),training=fitted['training'],
        seconds=time.monotonic()-started))
    print('FIT',family,heldout,'seeds3/steps500','seconds',round(time.monotonic()-started,2),flush=True)


def fit_fold(heldout):
    manifest=verify()
    for family in models.FAMILIES:
        path,meta=job_paths(family,heldout)
        if path.exists() or meta.exists():
            read_new(family,heldout,manifest)
            continue
        subprocess.run([sys.executable,'-B',str(b.HERE/'run.py'),'job','--family',family,'--match',heldout],check=True)
        read_new(family,heldout,manifest)
    print('FOLD COMPLETE',heldout,flush=True)


def load_bundle(heldout,manifest):
    bundle={name:b.read_old_prediction(family,heldout) for name,family in
        (('short','short_tcn'),('pp','pp_tcn'),('setseq','setseq'))}
    for key in ('contact','memory','combined'):
        bundle[key]=read_new(key+'_tcn',heldout,manifest)[0]
    return bundle


def interval(delta):
    delta=np.asarray(delta,dtype=np.float64)
    draws=np.random.default_rng(20261115).integers(0,7,size=(5000,7))
    low,high=np.quantile(delta[draws].mean(1),(.025,.975))
    return dict(mean_delta=float(delta.mean()),lower=float(low),upper=float(high),wins=int(np.sum(delta>0)),ties=int(np.sum(delta==0)))


def evaluate():
    manifest=verify()
    if (b.OUT/'results.json').exists():
        raise FileExistsError('Completed campaign immutable; start another experiment')
    records={(family,decoder):dict(family=family,decoder=decoder,rows=[]) for family in combine.FAMILIES for decoder in combine.DECODERS}
    hashes,native=[],[]
    for heldout in b.MATCHES:
        names=b.fair.split_names(heldout)
        train=[b.c.load(name,'r0',light=True) for name in names]
        stats=b.fair.statistics(train)
        data=b.c.load(heldout,'r0')
        observations=b.c.base.decoder_observations(b.fair.observations(data))
        bundle=load_bundle(heldout,manifest)
        for family in combine.FAMILIES:
            probability=combine.fuse(family,bundle,observations)
            for mode in combine.DECODERS:
                codes=combine.decode(probability,observations,stats,mode)
                row=b.fair.score(codes,data)
                row.update(match=heldout,training_names=list(names))
                records[family,mode]['rows'].append(row)
                hashes.append(dict(match=heldout,family=family,decoder=mode,sha256=hashlib.sha256(codes.astype(np.int8).tobytes()).hexdigest()))
        with np.load(b.CACHE/f'{heldout}.npz',allow_pickle=False) as archive:
            codes=archive['native_codes']
        native.append(dict(match=heldout,**b.fair.score(codes,data)))
        hashes.append(dict(match=heldout,family='native25',decoder='native',sha256=hashlib.sha256(codes.astype(np.int8).tobytes()).hexdigest()))
        print('EVAL',heldout,'28settings',flush=True)
        del train,data,observations,bundle
        gc.collect()
    control=records[CONTROL]
    for spec in records.values():
        spec['mean_spellF1']=float(np.mean([row['spell_f1_mean'] for row in spec['rows']]))
        spec['mean_macroF1']=float(np.mean([row['macro_f1'] for row in spec['rows']]))
        spec['paired_vs_control']=interval([a['spell_f1_mean']-z['spell_f1_mean'] for a,z in zip(spec['rows'],control['rows'])])
    if abs(control['mean_spellF1']-0.5957554001098855)>1e-12:
        raise ValueError('Original fair benchmark control not reproduced')
    primary=records[PRIMARY]
    effect=primary['paired_vs_control']
    promoted=effect['mean_delta']>=.005 and effect['wins']>=5 and effect['lower']>0
    ranked=sorted(records.values(),key=lambda row:(-row['mean_spellF1'],-row['mean_macroF1'],row['family'],row['decoder']))
    report=dict(primary=primary,control=control,primary_promoted_for_research=promoted,
        best_raw=dict(family=ranked[0]['family'],decoder=ranked[0]['decoder']),records=ranked,native_rule=native,
        manifest_sha256=manifest,decoded_sha256=hashes,configuration_rows=28,development_evaluations=196,
        new_neural_seed_fits=63,new_optimizer_updates=31500,new_public_evaluation=False,
        selection_from_public=False,outer_label_meta_fit=False)
    if verify()!=manifest:
        raise ValueError('Registry changed during evaluation')
    b.c.write_json(b.OUT/'results.json',report)
    print('PRIMARY',primary['mean_spellF1'],effect,'PROMOTED',promoted,flush=True)
    for spec in ranked[:8]:
        print(spec['family'],spec['decoder'],spec['mean_spellF1'],flush=True)


def rebuild():
    manifest=verify()
    report=json.loads((b.OUT/'results.json').read_text())
    if report['manifest_sha256']!=manifest:
        raise ValueError('Different manifest in results')
    hashes=[]
    for heldout in b.MATCHES:
        names=b.fair.split_names(heldout)
        train=[b.c.load(name,'r0',light=True) for name in names]
        stats=b.fair.statistics(train)
        data=b.c.load(heldout,'r0')
        observations=b.c.base.decoder_observations(b.fair.observations(data))
        bundle=load_bundle(heldout,manifest)
        for family in combine.FAMILIES:
            probability=combine.fuse(family,bundle,observations)
            for mode in combine.DECODERS:
                codes=combine.decode(probability,observations,stats,mode)
                hashes.append(dict(match=heldout,family=family,decoder=mode,sha256=hashlib.sha256(codes.astype(np.int8).tobytes()).hexdigest()))
        with np.load(b.CACHE/f'{heldout}.npz',allow_pickle=False) as archive:
            codes=archive['native_codes']
        hashes.append(dict(match=heldout,family='native25',decoder='native',sha256=hashlib.sha256(codes.astype(np.int8).tobytes()).hexdigest()))
    if hashes!=report['decoded_sha256']:
        raise ValueError('Independent decoder rebuild mismatch')
    b.c.write_json(b.OUT/'verification.json',dict(manifest_sha256=manifest,identical_decoded_arrays=len(hashes),
        original_control_identical=True,model_refit=False,old_weights_loaded=False))
    print('REBUILD IDENTICAL',len(hashes),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('register','job','fit-fold','all','evaluate','rebuild'))
    parser.add_argument('--family',choices=models.FAMILIES)
    parser.add_argument('--match',choices=b.MATCHES)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action=='register': register()
        elif args.action=='job': fit_job(args.family,args.match)
        elif args.action=='fit-fold': fit_fold(args.match)
        elif args.action=='evaluate': evaluate()
        elif args.action=='rebuild': rebuild()
        else:
            register()
            for name in b.MATCHES: fit_fold(name)
            evaluate()
