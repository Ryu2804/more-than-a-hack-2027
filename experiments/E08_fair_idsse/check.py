import argparse
import json
import subprocess
import sys
import os

import bridge as b
import classic
import neural
from threadpoolctl import threadpool_limits


def child(family):
    os.environ['FARRAS_TRAIN_DEVICE'] = 'cuda' if family in ('pp_tcn','short_tcn') else 'cpu'
    heldout = b.MATCHES[0]
    names = b.split_names(heldout)
    b.assert_fold(names, heldout)
    train = [b.c.load(name, 'r0', with_features=True) for name in names]
    data = b.c.load(heldout, 'r0', with_features=True)
    observations = b.observations(data)
    if family == 'touch_hold':
        codes = classic.touch_predict(observations)
    else:
        fitted = classic.tree_fit(train) if family == 'hgb' else neural.fit(train, family, steps=1)
        output = classic.tree_predict(fitted, observations) if family == 'hgb' else neural.predict(fitted, observations)
        codes = b.decoder.decode(b.probability_result(output['probability']),
            b.c.base.decoder_observations(observations), b.SPEC, b.statistics(train))['codes']
        if output['probability'].shape != (len(data['y']),3):
            raise ValueError('Smoke probability shape mismatch')
    record = dict(family=family, actual_seed_fits=3 if family in neural.FAMILIES else 0,
        steps_per_seed=1 if family in neural.FAMILIES else None, **b.score(codes, data))
    b.c.write_json(b.HERE / 'out' / f'smoke_{family}.json', record)
    print('SMOKE', family, 'PASS', flush=True)


def main():
    families = ('hgb','setseq','pp_tcn','short_tcn','touch_hold')
    records = []
    for family in families:
        subprocess.run([sys.executable, '-B', str(b.HERE / 'check.py'), '--family', family], check=True)
        records.append(json.loads((b.HERE/'out'/f'smoke_{family}.json').read_text()))
    b.c.write_json(b.HERE/'out/smoke_isolated.json', dict(records=records, process_exit_verified=True,
        meaningful_performance_estimate=False, device=str(b.original._device())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('hgb','setseq','pp_tcn','short_tcn','touch_hold'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        child(args.family) if args.family else main()
