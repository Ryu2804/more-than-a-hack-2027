import gc
import json
from dataclasses import replace

import numpy as np

import bridge as b
import context


def main():
    if (b.OUT / 'preregistered.json').exists():
        raise FileExistsError('Inputs frozen; do not regenerate contextual features')
    audit_path = b.HERE.parent / 'wave3/cache/idsse_audit.json'
    original_audit = json.loads(audit_path.read_text())
    records = []
    for name in b.MATCHES:
        clean_path = b.reference.ROOT / 'cache' / f'{name}_clean.npz'
        if not clean_path.is_file():
            raise FileNotFoundError('Missing pinned local25fps cache; no download fallback')
        prepared_path = b.c.base.pack_path(name, 'r0')
        entry = original_audit['matches'][name]['regimes']['r0']
        if b.c.digest(prepared_path) != entry['sha256']:
            raise ValueError('Prepared canonical input hash mismatch')
        clean = replace(b.reference.read_match(clean_path), labels=None)
        prepared = b.fair.observations(b.c.load(name,'r0'))
        output = context.build(clean, prepared, entry['seed'])
        control = np.asarray(output['control'])
        if control.shape != (2,len(prepared['X']),7) or not np.isfinite(control).all():
            raise ValueError('Context shape/finite contract failed')
        codes = np.asarray(output['native_codes'])
        if codes.shape != (len(prepared['X']),) or not np.isin(codes,(0,1,2)).all():
            raise ValueError('Invalid native rule output')
        b.CACHE.mkdir(parents=True,exist_ok=True)
        path = b.CACHE / f'{name}.npz'
        np.savez_compressed(path,control=control,native_codes=codes)
        records.append(dict(output['audit'],match=name,seed=entry['seed'],source25_sha256=b.c.digest(clean_path),
            prepared_sha256=b.c.digest(prepared_path),context_sha256=b.c.digest(path)))
        print('PREPARED',name,'samples',len(codes),flush=True)
        del clean,prepared,output,control,codes
        gc.collect()
    b.c.write_json(b.OUT/'preparation.json',dict(records=records,source_audit_sha256=b.c.digest(audit_path),
        context_source_sha256=b.c.digest(b.HERE/'context.py'),label_access=False))


if __name__ == '__main__':
    main()
