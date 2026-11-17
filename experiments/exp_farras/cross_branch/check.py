import gc

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bridge as b
import combine
import models


def main():
    heldout=b.MATCHES[0]
    names=b.fair.split_names(heldout)
    train=[b.c.load(name,'r0',with_features=True) for name in names]
    data=b.c.load(heldout,'r0',with_features=True)
    observation=b.fair.observations(data)
    decoded=b.c.base.decoder_observations(observation)
    stats=b.fair.statistics(train)
    records=[]
    with threadpool_limits(limits=1):
        for family in models.FAMILIES:
            fitted=models.fit(train,family,steps=1)
            output=models.predict(fitted,observation)
            if output['probability'].shape!=(len(data['y']),3):
                raise ValueError('Contextual model output shape mismatch')
            for mode in combine.DECODERS:
                codes=combine.decode(output,decoded,stats,mode)
                records.append(dict(family=family,decoder=mode,**b.fair.score(codes,data)))
            print('SMOKE',family,'PASS',flush=True)
            del fitted,output
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    b.c.write_json(b.OUT/'smoke.json',dict(records=records,new_seed_fits=9,steps=1,
        performance_estimate=False,training_names=list(names),heldout=heldout,backend=str(b.original._device())))


if __name__=='__main__':
    main()
