import bridge as b

FAMILIES = ('contact_tcn','memory_tcn','combined_tcn')


def fit(datasets, family, steps=500):
    if family not in FAMILIES:
        raise ValueError('Unknown contextual family')
    names = tuple(data['_name'] for data in datasets)
    if len(names)!=6 or len(set(names))!=6:
        raise ValueError('Contextual fit requires six unique training matches')
    records = [b.attach_controls(data,family) for data in datasets]
    config = dict(b.original._CONFIG_BY_NAME['control_tcn'],steps=steps,seeds=(0,1,2),batch=16,crop=384)
    fitted = b.original.fit(records,config)
    fitted.update(family=family,training_names=names)
    return fitted


def predict(fitted, observations):
    if 'y' in observations or 'y25' in observations:
        raise ValueError('Prediction observations must not contain GT')
    if observations['_name'] in fitted['training_names']:
        raise ValueError('Heldout model was trained on prediction match')
    records = b.attach_controls(observations,fitted['family'])
    output = b.original.predict(fitted,records)
    return b.result(output['probability'])
