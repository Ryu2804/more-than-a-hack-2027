import gc

import numpy as np
import torch

import audit
import averaging
import batch
import boundary_tasks
import bridge as b
import objectives

FAMILIES = ('balanced', 'focal', 'boundary', 'ranking', 'ema', 'contact')
SEEDS = (0, 1, 2)


def configuration(family, steps):
    if family not in FAMILIES:
        raise ValueError('Unknown event learning family')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    name = 'control_tcn' if family == 'contact' else 'pp_tcn'
    return b.original._resolve_config(dict(b.original._CONFIG_BY_NAME[name],
        steps=steps, seeds=SEEDS, batch=16, crop=384, lr=.002))


def make_model(config, family):
    net = b.original._make_model(config)
    if family == 'boundary':
        net.head = torch.nn.Conv1d(config['ch'], 3, 1)
    return net


def prepare(datasets, family, config):
    if family == 'contact':
        datasets = [batch.attach(data, b.load_context(data['_name'])) for data in datasets]
    raw = b.original._prepare_records(datasets, config)
    for dataset, record in zip(datasets, raw):
        record['blocks'] = b.original._normalize_blocks(dataset['blocks'], len(record['y']))
    records = b.old_sampling.prepare(raw, 'event_tcn', config['crop'])
    for dataset, record in zip(datasets, records):
        if family == 'balanced':
            record['weights'] = b.old_weights.spell_weights(record['y'], dataset['blocks'])
        if family == 'boundary':
            record['boundary_target'], record['boundary_valid'] = boundary_tasks.targets(record['y'], dataset['blocks'])
    return records


def loss(logits, y, views, weights, family, auxiliary):
    if family in ('balanced', 'focal'):
        return objectives.loss(logits, y, views, weights, family)
    if family in ('boundary', 'ranking'):
        return boundary_tasks.loss(logits, y, views, weights, family, auxiliary)
    return b.old_weights.state_loss(logits, y, views, weights, 'event_tcn')


def fit(datasets, family, steps=500):
    datasets = list(datasets)
    names = tuple(data['_name'] for data in datasets)
    missing = set(b.MATCHES) - set(names)
    if len(missing) != 1:
        raise ValueError('Fit must exclude exactly one known IDSSE match')
    heldout = next(iter(missing))
    b.fair.assert_fold(names, heldout)
    config = configuration(family, steps)
    records = prepare(datasets, family, config)
    grouped = b.original._balanced_index(records)
    mean, scale = b.original._player_standardisation(records)
    device = b.original._device()
    nets, seeds = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = make_model(config, family).to(device).train()
        shadow = averaging.create(net) if family == 'ema' else None
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], total_steps=steps) if steps > 1 else None
        averaged_updates = 0
        final_loss = None
        for _ in range(steps):
            x_np, pp_np, ppm_np, y_np, view_np, weight_np, target_np, valid_np = batch.make_batch(records, grouped, mean, scale, config, rng, family)
            x = torch.from_numpy(x_np).to(device)
            pp = torch.from_numpy(pp_np).to(device=device, dtype=torch.float32)
            ppm = torch.from_numpy(ppm_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            view = torch.from_numpy(view_np).to(device)
            state_weights = torch.from_numpy(weight_np).to(device)
            auxiliary = None if target_np is None else (
                torch.from_numpy(target_np).to(device), torch.from_numpy(valid_np).to(device))
            logits = net(x, pp, ppm)
            value = loss(logits, y, view, state_weights, family, auxiliary)
            if not torch.isfinite(value):
                raise ValueError('Training loss became nonfinite')
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            if shadow is not None:
                averaging.update(shadow, net, decay=.99)
                averaged_updates += 1
            if scheduler is not None:
                scheduler.step()
            final_loss = float(value.detach().cpu())
        count = sum(parameter.numel() for parameter in net.parameters())
        persisted = shadow if shadow is not None else net
        persisted.eval().to('cpu')
        nets.append(persisted)
        seeds.append(dict(seed=seed, updates=steps, parameters=count, averaged_updates=averaged_updates,
                          final_training_loss=final_loss))
        del optimizer, scheduler, net, shadow, x, pp, ppm, y, view, state_weights, logits, value, auxiliary
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    audit.training(heldout, family, names, seeds, steps, seeds[0]['parameters'])
    return dict(nets=nets, mean=mean, scale=scale, config=config, family=family,
        training_names=names, heldout=heldout, training=dict(matches=6, names=list(names), device=str(device),
        seeds=list(SEEDS), steps=steps, seed_fits=seeds, input_channels=36 if family == 'contact' else 29,
        event_sampling_probability=.5, match_audit=[dict(match=record['name'], source=record['source'],
        usable_halves=len(record['blocks']), eligible_short_events=len(record['short_events'])) for record in records]))


def predict(fitted, observed):
    audit.observations(fitted, observed)
    if fitted['family'] not in FAMILIES:
        raise ValueError('Unknown fitted family')
    if fitted['family'] == 'contact':
        observed = batch.attach(observed, b.load_context(observed['_name']))
    output = b.result(b.original.predict(fitted, observed)['probability'])
    audit.probability(output['probability'], len(observed['X']))
    return output
