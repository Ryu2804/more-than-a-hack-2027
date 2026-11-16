import gc

import numpy as np
import torch

import bridge as b
import consistency
import dropout
import style

FAMILIES = ('control', 'pp_dropout', 'modality_dropout', 'mixstyle', 'rdrop')
SEEDS = (0, 1, 2)
PARAMETERS = 135219


def configuration(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    return b.original._resolve_config(dict(b.original._CONFIG_BY_NAME['pp_tcn'], steps=steps,
        seeds=SEEDS, batch=16, crop=384, lr=.002))


def fit(datasets, source, family, steps=500):
    if family not in FAMILIES:
        raise ValueError('Unknown provider-generalization family')
    names = tuple(data['_name'] for data in datasets)
    if names != b.train_names(source) or set(names) & set(b.test_names(source)):
        raise ValueError('Training must use the complete declared source provider only')
    if any(data['_source'] != source for data in datasets):
        raise ValueError('Mixed training providers forbidden')
    config = configuration(steps)
    records = b.original._prepare_records(datasets, config)
    for data, record in zip(datasets, records):
        record['blocks'] = b.original._normalize_blocks(data['blocks'], len(record['y']))
    records = b.old_sampling.prepare(records, 'event_tcn', config['crop'])
    grouped = b.original._balanced_index(records)
    mean, scale = b.original._player_standardisation(records)
    stats = b.statistics(datasets)
    device = b.original._device()
    nets, seeds = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        augmentation_rng = np.random.default_rng(100000 + seed)
        net = b.original._make_model(config).to(device).train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], total_steps=steps) if steps > 1 else None
        for _ in range(steps):
            x_np, pp_np, ppm_np, y_np, view_np, weight_np = b.old_sampling.make_batch(
                records, grouped, mean, scale, config, rng, 'event_tcn')
            x = torch.from_numpy(x_np).to(device)
            pp = torch.from_numpy(pp_np).to(device=device, dtype=torch.float32)
            ppm = torch.from_numpy(ppm_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            views = torch.from_numpy(view_np).to(device)
            weights = torch.from_numpy(weight_np).to(device)
            if family in ('pp_dropout', 'modality_dropout'):
                x, pp, ppm = dropout.apply(x, pp, ppm, augmentation_rng, family)
            elif family == 'mixstyle':
                x = style.apply(x, augmentation_rng)
            first = net(x, pp, ppm)
            if family == 'rdrop':
                second = net(x, pp, ppm)
                value = consistency.loss(first, second, y, views, weights)
            else:
                second = None
                value = b.old_weights.state_loss(first, y, views, weights, 'event_tcn')
            if not torch.isfinite(value):
                raise ValueError('Nonfinite training objective')
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        count = sum(parameter.numel() for parameter in net.parameters())
        if count != PARAMETERS:
            raise ValueError('Architecture parameter count changed')
        seeds.append(dict(seed=seed, updates=steps, parameters=count, final_training_loss=float(value.detach().cpu()),
                          stochastic_forwards_per_update=2 if family == 'rdrop' else 1))
        net.eval().to('cpu')
        nets.append(net)
        del optimizer, scheduler, x, pp, ppm, y, views, weights, first, second, value
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return dict(nets=nets, mean=mean, scale=scale, config=config, family=family, source=source,
        training_names=names, training_stats=stats, training=dict(names=list(names), matches=len(names), provider=source,
        device=str(device), seed_fits=seeds, steps=steps, event_sampling_probability=.5,
        match_audit=[dict(match=record['name'], source=record['source'],
                          eligible_short_events=len(record['short_events'])) for record in records]))


def predict(fitted, observed):
    if 'y' in observed or 'y25' in observed:
        raise ValueError('Target labels must not be available to inference')
    if tuple(fitted['training_names']) != b.train_names(fitted['source']):
        raise ValueError('Checkpoint source training membership changed')
    if observed['_name'] not in b.test_names(fitted['source']) or observed['_name'] in fitted['training_names']:
        raise ValueError('Prediction match/provider overlaps source training')
    return b.result(b.original.predict(fitted, observed)['probability'])
