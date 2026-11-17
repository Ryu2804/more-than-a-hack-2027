import gc

import numpy as np
import torch

import architecture
import bridge as b
import sampling
import weights

FAMILIES = ('spell_balanced_tcn', 'event_tcn', 'dual_scale_tcn')
SEEDS = (0, 1, 2)


def configuration(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    return b.original._resolve_config(dict(b.original._CONFIG_BY_NAME['pp_tcn'],
        steps=steps, seeds=SEEDS, batch=16, crop=384, lr=.002))


def fit(datasets, family, steps=500):
    if family not in FAMILIES:
        raise ValueError('Unknown spell training family')
    datasets = list(datasets)
    names = tuple(data['_name'] for data in datasets)
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError('Training requires six unique matches')
    config = configuration(steps)
    raw = b.original._prepare_records(datasets, config)
    for dataset, record in zip(datasets, raw):
        record['blocks'] = b.original._normalize_blocks(dataset['blocks'], len(record['y']))
    records = sampling.prepare(raw, family, config['crop'])
    grouped = b.original._balanced_index(records)
    mean, scale = b.original._player_standardisation(records)
    device = b.original._device()
    nets, audit = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = architecture.make_model(config, family).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], total_steps=steps) if steps > 1 else None
        net.train()
        final_loss = None
        for _ in range(steps):
            x_np, pp_np, ppm_np, y_np, view_np, weight_np = sampling.make_batch(records, grouped, mean, scale, config, rng, family)
            x = torch.from_numpy(x_np).to(device)
            pp = torch.from_numpy(pp_np).to(device=device, dtype=torch.float32)
            ppm = torch.from_numpy(ppm_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            view = torch.from_numpy(view_np).to(device)
            state_weights = torch.from_numpy(weight_np).to(device)
            logits = net(x, pp, ppm)
            loss = weights.state_loss(logits, y, view, state_weights, family)
            if not torch.isfinite(loss):
                raise ValueError('Training loss became nonfinite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            final_loss = float(loss.detach().cpu())
        count = sum(parameter.numel() for parameter in net.parameters())
        net.eval().to('cpu')
        nets.append(net)
        audit.append(dict(seed=seed, updates=steps, parameters=count, final_training_loss=final_loss))
        del optimizer, scheduler, x, pp, ppm, y, view, state_weights, logits, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    record_audit = []
    for record in records:
        record_audit.append(dict(match=record['name'], source=record['source'],
                                 usable_halves=len(record['blocks']),
                                 eligible_short_events=len(record.get('short_events', []))))
    return dict(nets=nets, mean=mean, scale=scale, config=config, family=family,
                training_names=names, training=dict(matches=len(names), names=list(names),
                device=str(device), seeds=list(SEEDS), steps=steps, seed_fits=audit,
                match_audit=record_audit, input_channels=29,
                event_sampling_probability=.5 if family == 'event_tcn' else 0.0))


def predict(fitted, observations):
    if fitted['family'] not in FAMILIES:
        raise ValueError('Unknown fitted family')
    if 'y' in observations or 'y25' in observations:
        raise ValueError('Prediction observations must not include GT')
    if observations['_name'] in fitted['training_names']:
        raise ValueError('Prediction match was included in training')
    return b.result(b.original.predict(fitted, observations)['probability'])
