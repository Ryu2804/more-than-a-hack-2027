"""Parse Metrica Game 3 using per-format player channels and event-period bounds."""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / 'wave1'
spec = importlib.util.spec_from_file_location('wave1_data', FIRST / 'data.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE = base.RYU / '_cache/ext_metrica'


def prepare():
    paths = {key: SOURCE / f'Sample_Game_3_{suffix}' for key, suffix in
             [('tracking', 'tracking.txt'), ('metadata', 'metadata.xml'), ('events', 'events.json')]}
    meta = ET.parse(paths['metadata']).getroot()
    events = json.loads(paths['events'].read_text())['data']
    team_of = {p.attrib['id']: 0 if p.attrib['teamId'] == 'FIFATMA' else 1
               for p in meta.iter('Player')}
    ids = list(team_of)
    index_of = {player: i for i, player in enumerate(ids)}
    channels = {c.attrib['id']: c.attrib for c in meta.iter('PlayerChannel')}
    formats = sorted(meta.iter('DataFormatSpecification'), key=lambda f: int(f.attrib['startFrame']))
    raw, ball_rows, frame_rows = [], [], []
    with paths['tracking'].open() as stream:
        for line in stream:
            frame, player_values, ball_values = line.strip().split(':')
            frame_rows.append(int(frame))
            raw.append(np.fromstring(player_values.replace(';', ','), sep=',', dtype=np.float32))
            ball_rows.append(np.fromstring(ball_values, sep=',', dtype=np.float32))
    frames = np.asarray(frame_rows)
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError('Game 3 tracking is not contiguous one-based frames')
    raw = np.stack(raw).reshape(len(frames), -1, 2)
    ball = np.stack(ball_rows)
    players = np.full((len(frames), len(ids), 2), np.nan, np.float32)
    coverage = np.zeros(len(frames), np.uint8)
    format_report = []
    for fmt in formats:
        a, b = int(fmt.attrib['startFrame']) - 1, int(fmt.attrib['endFrame'])
        refs = [channels[r.attrib['playerChannelId']] for r in fmt.iter('PlayerChannelRef')]
        if len(refs) != 2 * raw.shape[1]:
            raise ValueError('Player-slot width differs from active metadata format')
        selected = []
        for x, y in zip(refs[::2], refs[1::2]):
            if x['channelId'] != 'x' or y['channelId'] != 'y' or x['playerId'] != y['playerId']:
                raise ValueError('Metadata coordinates are not paired x/y for same player')
            selected.append(index_of[x['playerId']])
        if len(set(selected)) != len(selected):
            raise ValueError('Duplicate active player identity')
        players[a:b, selected, :] = raw[a:b]
        coverage[a:b] += 1
        format_report.append(dict(first_frame=a + 1, last_frame=b,
            home_slots=sum(team_of[ids[i]] == 0 for i in selected),
            away_slots=sum(team_of[ids[i]] == 1 for i in selected)))
    if not np.all(coverage == 1):
        raise ValueError('Metadata formats must partition every tracking frame once')
    period = np.zeros(len(frames), np.int8)
    event_rows = []
    period_bounds = []
    for half in sorted({e['period'] for e in events}):
        group = [e for e in events if e['period'] == half]
        start = min(e['start']['frame'] for e in group)
        end = max(e['end']['frame'] for e in group)
        if start < 1 or end > len(frames) or np.any(period[start - 1:end]):
            raise ValueError('Event period bounds overlap or exceed tracking')
        period[start - 1:end] = half
        period_bounds.append(dict(period=int(half), first_frame=int(start), last_frame=int(end)))
    for event in events:
        subtype = event['subtypes']
        subtypes = [subtype] if isinstance(subtype, dict) else (subtype or [])
        event_rows.append({'Period': event['period'], 'Team': 'Home' if event['team']['id'] == 'FIFATMA' else 'Away',
            'Type': event['type']['name'], 'Subtype': '|'.join(s['name'] for s in subtypes),
            'Start Frame': event['start']['frame'], 'End Frame': event['end']['frame']})
    y25 = base.event_labels(pd.DataFrame(event_rows), period, frames)
    blocks25 = base.period_blocks(period)
    sampled = np.concatenate([np.arange(a, b, 5) for a, b in blocks25])
    p, ball = players[sampled], ball[sampled]
    team = np.array([team_of[i] for i in ids])
    d = np.linalg.norm((p - ball[:, None]) * [105.3, 68], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    distances = np.column_stack([d[:, team == t].min(axis=1) for t in (0, 1)])
    blocks, offset = [], 0
    for a, b in blocks25:
        count = len(range(a, b, 5))
        blocks.append((offset, offset + count))
        offset += count
    X = base.features(p, ball, distances, blocks)
    X_occluded = base.occlude(X, blocks, 2030)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite holdout features')
    cache = HERE / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / 'holdout_3.npz', X=X, X_occluded=X_occluded, y=y25[sampled], y25=y25,
             sampled=sampled, blocks=np.asarray(blocks), blocks25=np.asarray(blocks25),
             n_frames=len(frames), fps=5)
    report = dict(source_sha256={key: base.digest(path) for key, path in paths.items()},
        parser_sha256=base.digest(Path(__file__)), frames=len(frames), active_slots=int(raw.shape[1]),
        unique_players=len(ids), formats=format_report, periods=period_bounds,
        known_label_counts={label: int((y25 == k).sum()) for k, label in enumerate(['H', 'A', 'D'])},
        unknown_frames=int((y25 < 0).sum()), feature_shape=list(X.shape),
        visible_fraction=float(X[:, 3].mean()), occluded_visible_fraction=float(X_occluded[:, 3].mean()),
        team_assignment='FIFATMA=H; FIFATMB=A; resolve every active channel through frame-specific metadata',
        period_rule='Only first-event-start through last-event-end of each declared event period is included; no inferred halftime labels.',
        pitch_note='Feature scaling reuses wave1 convention 105.3x68m; Game3 metadata declares105x68m.',
        prior_exposure='Game3 raw data/imbalance was discussed in Ryu notes; corrected parser and predictions not used for wave2 selection.',
        label_rule='Same event-derived proxy rules as Games1/2; not corrected tracking ground truth.')
    (cache / 'holdout_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['frames', 'unique_players', 'periods', 'known_label_counts', 'feature_shape']}), flush=True)


if __name__ == '__main__':
    prepare()
