"""Local-only features; external event labels never use competition ground truth."""
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKSPACE = REPO.parent
RYU = Path(os.environ.get('FARRAS_SOURCE_EXPERIMENTS', str(REPO / 'experiments'))).resolve()
CACHE = HERE / 'cache'
FPS = 5
STRIDE = 5
FEATURE_NAMES = ['d_home', 'd_away', 'margin', 'ball_visible', 'ball_x', 'ball_y',
                 'ball_speed', 'ball_boundary_distance', 'median_player_speed',
                 'moving_fraction', 'player_x_spread', 'player_y_spread',
                 'player_centroid_x', 'player_centroid_y', 'gap_age', 'gap_remaining']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def period_blocks(period):
    return [bounds for value in np.unique(period) if value > 0 for bounds in runs(period == value)]


def event_labels(events, period, frame):
    y = np.full(len(frame), -1, np.int8)
    for half in np.unique(period):
        if half <= 0:
            continue
        indices = np.flatnonzero(period == half)
        times = frame[indices]
        marks = []
        for _, event in events[events['Period'] == half].iterrows():
            kind, sub = str(event['Type']), str(event['Subtype'])
            start, end = int(event['Start Frame']), int(event['End Frame'])
            team = 0 if event['Team'] == 'Home' else 1
            if kind in ('PASS', 'CARRY', 'RECOVERY', 'SET PIECE', 'SHOT'):
                marks.extend([(start, 0, -1), (start, 1, team)])
                if kind == 'SHOT' and 'GOAL' in sub.upper():
                    marks.append((end, 2, -1))
            elif kind in ('BALL OUT', 'FAULT RECEIVED', 'CARD'):
                marks.append((start, 2, -1))
        marks.sort()
        owner, dead, previous = -1, False, None
        for time, kind, team in marks + [(int(times[-1]) + 1, 3, -1)]:
            if previous is not None and owner >= 0:
                a, b = np.searchsorted(times, [previous, time])
                y[indices[a:b]] = 2 if dead else owner
            if kind == 0:
                dead = False
            elif kind == 1:
                owner = team
            elif kind == 2:
                dead = True
            previous = time
    return y


def gap_features(X, blocks):
    X[:, 14:16] = 0
    for begin, end in blocks:
        for a, b in runs(X[begin:end, 3] == 0):
            a, b = a + begin, b + begin
            X[a:b, 14] = np.minimum(np.arange(b - a) / FPS, 60)
            X[a:b, 15] = np.minimum(np.arange(b - a, 0, -1) / FPS, 60)


def features(players, ball, distances, blocks):
    n = len(ball)
    X = np.zeros((n, 16), np.float32)
    visible = np.isfinite(ball).all(axis=1)
    X[:, :2] = np.clip(np.nan_to_num(distances, nan=30, posinf=30), 0, 30)
    X[:, 2] = np.clip((X[:, 1] - X[:, 0]) / 3, -10, 10)
    X[:, 3] = visible
    X[:, 4:6] = np.nan_to_num(ball, nan=.5)
    X[:, 7] = np.where(visible, np.minimum.reduce([ball[:, 0] * 105.3,
        (1 - ball[:, 0]) * 105.3, ball[:, 1] * 68, (1 - ball[:, 1]) * 68]), 17)
    X[:, 7] = np.clip(X[:, 7], -5, 34)
    speeds = np.full(players.shape[:2], np.nan, np.float32)
    for a, b in blocks:
        if b - a > 1:
            delta = np.diff(players[a:b], axis=0) * [105.3, 68]
            speeds[a + 1:b] = np.minimum(np.linalg.norm(delta, axis=2) * FPS, 12)
            bd = np.diff(ball[a:b], axis=0) * [105.3, 68]
            X[a + 1:b, 6] = np.minimum(np.linalg.norm(bd, axis=1) * FPS, 40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        X[:, 8] = np.nanmedian(speeds, axis=1)
        count = np.isfinite(speeds).sum(axis=1)
        X[:, 9] = (speeds > 1.5).sum(axis=1) / np.maximum(count, 1)
        quantiles = np.nanpercentile(players, [10, 90], axis=1)
        X[:, 10:12] = quantiles[1] - quantiles[0]
        X[:, 12:14] = np.nanmean(players, axis=1)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    gap_features(X, blocks)
    return X


def occlude(X, blocks, seed):
    X = X.copy()
    rng = np.random.default_rng(seed)
    for a, b in blocks:
        position, missing = a, False
        while position < b:
            duration = max(1, int(rng.exponential(10 if missing else 15) * FPS))
            end = min(position + duration, b)
            if missing:
                X[position:end, :2] = 30
                X[position:end, 2:4] = 0
                X[position:end, 4:6] = .5
                X[position:end, 6] = 0
                X[position:end, 7] = 17
            position, missing = end, not missing
        reappeared = (X[a + 1:b, 3] > 0) & (X[a:b - 1, 3] == 0)
        X[a + 1:b, 6][reappeared] = 0
    gap_features(X, blocks)
    return X


def load_external(number):
    directory = RYU / '_cache/ext_metrica'
    tables = [pd.read_csv(directory / f'Sample_Game_{number}_RawTrackingData_{side}_Team.csv',
                          header=None, skiprows=3) for side in ('Home', 'Away')]
    home, away = tables
    if not np.array_equal(home.iloc[:, :2], away.iloc[:, :2]):
        raise ValueError('Home/Away external timelines differ')
    period = home.iloc[:, 0].to_numpy(dtype=int)
    frame = home.iloc[:, 1].to_numpy(dtype=int)
    blocks25 = period_blocks(period)
    events = pd.read_csv(directory / f'Sample_Game_{number}_RawEventsData.csv')
    y25 = event_labels(events, period, frame)
    home_xy = home.iloc[:, 3:-2].to_numpy(np.float32).reshape(len(home), -1, 2)
    away_xy = away.iloc[:, 3:-2].to_numpy(np.float32).reshape(len(away), -1, 2)
    ball25 = home.iloc[:, -2:].to_numpy(np.float32)
    sampled = np.concatenate([np.arange(a, b, STRIDE) for a, b in blocks25])
    players = np.concatenate([home_xy[sampled], away_xy[sampled]], axis=1)
    ball = ball25[sampled]
    team = np.r_[np.zeros(home_xy.shape[1], int), np.ones(away_xy.shape[1], int)]
    distance = np.linalg.norm((players - ball[:, None]) * [105.3, 68], axis=2)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    distances = np.column_stack([distance[:, team == k].min(axis=1) for k in (0, 1)])
    blocks, offset = [], 0
    for a, b in blocks25:
        size = len(range(a, b, STRIDE))
        blocks.append((offset, offset + size))
        offset += size
    X = features(players, ball, distances, blocks)
    return dict(X=X, X_occluded=occlude(X, blocks, 2027 + number), y=y25[sampled],
                y25=y25, sampled=sampled, blocks=np.array(blocks), blocks25=np.array(blocks25),
                n_frames=len(period), fps=FPS)


def load_target():
    sys.path.insert(0, str(REPO))
    from games import half_video_blocks
    flat = np.load(RYU / '_cache/atd_full.npy', mmap_mode='r')
    distances = np.load(RYU / '_cache/team_dist.npy', mmap_mode='r')
    blocks25 = [(a, b + 1) for _, a, b in half_video_blocks('BAR-ATH')]
    sampled = np.concatenate([np.arange(a, b, STRIDE) for a, b in blocks25])
    players = flat[sampled, :-3].reshape(len(sampled), 46, 3)[:, :, :2]
    ball = flat[sampled, -3:-1]
    blocks, offset = [], 0
    for a, b in blocks25:
        size = len(range(a, b, STRIDE))
        blocks.append((offset, offset + size))
        offset += size
    X = features(players, ball, distances[sampled], blocks)
    return dict(X=X, y=np.full(len(sampled), -1, np.int8), sampled=sampled,
                blocks=np.array(blocks), blocks25=np.array(blocks25), n_frames=len(flat), fps=FPS)


def expand(prediction, data):
    out = np.full(int(data['n_frames']), 3, np.int8)
    for (a, b), (first, last) in zip(data['blocks'], data['blocks25']):
        out[first:last] = np.repeat(prediction[a:b], STRIDE)[:last - first]
    return out


def inputs():
    paths = [RYU / '_cache'/name for name in ('atd_full.npy', 'team_dist.npy')]
    for game in (1, 2):
        paths.extend(RYU / '_cache/ext_metrica'/f'Sample_Game_{game}_{suffix}.csv' for suffix in
                     ('RawEventsData', 'RawTrackingData_Home_Team', 'RawTrackingData_Away_Team'))
    return paths


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {str(path.relative_to(WORKSPACE)): digest(path) for path in inputs()}
    for number in (1, 2):
        data = load_external(number)
        np.savez(CACHE / f'external_{number}.npz', **data)
        print(f'external_{number}: {len(data["X"])} samples, classes '
              f'{np.bincount(data["y"][data["y"] >= 0], minlength=3).tolist()}', flush=True)
    target = load_target()
    np.savez(CACHE / 'target.npz', **target)
    manifest['feature_code_sha256'] = digest(Path(__file__))
    manifest['feature_names'] = FEATURE_NAMES
    manifest['source_notice'] = 'Inputs in exp_ryu are read-only; competition cache provenance is inherited, not regenerated.'
    (CACHE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'target: {len(target["X"])} samples; visible {target["X"][:,3].mean():.3f}', flush=True)


if __name__ == '__main__':
    prepare()
