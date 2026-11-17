import hashlib
import json

import access
access.initialize()

import numpy as np

import bridge as b


STRESS = b.c.import_path('provider_transfer_stress_math', b.c.HERE / 'prepare_stress.py')
OBSERVATION_KEYS = ('X', 'players', 'ball', 'teams', 'blocks', 'blocks25', 'n_frames', 'fps', 'sampled', 'half')
LABEL_KEYS = ('y', 'y25', 'blocks', 'blocks25', 'n_frames', 'fps', 'sampled')


def save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Preparation must not overwrite an existing carrier: ' + str(path))
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError('Object arrays forbidden in carriers')
    np.savez_compressed(path, **arrays)


def gap(observed, name):
    blocks = np.asarray(observed['blocks'], dtype=np.int64)
    groups = STRESS._half_groups(blocks, np.asarray(observed['blocks25']))
    visible = np.isfinite(observed['ball']).all(axis=1)
    dropped = np.zeros(len(visible), dtype=bool)
    seed = 20261119 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
    audits = []
    for index, group in enumerate(groups):
        mask, spans, count = STRESS._drop_half(visible, blocks, group, seed + index)
        dropped |= mask
        audits.append(dict(half=index + 1, source_visible=count, dropped=int(mask.sum()), spans=spans))
    ball = observed['ball'].copy()
    ball[dropped] = np.nan
    result = {**observed, 'ball': ball}
    distances = b.c.base.decoder_observations(result)['distances']
    result['X'] = b.wave1.features(result['players'], ball, distances, blocks.tolist())
    return result, dict(seed=seed, visible_before=int(visible.sum()), dropped=int(dropped.sum()), halves=audits)


def main():
    path = b.OUT / 'preparation.json'
    if path.exists():
        report = json.loads(path.read_text())
        for row in report['records']:
            for field, digest_field in (('original_pack', 'original_pack_sha256'), ('original_features', 'original_features_sha256'), ('label_path', 'label_sha256')):
                if b.c.digest(b.c.REPO / row[field]) != row[digest_field]:
                    raise ValueError('Existing preparation source/vault changed')
            for case in row['cases']:
                for field in ('observations', 'features'):
                    if b.c.digest(b.c.REPO / case[field]) != case[field + '_sha256']:
                        raise ValueError('Existing observation preparation changed')
        print('PREPARATION VERIFIED', len(report['records']), flush=True)
        return
    records = []
    for name in (*b.IDSSE, *b.SC):
        source = b.c.load(name, b.regime(name))
        observed = {key: source[key] for key in OBSERVATION_KEYS if key in source}
        labels = {key: source[key] for key in LABEL_KEYS}
        label_path = b.CACHE / 'labels' / f'{name}.npz'
        save(label_path, labels)
        changed, stress = gap(observed, name)
        cases = []
        for condition, data in (('clean', observed), ('gap', changed)):
            observation_path, feature_path = b.case_paths(name, condition)
            if set(data) & {'y', 'y25'}:
                raise ValueError('Target labels entered observation carrier')
            if condition == 'clean':
                with np.load(b.original_features(name), allow_pickle=False) as archive:
                    features = {key: archive[key] for key in archive.files}
            else:
                features = b.feature_builder.build(data)
            features = {key: np.asarray(value) for key, value in features.items()}
            b.original._feature_arrays({**data, '_features': features}, True, False)
            save(observation_path, data)
            save(feature_path, features)
            cases.append(dict(condition=condition, observations=observation_path.relative_to(b.c.REPO).as_posix(),
                observations_sha256=b.c.digest(observation_path), features=feature_path.relative_to(b.c.REPO).as_posix(),
                features_sha256=b.c.digest(feature_path)))
        records.append(dict(match=name, provider=b.provider(name), regime=b.regime(name), samples=len(source['X']),
            original_pack=b.original_pack(name).relative_to(b.c.REPO).as_posix(), original_pack_sha256=b.c.digest(b.original_pack(name)),
            original_features=b.original_features(name).relative_to(b.c.REPO).as_posix(),
            original_features_sha256=b.c.digest(b.original_features(name)), label_path=label_path.relative_to(b.c.REPO).as_posix(),
            label_sha256=b.c.digest(label_path), cases=cases, stress=stress, observations_label_free=True))
        print('PREPARED', name, 'clean/gap', 'samples', len(source['X']), flush=True)
    b.c.write_json(path, dict(records=records, metric_evaluation=False, target_game_used=False,
        unique_external_matches=27, observation_conditions=54, guard=access.report()))
    print('PREPARED ALL', len(records), 'NO METRIC EVALUATION', flush=True)


if __name__ == '__main__':
    main()
