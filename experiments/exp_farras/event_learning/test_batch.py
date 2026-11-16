import unittest
from unittest.mock import patch

import numpy as np

import batch
import bridge as b


FAMILIES = ('balanced', 'focal', 'boundary', 'ranking', 'ema', 'contact')


def feature_record(name='first', identity=0, source='idsse'):
    length = 1536
    frames = np.arange(length, dtype=np.float32)
    player = np.zeros((2, length, 23), dtype=np.float16)
    valid = np.empty((2, length), dtype=np.bool_)
    ball = np.ones((2, length, 5), dtype=np.float16)
    pp = np.zeros((2, length, 48, 10), dtype=np.float16)
    ppm = np.empty((2, length, 48), dtype=np.bool_)
    control = np.empty((2, length, 7), dtype=np.float16)
    signs = np.asarray((-1, 1, -1, 1, -1, 1, -1), dtype=np.float32)
    raw_control = 1 + frames[:, None] / 2048 + np.arange(7)[None, :] / 8
    for view in range(2):
        player[view, :, 0] = frames
        player[view, :, 1] = view
        player[view, :, 2] = identity
        player[view, :, 3:] = (frames[:, None] % 17) / 8
        valid[view] = (np.arange(length) + view) % 7 != 0
        ball[view, :, 0] = frames / 8 + 1
        ball[view, :, 1] = view + 1
        ball[view, 19::113, 2] = np.nan
        pp[view, :, :, 0] = frames[:, None] / 8
        pp[view, :, :, 1] = view + np.arange(48)[None, :] / 64
        pp[view, :, :, 2:6] = 2
        pp[view, :, :, 6:9] = 3 + view
        pp[view, :, :, 9] = 4
        ppm[view] = (np.arange(length)[:, None] + np.arange(48)[None, :] + view) % 5 != 0
        control[view] = raw_control * (signs if view else 1)
    player[0, 11::127, 5] = np.nan
    labels = np.full(length, 2, dtype=np.int8)
    labels[30:35] = 0
    labels[200:700] = 1
    labels[820:825] = 1
    labels[1000:1019] = 0
    labels[1200:1230] = 1
    labels[1400:1410] = -1
    return dict(name=name, source=source, y=labels, blocks=[(0, 768), (768, length)],
                player=player, valid=valid, ball=ball, pp=pp, ppm=ppm, control=control)


def prepared_records(family, multiple=False):
    sources = [feature_record()]
    if multiple:
        sources.extend((feature_record(identity=1), feature_record('second', 2),
                        feature_record('third', 3, 'skillcorner')))
    records = b.old_sampling.prepare(sources, 'event_tcn', 384)
    for record in records:
        if family == 'balanced':
            record['weights'] = b.old_weights.spell_weights(record['y'], record['blocks'])
        if family == 'boundary':
            target = np.empty(len(record['y']), dtype=np.float32)
            valid = np.empty(len(record['y']), dtype=np.bool_)
            for start, end in record['blocks']:
                target[start:end], valid[start:end] = b.training.boundary_targets(record['y'][start:end])
            record.update(boundary_target=target, boundary_valid=valid)
        for value in record.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    return records


def config(family, batch_size=8, mask_full=.3, noise=.5):
    return dict(name='control_tcn' if family == 'contact' else 'pp_tcn', batch=batch_size,
                crop=384, mask_full=mask_full, noise=noise)


class EventBatch(unittest.TestCase):
    def test_all_families_preserve_old_event_batch_and_rng_order(self):
        mean = np.linspace(-2, 2, 23, dtype=np.float32)
        scale = np.linspace(.5, 2, 23, dtype=np.float32)
        for family in FAMILIES:
            records = prepared_records(family, multiple=True)
            grouped = b.original._balanced_index(records)
            for seed, mask_full, noise in ((0, .3, .5), (3, 1., 0.), (7, 0., .5)):
                with self.subTest(family=family, seed=seed):
                    settings = config(family, mask_full=mask_full, noise=noise)
                    actual_rng = np.random.default_rng(seed)
                    expected_rng = np.random.default_rng(seed)
                    actual = batch.make_batch(records, grouped, mean, scale, settings, actual_rng, family)
                    expected = b.old_sampling.make_batch(records, grouped, mean, scale, settings,
                                                         expected_rng, 'event_tcn')
                    self.assertEqual(len(actual), 8)
                    np.testing.assert_array_equal(actual[0][:, :29], expected[0])
                    for result, previous in zip(actual[1:6], expected[1:]):
                        np.testing.assert_array_equal(result, previous)
                    self.assertEqual(actual_rng.bit_generator.state, expected_rng.bit_generator.state)
                    self.assertEqual(actual[0].shape, (8, 36 if family == 'contact' else 29, 384))
                    self.assertEqual(actual[0].dtype, np.dtype(np.float32))
                    self.assertEqual(actual[1].dtype, np.dtype(np.float16))
                    self.assertEqual(actual[2].dtype, np.dtype(np.bool_))
                    self.assertEqual(actual[3].dtype, np.dtype(np.int64))
                    self.assertEqual(actual[4].dtype, np.dtype(np.int64))
                    self.assertEqual(actual[5].dtype, np.dtype(np.float32))
                    self.assertTrue(np.isfinite(actual[0]).all())
                    if family == 'boundary':
                        self.assertEqual(actual[6].shape, (8, 384))
                        self.assertEqual(actual[6].dtype, np.dtype(np.float32))
                        self.assertEqual(actual[7].dtype, np.dtype(np.bool_))
                    else:
                        self.assertIsNone(actual[6])
                        self.assertIsNone(actual[7])

    def test_crops_align_full_arrays_views_masks_and_halves(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                records = prepared_records(family, multiple=True)
                grouped = b.original._balanced_index(records)
                x, pp, ppm, y, views, weights, targets, valid = batch.make_batch(
                    records, grouped, np.zeros(23, dtype=np.float32), np.ones(23, dtype=np.float32),
                    config(family, batch_size=32, noise=0.), np.random.default_rng(981), family,
                )
                halves, sampled_sources = set(), set()
                for index, view in enumerate(views):
                    start = int(x[index, 0, 0])
                    end = start + 384
                    record = records[int(x[index, 2, 0])]
                    self.assertTrue(any(left <= start and end <= right for left, right in record['blocks']))
                    halves.add(start // 768)
                    sampled_sources.add(record['source'])
                    np.testing.assert_array_equal(x[index, 0], np.arange(start, end))
                    np.testing.assert_array_equal(x[index, 1], view)
                    np.testing.assert_array_equal(x[index, 23], record['valid'][view, start:end])
                    np.testing.assert_array_equal(y[index], record['y'][start:end])
                    np.testing.assert_array_equal(weights[index], record['weights'][start:end])
                    np.testing.assert_array_equal(ppm[index], record['ppm'][view, start:end])
                    masked = x[index, 28] == 0
                    expected_ball = np.nan_to_num(record['ball'][view, start:end].astype(np.float32))
                    expected_ball[masked] = 0
                    np.testing.assert_array_equal(x[index, 24:29], expected_ball.T)
                    expected_pp = record['pp'][view, start:end].copy()
                    expected_pp[masked, :, 6:9] = 0
                    np.testing.assert_array_equal(pp[index], expected_pp)
                    if family == 'contact':
                        expected_control = record['control'][view, start:end].astype(np.float32)
                        expected_control[masked] = 0
                        np.testing.assert_array_equal(x[index, 29:36], expected_control.T)
                    if family == 'boundary':
                        np.testing.assert_array_equal(targets[index], record['boundary_target'][start:end])
                        np.testing.assert_array_equal(valid[index], record['boundary_valid'][start:end])
                self.assertEqual(halves, {0, 1})
                self.assertEqual(set(views), {0, 1})
                self.assertEqual(sampled_sources, {'idsse', 'skillcorner'})

    def test_boundary_crop_first_frame_keeps_full_match_target_and_validity(self):
        records = prepared_records('boundary')
        record = records[0]
        with patch.object(b.old_sampling, 'draw', return_value=(record, 200, 1)):
            actual = batch.make_batch(records, b.original._balanced_index(records), np.zeros(23),
                                      np.ones(23), config('boundary', batch_size=1, noise=0.),
                                      np.random.default_rng(0), 'boundary')
        self.assertEqual(actual[6][0, 0], 1.)
        self.assertTrue(actual[7][0, 0])
        np.testing.assert_array_equal(actual[6][0], record['boundary_target'][200:584])
        np.testing.assert_array_equal(actual[7][0], record['boundary_valid'][200:584])
        _, cropped_valid = b.training.boundary_targets(record['y'][200:584])
        self.assertFalse(cropped_valid[0])

    def test_balanced_weights_use_full_spell_duration_not_crop_duration(self):
        records = prepared_records('balanced')
        record = records[0]
        with patch.object(b.old_sampling, 'draw', return_value=(record, 210, 0)):
            actual = batch.make_batch(records, b.original._balanced_index(records), np.zeros(23),
                                      np.ones(23), config('balanced', batch_size=1, noise=0.),
                                      np.random.default_rng(0), 'balanced')
        np.testing.assert_array_equal(actual[5][0], np.full(384, .25, dtype=np.float32))

    def test_contact_uses_raw_view_controls_and_shared_ball_pp_mask(self):
        records = prepared_records('contact')
        record = records[0]
        start, end = 180, 564
        mask = np.arange(384) % 3 == 0
        with patch.object(b.old_sampling, 'draw', side_effect=((record, start, 0), (record, start, 1))) as draw:
            with patch.object(b.original, '_random_mask', side_effect=lambda *args: mask.copy()):
                x, pp, ppm, y, views, weights, target, valid = batch.make_batch(
                    records, b.original._balanced_index(records), np.zeros(23), np.ones(23),
                    config('contact', batch_size=2, noise=0.), np.random.default_rng(8), 'contact',
                )
        self.assertEqual(draw.call_count, 2)
        self.assertTrue(all(call.kwargs == {'event_probability': .5} for call in draw.call_args_list))
        np.testing.assert_array_equal(views, [0, 1])
        for view in (0, 1):
            raw = record['control'][view, start:end].astype(np.float32)
            np.testing.assert_array_equal(x[view, 29:36][:, ~mask], raw[~mask].T)
            np.testing.assert_array_equal(x[view, 29:36, mask], 0.)
            np.testing.assert_array_equal(x[view, 24:29, mask], 0.)
            np.testing.assert_array_equal(pp[view, mask, :, 6:9], 0.)
            np.testing.assert_array_equal(pp[view, :, :, :6], record['pp'][view, start:end, :, :6])
            np.testing.assert_array_equal(pp[view, :, :, 9], record['pp'][view, start:end, :, 9])
            np.testing.assert_array_equal(ppm[view], record['ppm'][view, start:end])
            np.testing.assert_array_equal(y[view], record['y'][start:end])
            np.testing.assert_array_equal(weights[view], record['weights'][start:end])
        signs = np.asarray((-1, 1, -1, 1, -1, 1, -1), dtype=np.float32)
        np.testing.assert_array_equal(x[1, 29:36], x[0, 29:36] * signs[:, None])
        self.assertIsNone(target)
        self.assertIsNone(valid)

    def test_batch_never_mutates_provided_arrays(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                records = prepared_records(family)
                before = {key: value.copy() for key, value in records[0].items() if isinstance(value, np.ndarray)}
                batch.make_batch(records, b.original._balanced_index(records), np.zeros(23), np.ones(23),
                                 config(family), np.random.default_rng(4), family)
                for key, value in before.items():
                    np.testing.assert_array_equal(records[0][key], value)
                    self.assertFalse(records[0][key].flags.writeable)

    def test_unknown_family_rejected_before_rng_draw(self):
        rng = np.random.default_rng(0)
        state = rng.bit_generator.state
        with self.assertRaises(ValueError):
            batch.make_batch([], {}, np.zeros(23), np.ones(23), config('focal'), rng, 'event_tcn')
        self.assertEqual(rng.bit_generator.state, state)


class ForbiddenLabels:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Ground truth accessed')

    def __len__(self):
        raise AssertionError('Ground truth accessed')


class FeatureAttachment(unittest.TestCase):
    def test_new_maps_preserve_observations_and_do_not_read_ground_truth(self):
        source = feature_record()
        features = {key: source[key] for key in ('player', 'valid', 'ball', 'pp', 'ppm')}
        data = dict(_features=features, _name='heldout', blocks=source['blocks'],
                    y=ForbiddenLabels(), y25=ForbiddenLabels())
        controls = source['control']
        controls.setflags(write=False)
        attached = batch.attach(data, controls)
        self.assertIsNot(attached, data)
        self.assertIsNot(attached['_features'], features)
        self.assertNotIn('control', features)
        self.assertIs(attached['_features']['control'], controls)
        for key in features:
            self.assertIs(attached['_features'][key], features[key])
        self.assertIs(attached['y'], data['y'])
        self.assertIs(attached['y25'], data['y25'])
        self.assertIs(attached['blocks'], data['blocks'])
        np.testing.assert_array_equal(attached['_features']['control'], controls)

    def test_unlabeled_observations_accept_both_views_without_gt_lookup(self):
        source = feature_record()
        observations = {'_features': {'player': source['player']}}
        attached = batch.attach(observations, source['control'])
        self.assertEqual(set(attached), {'_features'})
        np.testing.assert_array_equal(attached['_features']['control'], source['control'])
        self.assertNotIn('control', observations['_features'])

    def test_control_shape_grid_and_finiteness_required(self):
        observations = {'_features': {'player': np.zeros((2, 9, 23), dtype=np.float16)}}
        invalid = [np.zeros((9, 7)), np.zeros((1, 9, 7)), np.zeros((2, 8, 7)),
                   np.zeros((2, 9, 6)), np.zeros((2, 9, 8))]
        for bad_value in (np.nan, np.inf, -np.inf):
            control = np.zeros((2, 9, 7), dtype=np.float32)
            control[1, 4, 6] = bad_value
            invalid.append(control)
        for control in invalid:
            with self.subTest(shape=control.shape):
                with self.assertRaises(ValueError):
                    batch.attach(observations, control)
        self.assertNotIn('control', observations['_features'])

    def test_existing_control_is_replaced_only_in_new_feature_map(self):
        player = np.zeros((2, 9, 23), dtype=np.float16)
        old = np.zeros((2, 9, 7), dtype=np.float32)
        new = np.ones((2, 9, 7), dtype=np.float32)
        observations = {'_features': {'player': player, 'control': old}}
        attached = batch.attach(observations, new)
        self.assertIs(observations['_features']['control'], old)
        self.assertIs(attached['_features']['control'], new)


if __name__ == '__main__':
    unittest.main()
