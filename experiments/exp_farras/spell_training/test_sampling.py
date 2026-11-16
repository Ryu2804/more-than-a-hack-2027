import unittest
from collections import Counter

import numpy as np

import bridge as b
import sampling


def label_record(name, labels, blocks, source='idsse'):
    return {'name': name, 'source': source, 'y': np.asarray(labels, dtype=np.int8), 'blocks': blocks}


def feature_record(name='train', identity=0):
    length = 1536
    frames = np.arange(length, dtype=np.float32)
    player = np.zeros((2, length, 23), dtype=np.float16)
    ball = np.ones((2, length, 5), dtype=np.float16)
    pp = np.zeros((2, length, 48, 10), dtype=np.float16)
    valid = np.empty((2, length), dtype=bool)
    ppm = np.empty((2, length, 48), dtype=bool)
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
    player[0, 11::127, 5] = np.nan
    labels = np.full(length, 2, dtype=np.int8)
    labels[30:35] = 0
    labels[200:700] = 1
    labels[820:825] = 1
    labels[1000:1019] = 0
    labels[1200:1230] = 1
    labels[1400:1410] = -1
    result = label_record(name, labels, [(0, 768), (768, length)])
    result.update(player=player, ball=ball, pp=pp, valid=valid, ppm=ppm)
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


class Preparation(unittest.TestCase):
    def test_complete_short_events_exclude_half_unknown_and_duration_edges(self):
        labels = np.full(1124, 2, dtype=np.int8)
        for start, end in ((0, 3), (30, 35), (50, 70), (80, 81), (91, 95), (509, 514), (600, 619), (1020, 1024), (1040, 1042)):
            labels[start:end] = 0
        labels[[81, 90]] = -1
        blocks = np.array([[0, 512], [512, 1024], [1024, 1124]])
        record = label_record('match', labels, blocks)
        labels.setflags(write=False)
        record['y'].setflags(write=False)
        blocks.setflags(write=False)
        before = record['y'].copy()
        actual = sampling.prepare([record], 'event_tcn', 384)[0]
        np.testing.assert_array_equal(actual['short_events'], [[0, 512, 30, 35], [512, 1024, 600, 619]])
        self.assertEqual(actual['short_events'].dtype, np.dtype(np.int64))
        self.assertEqual(actual['blocks'], [(0, 512), (512, 1024)])
        self.assertIsNot(actual, record)
        self.assertIs(actual['y'], record['y'])
        self.assertNotIn('weights', record)
        self.assertNotIn('short_events', record)
        np.testing.assert_array_equal(record['blocks'], blocks)
        np.testing.assert_array_equal(record['y'], before)
        self.assertEqual(actual['weights'].shape, (1124,))
        np.testing.assert_array_equal(actual['weights'][0:3], 2)
        np.testing.assert_array_equal(actual['weights'][509:514], 2)
        np.testing.assert_array_equal(actual['weights'][50:70], 1)
        np.testing.assert_array_equal(actual['weights'][80:82], 1)
        np.testing.assert_array_equal(actual['weights'][1040:1042], 2)

    def test_full_weights_precede_crop_and_non_event_families_have_no_events(self):
        record = feature_record()
        for family in ('spell_balanced_tcn', 'dual_scale_tcn'):
            with self.subTest(family=family):
                actual = sampling.prepare([record], family, 384)[0]
                self.assertEqual(actual['short_events'].shape, (0, 4))
                expected = .25 if family == 'spell_balanced_tcn' else 1
                np.testing.assert_array_equal(actual['weights'][200:700], expected)
                unknown = 0 if family == 'spell_balanced_tcn' else 1
                np.testing.assert_array_equal(actual['weights'][1400:1410], unknown)
                self.assertIs(actual['pp'], record['pp'])


class EventDraw(unittest.TestCase):
    def records(self):
        length = 6144
        rich = np.full(length, 2, dtype=np.int8)
        sparse = rich.copy()
        empty = rich.copy()
        for start in (500, 1200, 1900, 2600, 3572, 4272, 4972, 5672):
            rich[start:start + 5] = 0
        for start in (1800, 4800):
            sparse[start:start + 5] = 1
        blocks = [(0, 3072), (3072, length)]
        records = [label_record('rich', rich, blocks) for _ in range(3)]
        records += [label_record('sparse', sparse, blocks), label_record('none', empty, blocks)]
        records += [label_record('other-source', empty, blocks, 'skillcorner')]
        return sampling.prepare(records, 'event_tcn', 384)

    def test_zero_probability_preserves_original_draw_and_rng(self):
        records = self.records()
        grouped = b.original._balanced_index(records)
        actual_rng = np.random.default_rng(321)
        original_rng = np.random.default_rng(321)
        for _ in range(200):
            actual = sampling.draw(records, grouped, actual_rng, 384, 0)
            expected = b.original._sample_record(records, grouped, original_rng, 384)
            self.assertIs(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])
        np.testing.assert_array_equal(actual_rng.random(16), original_rng.random(16))

    def test_enrichment_keeps_source_match_variant_and_view_balance(self):
        records = self.records()
        grouped = b.original._balanced_index(records)
        fractions = []
        draws = 12000
        for probability in (0, .5):
            rng = np.random.default_rng(420 + int(probability * 10))
            hits = 0
            identities = Counter()
            variants = Counter()
            views = Counter()
            for _ in range(draws):
                record, start, view = sampling.draw(records, grouped, rng, 384, probability)
                identities[record['source'], record['name']] += 1
                variants[id(record)] += 1
                views[view] += 1
                hits += int(np.isin(record['y'][start:start + 384], (0, 1)).any())
                self.assertTrue(any(left <= start and start + 384 <= right for left, right in record['blocks']))
            fractions.append(hits / draws)
            for name in ('rich', 'sparse', 'none'):
                self.assertAlmostEqual(identities['idsse', name] / draws, 1 / 6, delta=.02)
            self.assertAlmostEqual(identities['skillcorner', 'other-source'] / draws, .5, delta=.02)
            for record in records[:3]:
                self.assertAlmostEqual(variants[id(record)] / draws, 1 / 18, delta=.012)
            self.assertAlmostEqual(views[0] / draws, .5, delta=.02)
        self.assertGreater(fractions[1] - fractions[0], .08)
        self.assertGreater(fractions[1], 1.7 * fractions[0])
        self.assertAlmostEqual(fractions[1], .5 * fractions[0] + .5 / 3, delta=.02)
        no_events = [record['name'] for record in records if len(record['short_events']) == 0]
        self.assertEqual(no_events, ['none', 'other-source'])

    def test_event_choice_uniform_across_events_not_halves_and_jitter_stays_inside_half(self):
        labels = np.full(5120, 2, dtype=np.int8)
        for start in (200, 500, 800, 3000):
            labels[start:start + 5] = 0
        records = sampling.prepare([label_record('match', labels, [(0, 1024), (1024, 5120)])], 'event_tcn', 384)
        grouped = b.original._balanced_index(records)
        rng = np.random.default_rng(160)
        counts = Counter()
        for _ in range(4000):
            record, start, _ = sampling.draw(records, grouped, rng, 384, 1)
            containing = [event for event in record['short_events'] if start <= event[2] and event[3] <= start + 384]
            self.assertTrue(containing)
            event = min(containing, key=lambda item: abs((int(item[2]) + int(item[3])) // 2 - (start + 192)))
            block_start, block_end, run_start, run_end = map(int, event)
            center = (run_start + run_end) // 2
            lower = max(block_start, min(center - 192 - 96, block_end - 384))
            upper = max(block_start, min(center - 192 + 96, block_end - 384))
            self.assertGreaterEqual(start, lower)
            self.assertLessEqual(start, upper)
            self.assertLessEqual(start + 384, block_end)
            counts[run_start] += 1
        for start in (200, 500, 800, 3000):
            self.assertAlmostEqual(counts[start] / 4000, .25, delta=.035)

    def test_no_event_match_retains_baseline_window_without_fake_event(self):
        records = [record for record in self.records() if record['name'] == 'none']
        grouped = b.original._balanced_index(records)
        actual_rng = np.random.default_rng(67)
        expected_rng = np.random.default_rng(67)
        for _ in range(100):
            expected = b.original._sample_record(records, grouped, expected_rng, 384)
            expected_rng.random()
            actual = sampling.draw(records, grouped, actual_rng, 384, .5)
            self.assertIs(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])
        np.testing.assert_array_equal(actual_rng.random(12), expected_rng.random(12))


class BatchConsumers(unittest.TestCase):
    def test_non_event_batches_match_original_mask_noise_pp_and_rng(self):
        source = feature_record()
        mean = np.linspace(-2, 2, 23, dtype=np.float32)
        scale = np.linspace(.5, 2, 23, dtype=np.float32)
        for family in ('spell_balanced_tcn', 'dual_scale_tcn'):
            records = sampling.prepare([source], family, 384)
            grouped = b.original._balanced_index(records)
            for seed, mask_full, noise in ((0, .3, .5), (3, 1., 0.), (7, 0., .5)):
                with self.subTest(family=family, seed=seed):
                    config = dict(name='pp_tcn', batch=8, crop=384, mask_full=mask_full, noise=noise)
                    rng = np.random.default_rng(seed)
                    original_rng = np.random.default_rng(seed)
                    actual = sampling.make_batch(records, grouped, mean, scale, config, rng, family)
                    expected = b.original._make_batch(records, grouped, mean, scale, config, original_rng)
                    for result, original in zip(actual[:5], expected):
                        np.testing.assert_array_equal(result, original)
                    np.testing.assert_array_equal(rng.random(10), original_rng.random(10))
                    self.assertEqual(actual[5].dtype, np.dtype(np.float32))
                    self.assertTrue(np.isfinite(actual[0]).all())
        self.assertTrue(np.isnan(source['player']).any())
        self.assertTrue(np.isnan(source['ball']).any())
        np.testing.assert_array_equal(source['pp'][..., 6:9], np.broadcast_to(np.array([3, 4])[:, None, None, None], source['pp'][..., 6:9].shape))

    def test_batches_align_full_weights_labels_views_and_masked_features_inside_half(self):
        sources = [feature_record('first', 0), feature_record('second', 1)]
        for family in ('spell_balanced_tcn', 'event_tcn', 'dual_scale_tcn'):
            with self.subTest(family=family):
                records = sampling.prepare(sources, family, 384)
                grouped = b.original._balanced_index(records)
                config = dict(name='pp_tcn', batch=32, crop=384, mask_full=.3, noise=0.)
                x, pp, ppm, y, views, frame_weights = sampling.make_batch(
                    records, grouped, np.zeros(23, dtype=np.float32), np.ones(23, dtype=np.float32),
                    config, np.random.default_rng(981), family,
                )
                halves = set()
                for index, view in enumerate(views):
                    start = int(x[index, 0, 0])
                    end = start + 384
                    identity = int(x[index, 2, 0])
                    record = records[identity]
                    self.assertTrue(any(left <= start and end <= right for left, right in record['blocks']))
                    halves.add(start // 768)
                    np.testing.assert_array_equal(x[index, 0], np.arange(start, end))
                    np.testing.assert_array_equal(x[index, 1], view)
                    np.testing.assert_array_equal(x[index, 23], record['valid'][view, start:end])
                    np.testing.assert_array_equal(y[index], record['y'][start:end])
                    np.testing.assert_array_equal(frame_weights[index], record['weights'][start:end])
                    np.testing.assert_array_equal(ppm[index], record['ppm'][view, start:end])
                    masked = x[index, 28] == 0
                    expected_ball = np.nan_to_num(record['ball'][view, start:end].astype(np.float32))
                    expected_ball[masked] = 0
                    np.testing.assert_array_equal(x[index, 24:29], expected_ball.T)
                    expected_pp = record['pp'][view, start:end].copy()
                    expected_pp[masked, :, 6:9] = 0
                    np.testing.assert_array_equal(pp[index], expected_pp)
                    long_spell = (np.arange(start, end) >= 200) & (np.arange(start, end) < 700)
                    if family == 'spell_balanced_tcn':
                        np.testing.assert_array_equal(frame_weights[index, long_spell], .25)
                self.assertEqual(halves, {0, 1})
                self.assertEqual(set(views), {0, 1})


if __name__ == '__main__':
    unittest.main()
