import unittest

import numpy as np

import core
import m06_short_guard
run = core.c.import_path('wave6_test_runner', core.HERE / 'run.py')


class CampaignTests(unittest.TestCase):
    def test_bootstrap_clusters_repeated_conditions_within_match(self):
        reference, candidate = [], []
        for provider in ('idsse', 'skillcorner', 'metrica'):
            for index, delta in enumerate((0.02, -0.005)):
                for regime in ('clean', 'degraded'):
                    row = dict(source=provider, match=provider + str(index), regime=regime, spell_f1_mean=0.6)
                    reference.append(row)
                    candidate.append({**row, 'spell_f1_mean': 0.6 + delta})
        actual = run.paired_interval(candidate, reference)
        repeated = [row for row in candidate if row['source'] == 'metrica']
        repeated_reference = [row for row in reference if row['source'] == 'metrica']
        duplicate = run.paired_interval(candidate + repeated, reference + repeated_reference)
        self.assertEqual(actual, duplicate)
        self.assertAlmostEqual(np.mean(actual['provider_deltas']), 0.0075)
        self.assertLess(actual['lower'], actual['upper'])

    def test_custom_postprocess_count_enters_lambda_selection(self):
        d = core.decoder
        team = np.repeat([0.2, 0.6, 0.3, 0.65, 0.15, 0.55, 0.2], [20, 8, 18, 7, 17, 9, 20])
        n = len(team)
        observations = dict(X=np.zeros((n, 16)), blocks=np.array([[0, n]]), distances=np.full((n, 2), 10.0))
        prediction = core.from_heads(np.zeros(n), team)
        dead = np.zeros(n, dtype=bool)
        home, away = d._team_evidence(team, observations['X'], observations, core.SPEC)
        candidates = []
        for lam in d.LAMBDA_GRID:
            codes = d._decode_live_team(home, away, observations['blocks'], dead, np.empty(0), lam)
            codes = d._merge_short_runs(codes, observations['blocks'], 5)
            guarded = m06_short_guard.guard(codes, observations, prediction, 2, 0.70)
            candidates.append((lam, d.n_possessions(guarded, observations['blocks']), guarded))
        for target in (1.0, 2.0, 3.0, 4.0, 5.0):
            with self.subTest(target=target):
                expected = min(candidates, key=lambda row: (abs(row[1] - target), row[0]))[2]
                stats = dict(rate=target * 300 / n, dead_share=0.0)
                actual = m06_short_guard.predict({'short': prediction}, observations, stats,
                    dict(max_s=2, confidence=0.70))
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
