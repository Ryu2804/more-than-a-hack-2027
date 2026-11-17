import unittest

import numpy as np
import torch
from torch.nn import functional as F

import weights


FAMILIES = ('spell_balanced_tcn', 'event_tcn', 'dual_scale_tcn')


class FullMatchWeights(unittest.TestCase):
    def test_complete_run_duration_curve_and_clipping_without_normalization(self):
        labels = [2]
        spans = []
        for index, length in enumerate((1, 5, 20, 80, 500)):
            start = len(labels)
            labels.extend([index % 2] * length)
            spans.append((start, len(labels), length))
            labels.append(2)
        labels = np.asarray(labels, dtype=np.int8)
        labels.setflags(write=False)
        blocks = np.array([[0, len(labels)]])
        blocks.setflags(write=False)
        actual = weights.spell_weights(labels, blocks)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        for start, end, length in spans:
            np.testing.assert_allclose(actual[start:end], np.clip(np.sqrt(20 / length), .25, 4))
        np.testing.assert_array_equal(actual[labels == 2], 1)
        self.assertGreater(actual[1], actual[3])
        self.assertLess(actual[-2], 1)

    def test_half_edges_and_unknown_adjacent_spells_are_censored(self):
        labels = np.array([0, 0, 2, 1, 2, 0, 0, 0, 2, 1, 2, 1, -1, 0, 2, 1], dtype=np.int8)
        before = labels.copy()
        actual = weights.spell_weights(labels, [(0, 6), (6, 16)])
        expected = np.ones(16, dtype=np.float32)
        expected[[3, 9]] = 4
        expected[12] = 0
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(labels, before)

    def test_long_spell_crop_keeps_full_match_duration_weight(self):
        labels = np.array([2] * 30 + [0] * 800 + [2] * 30, dtype=np.int8)
        actual = weights.spell_weights(labels, [(0, len(labels))])
        np.testing.assert_array_equal(actual[200:584], np.full(384, .25, dtype=np.float32))
        np.testing.assert_array_equal(actual[:30], 1)

    def test_opponent_change_completes_both_runs_but_unknown_break_does_not(self):
        labels = np.array([2, 0, 0, 1, 1, 1, 2, 0, -1, 1, 2])
        actual = weights.spell_weights(labels, [(0, len(labels))])
        np.testing.assert_allclose(actual[1:3], np.sqrt(10), rtol=1e-6)
        np.testing.assert_allclose(actual[3:6], np.sqrt(20 / 3), rtol=1e-6)
        np.testing.assert_array_equal(actual[7:10], [1, 0, 1])


class StateLoss(unittest.TestCase):
    def setUp(self):
        self.y = torch.tensor([[0, 1, 2, -1], [1, 0, -1, 2]])
        self.views = torch.tensor([0, 1])
        self.logits = torch.tensor([
            [[-.8, .4, 1.2, -.2], [.1, -.3, .7, 1.4]],
            [[.5, -.9, -.1, .6], [-.4, .8, 1.1, -.5]],
        ], dtype=torch.float64)
        self.frame_weights = torch.tensor([[.25, 4, 1, 9], [2, .5, 7, 1]], dtype=torch.float64)

    def test_numerics_and_analytic_gradient_masks_for_every_family(self):
        known = (self.y >= 0).double()
        live = ((self.y == 0) | (self.y == 1)).double()
        target_dead = (self.y == 2).double()
        target_team = (self.y == self.views[:, None]).double()
        dead_bce = F.binary_cross_entropy_with_logits(self.logits[:, 0], target_dead, reduction='none')
        team_bce = F.binary_cross_entropy_with_logits(self.logits[:, 1], target_team, reduction='none')
        for family in FAMILIES:
            with self.subTest(family=family):
                logits = self.logits.clone().requires_grad_()
                loss = weights.state_loss(logits, self.y, self.views, self.frame_weights, family)
                if family == 'spell_balanced_tcn':
                    dead_factor = known / known.sum()
                    team_factor = live * self.frame_weights / (live * self.frame_weights).sum()
                else:
                    dead_factor = known * self.frame_weights / known.sum()
                    team_factor = live * self.frame_weights / live.sum()
                expected = (dead_bce * dead_factor).sum() + (team_bce * team_factor).sum()
                torch.testing.assert_close(loss, expected)
                loss.backward()
                expected_gradient = torch.stack((
                    (logits[:, 0].detach().sigmoid() - target_dead) * dead_factor,
                    (logits[:, 1].detach().sigmoid() - target_team) * team_factor,
                ), dim=1)
                torch.testing.assert_close(logits.grad, expected_gradient)
                self.assertTrue(torch.equal(logits.grad[:, 0][self.y == -1], torch.zeros(2, dtype=torch.float64)))
                self.assertTrue(torch.equal(logits.grad[:, 1][live == 0], torch.zeros(4, dtype=torch.float64)))

    def test_unknown_logit_values_and_weights_cannot_change_known_loss(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                original = weights.state_loss(self.logits, self.y, self.views, self.frame_weights, family)
                changed = self.logits.clone()
                changed.masked_fill_((self.y == -1)[:, None, :], 50)
                changed[:, 1][self.y == 2] = -50
                frame_weights = self.frame_weights.clone()
                frame_weights[self.y == -1] = 12345
                actual = weights.state_loss(changed, self.y, self.views, frame_weights, family)
                torch.testing.assert_close(actual, original)

    def test_all_unknown_and_no_live_have_finite_zero_masked_gradient(self):
        for family in FAMILIES:
            for state in (-1, 2):
                with self.subTest(family=family, state=state):
                    y = torch.full((2, 5), state)
                    logits = torch.zeros((2, 2, 5), requires_grad=True)
                    frame_weights = torch.zeros((2, 5)) if state == -1 else torch.ones((2, 5))
                    loss = weights.state_loss(logits, y, torch.tensor([0, 1]), frame_weights, family)
                    self.assertTrue(torch.isfinite(loss))
                    if state == -1:
                        self.assertEqual(loss.item(), 0)
                    else:
                        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
                    loss.backward()
                    np.testing.assert_array_equal(logits.grad[:, 1].numpy(), 0)
                    if state == -1:
                        np.testing.assert_array_equal(logits.grad.numpy(), 0)

    def test_view_flip_and_team_relabeling_symmetry(self):
        swapped_y = torch.where(self.y < 2, 1 - self.y, self.y)
        swapped_y[self.y == -1] = -1
        for family in FAMILIES:
            with self.subTest(family=family):
                expected = weights.state_loss(self.logits, self.y, self.views, self.frame_weights, family)
                actual = weights.state_loss(self.logits, swapped_y, 1 - self.views, self.frame_weights, family)
                torch.testing.assert_close(actual, expected)
                swapped_logits = self.logits.clone()
                swapped_logits[:, 1] *= -1
                actual = weights.state_loss(swapped_logits, self.y, 1 - self.views, self.frame_weights, family)
                torch.testing.assert_close(actual, expected)

    def test_spell_team_normalization_is_scale_invariant_even_below_one(self):
        for factor in (.001, 3, 100):
            logits = self.logits.clone().requires_grad_()
            original = self.logits.clone().requires_grad_()
            actual = weights.state_loss(logits, self.y, self.views, self.frame_weights * factor, 'spell_balanced_tcn')
            expected = weights.state_loss(original, self.y, self.views, self.frame_weights, 'spell_balanced_tcn')
            torch.testing.assert_close(actual, expected)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(logits.grad, original.grad)
        logits = torch.zeros((1, 2, 1), requires_grad=True)
        loss = weights.state_loss(logits, torch.tensor([[0]]), torch.tensor([0]), torch.tensor([[.25]]), 'spell_balanced_tcn')
        self.assertAlmostEqual(loss.item(), 2 * np.log(2), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[[.5], [-.5]]]))

    def test_short_families_use_unweighted_counts_for_both_heads(self):
        logits = torch.zeros((1, 2, 4), dtype=torch.float64)
        y = torch.tensor([[0, 1, 2, -1]])
        frame_weights = torch.tensor([[2., 1., 1., 100.]], dtype=torch.float64)
        expected = np.log(2) * (4 / 3 + 3 / 2)
        for family in ('event_tcn', 'dual_scale_tcn'):
            with self.subTest(family=family):
                loss = weights.state_loss(logits, y, torch.tensor([0]), frame_weights, family)
                self.assertAlmostEqual(loss.item(), expected, places=12)


if __name__ == '__main__':
    unittest.main()
