import math
import unittest

import torch
from torch.nn import functional as F

import objectives


FAMILIES = ('balanced', 'focal')


def reference_loss(logits, labels, views, weights, family):
    dead_sum = team_sum = team_weight_sum = 0.0
    known_count = live_count = 0
    for batch, row in enumerate(labels.tolist()):
        for frame, label in enumerate(row):
            if label < 0:
                continue
            known_count += 1
            weight = weights[batch, frame].item()
            dead_logit = logits[batch, 0, frame].item()
            dead_bce = max(dead_logit, 0.0) - dead_logit * (label == 2) + math.log1p(math.exp(-abs(dead_logit)))
            dead_sum += dead_bce * (weight if family == 'focal' else 1.0)
            if label not in (0, 1):
                continue
            live_count += 1
            team_logit = logits[batch, 1, frame].item()
            signed_logit = team_logit if label == views[batch].item() else -team_logit
            team_bce = max(-signed_logit, 0.0) + math.log1p(math.exp(-abs(signed_logit)))
            if family == 'focal':
                team_bce *= (-math.expm1(-team_bce)) ** 2
            team_sum += team_bce * weight
            team_weight_sum += weight
    denominator = team_weight_sum if family == 'balanced' else live_count
    return dead_sum / max(known_count, 1) + team_sum / (denominator if denominator > 0 else 1)


class ConditionalObjectives(unittest.TestCase):
    def setUp(self):
        self.y = torch.tensor([[0, 1, 2, -1, 0], [1, 0, -1, 2, 1]])
        self.views = torch.tensor([0, 1])
        self.logits = torch.tensor([
            [[-.8, .4, 1.2, -.2, .9], [.1, -.3, .7, 1.4, -1.1]],
            [[.5, -.9, -.1, .6, -.7], [-.4, .8, 1.1, -.5, 1.3]],
        ], dtype=torch.float64)
        self.weights = torch.tensor([[.25, 4, 3, 9, 2], [2, .5, 7, 1, 2]], dtype=torch.float64)

    def test_scalar_matches_independent_stable_numeric_reference(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                actual = objectives.loss(self.logits, self.y, self.views, self.weights, family)
                self.assertEqual(actual.ndim, 0)
                self.assertAlmostEqual(actual.item(), reference_loss(
                    self.logits, self.y, self.views, self.weights, family,
                ), places=12)

    def test_analytic_gradients_include_focal_modulation_derivative(self):
        known = (self.y >= 0).double()
        live = ((self.y == 0) | (self.y == 1)).double()
        dead_target = (self.y == 2).double()
        team_target = (self.y == self.views[:, None]).double()
        team_bce = F.binary_cross_entropy_with_logits(self.logits[:, 1], team_target, reduction='none')
        probability_correct = torch.exp(-team_bce)
        for family in FAMILIES:
            with self.subTest(family=family):
                logits = self.logits.clone().requires_grad_()
                actual = objectives.loss(logits, self.y, self.views, self.weights, family)
                actual.backward()
                if family == 'balanced':
                    dead_factor = known / known.sum()
                    team_factor = live * self.weights / (live * self.weights).sum()
                else:
                    dead_factor = known * self.weights / known.sum()
                    modulation_derivative = (
                        (1 - probability_correct).square()
                        + 2 * team_bce * probability_correct * (1 - probability_correct)
                    )
                    team_factor = live * self.weights * modulation_derivative / live.sum()
                expected = torch.stack((
                    (self.logits[:, 0].sigmoid() - dead_target) * dead_factor,
                    (self.logits[:, 1].sigmoid() - team_target) * team_factor,
                ), dim=1)
                torch.testing.assert_close(logits.grad, expected, rtol=1e-12, atol=1e-12)
                self.assertTrue(torch.equal(logits.grad[:, 0][self.y < 0], torch.zeros(2, dtype=torch.float64)))
                self.assertTrue(torch.equal(logits.grad[:, 1][live == 0], torch.zeros(4, dtype=torch.float64)))

    def test_finite_difference_gradients(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                self.assertTrue(torch.autograd.gradcheck(
                    lambda logits: objectives.loss(logits, self.y, self.views, self.weights, family),
                    (self.logits.clone().requires_grad_(),),
                    eps=1e-6, atol=1e-6, rtol=1e-4,
                ))

    def test_unknown_and_dead_team_logits_do_not_affect_loss_or_gradient(self):
        changed = self.logits.clone()
        changed.masked_fill_((self.y < 0)[:, None, :], -10000)
        changed[:, 1][self.y == 2] = 10000
        weights = self.weights.clone()
        weights[self.y < 0] = 12345
        for family in FAMILIES:
            with self.subTest(family=family):
                original = self.logits.clone().requires_grad_()
                modified = changed.clone().requires_grad_()
                expected = objectives.loss(original, self.y, self.views, self.weights, family)
                actual = objectives.loss(modified, self.y, self.views, weights, family)
                torch.testing.assert_close(actual, expected)
                expected.backward()
                actual.backward()
                torch.testing.assert_close(modified.grad, original.grad)

    def test_focal_keeps_frozen_short_weighted_dead_loss_and_gradient(self):
        y = torch.tensor([[2, -1, 2, 2]])
        weights = torch.tensor([[1., 99., 2., .25]], dtype=torch.float64)
        source = torch.tensor([[[-2., 100., .4, 3.], [8., -9., 4., -5.]]], dtype=torch.float64)
        original = source.clone().requires_grad_()
        focal = source.clone().requires_grad_()
        expected = objectives.b.old_weights.state_loss(original, y, torch.tensor([0]), weights, family='event_tcn')
        actual = objectives.loss(focal, y, torch.tensor([0]), weights, 'focal')
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(focal.grad, original.grad)
        torch.testing.assert_close(focal.grad[:, 1], torch.zeros_like(focal.grad[:, 1]), rtol=0, atol=0)

    def test_all_unknown_and_no_home_away_are_finite(self):
        for family in FAMILIES:
            for label in (-1, 2):
                with self.subTest(family=family, label=label):
                    y = torch.full((2, 5), label)
                    logits = torch.zeros((2, 2, 5), dtype=torch.float64, requires_grad=True)
                    weights = torch.full((2, 5), 7. if label < 0 else 1., dtype=torch.float64)
                    actual = objectives.loss(logits, y, self.views, weights, family)
                    self.assertTrue(torch.isfinite(actual))
                    self.assertAlmostEqual(actual.item(), 0.0 if label < 0 else math.log(2), places=12)
                    actual.backward()
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    torch.testing.assert_close(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]), rtol=0, atol=0)
                    if label < 0:
                        torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad), rtol=0, atol=0)

    def test_extreme_logits_keep_finite_loss_and_gradient(self):
        source = torch.tensor([
            [[10000., -10000., -10000., 10000., -10000.], [-10000., -10000., 10000., -10000., 10000.]],
            [[-10000., 10000., -10000., 10000., 10000.], [-10000., 10000., -10000., 10000., 10000.]],
        ])
        for family in FAMILIES:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(family=family, dtype=dtype):
                    logits = source.to(dtype).clone().requires_grad_()
                    weights = self.weights.to(dtype)
                    actual = objectives.loss(logits, self.y, self.views, weights, family)
                    self.assertTrue(torch.isfinite(actual))
                    self.assertAlmostEqual(actual.item(), reference_loss(
                        logits, self.y, self.views, weights, family,
                    ), delta=.005 if dtype == torch.float32 else 1e-9)
                    actual.backward()
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    torch.testing.assert_close(logits.grad[:, 0][self.y < 0], torch.zeros(2, dtype=dtype), rtol=0, atol=0)
                    live = (self.y == 0) | (self.y == 1)
                    torch.testing.assert_close(logits.grad[:, 1][~live], torch.zeros(4, dtype=dtype), rtol=0, atol=0)

    def test_team_view_flip_preserves_loss_and_reverses_team_gradient(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                original = self.logits.clone().requires_grad_()
                flipped = self.logits.clone()
                flipped[:, 1].neg_()
                flipped.requires_grad_()
                expected = objectives.loss(original, self.y, self.views, self.weights, family)
                actual = objectives.loss(flipped, self.y, 1 - self.views, self.weights, family)
                torch.testing.assert_close(actual, expected)
                expected.backward()
                actual.backward()
                torch.testing.assert_close(flipped.grad[:, 0], original.grad[:, 0])
                torch.testing.assert_close(flipped.grad[:, 1], -original.grad[:, 1])
                relabeled = torch.where((self.y == 0) | (self.y == 1), 1 - self.y, self.y)
                relabeled_loss = objectives.loss(self.logits, relabeled, 1 - self.views, self.weights, family)
                torch.testing.assert_close(relabeled_loss, expected.detach())

    def test_balanced_uses_weight_mass_not_clamped_or_live_count(self):
        y = torch.tensor([[0, 1, 2, -1]])
        logits = torch.zeros((1, 2, 4), dtype=torch.float64)
        weights = torch.tensor([[.125, .25, 17., 99.]], dtype=torch.float64)
        for scale in (.001, 1., 100.):
            with self.subTest(scale=scale):
                actual = objectives.loss(logits, y, torch.tensor([0]), weights * scale, 'balanced')
                self.assertAlmostEqual(actual.item(), 2 * math.log(2), places=12)

    def test_focal_uses_short_weights_and_unweighted_denominators_without_alpha(self):
        y = torch.tensor([[0, 1, 2, -1]])
        logits = torch.zeros((1, 2, 4), dtype=torch.float64)
        weights = torch.tensor([[2., 1., 1., 100.]], dtype=torch.float64)
        expected = math.log(2) * (4 / 3 + .25 * 3 / 2)
        actual = objectives.loss(logits, y, torch.tensor([0]), weights, 'focal')
        self.assertAlmostEqual(actual.item(), expected, places=12)
        scaled = objectives.loss(logits, y, torch.tensor([0]), weights * 3, 'focal')
        torch.testing.assert_close(scaled, 3 * actual)

    def test_zero_live_weight_keeps_balanced_dead_loss_and_zero_team_gradient(self):
        y = torch.tensor([[0, 1]])
        for family in FAMILIES:
            with self.subTest(family=family):
                logits = torch.zeros((1, 2, 2), dtype=torch.float64, requires_grad=True)
                actual = objectives.loss(logits, y, torch.tensor([0]), torch.zeros((1, 2), dtype=torch.float64), family)
                self.assertAlmostEqual(actual.item(), math.log(2) if family == 'balanced' else 0., places=12)
                actual.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                torch.testing.assert_close(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]), rtol=0, atol=0)

    def test_focal_hard_examples_keep_gradient_while_confident_easy_are_suppressed(self):
        y = torch.tensor([[0, 0, 1, 1]])
        logits = torch.tensor([[[0., 0., 0., 0.], [-4., 4., 4., -4.]]], dtype=torch.float64, requires_grad=True)
        actual = objectives.loss(logits, y, torch.tensor([0]), torch.ones((1, 4), dtype=torch.float64), 'focal')
        actual.backward()
        gradient = logits.grad[0, 1]
        self.assertLess(gradient[0].item(), 0)
        self.assertLess(gradient[1].item(), 0)
        self.assertGreater(gradient[2].item(), 0)
        self.assertGreater(gradient[3].item(), 0)
        self.assertGreater(abs(gradient[0].item()), 1000 * abs(gradient[1].item()))
        torch.testing.assert_close(gradient[0], -gradient[2])
        torch.testing.assert_close(gradient[1], -gradient[3])
        bce_gradient = (logits.detach()[0, 1].sigmoid() - (y[0] == 0).double()) / 4
        self.assertGreater(abs(gradient[0].item()), abs(bce_gradient[0].item()))
        self.assertLess(abs(gradient[1].item()), .01 * abs(bce_gradient[1].item()))

    def test_inputs_remain_unchanged(self):
        inputs = (self.logits, self.y, self.views, self.weights)
        before = tuple(value.clone() for value in inputs)
        for family in FAMILIES:
            objectives.loss(*inputs, family)
            for actual, expected in zip(inputs, before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            objectives.loss(self.logits, self.y, self.views, self.weights, 'unsupported')


if __name__ == '__main__':
    unittest.main()
