import unittest

import torch
from torch import nn
from torch.nn import functional as F

import architecture
import bridge as b


FAMILIES = ("spell_balanced_tcn", "event_tcn", "dual_scale_tcn")
CONFIG = dict(name="pp_tcn", ch=8, dilations=(1, 2, 4), dropout=0.15)


def reference_conv(x, weight, bias, dilation):
    length = x.shape[-1]
    padded = F.pad(x, (2 * dilation, 2 * dilation))
    taps = torch.stack(
        [padded[..., k * dilation:k * dilation + length] for k in range(5)],
        dim=-1,
    )
    result = torch.einsum("bctk,ock->bot", taps, weight)
    if bias is not None:
        result = result + bias[None, :, None]
    return result


class ReferenceScaleConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        conv = self.conv
        return 0.5 * (
            reference_conv(x, conv.weight, conv.bias, conv.dilation[0])
            + reference_conv(x, conv.weight, conv.bias, 1)
        )


def inputs(length=11, dtype=torch.float32):
    generator = torch.Generator().manual_seed(87 + length)
    x = torch.randn(2, 29, length, generator=generator, dtype=dtype)
    pp = torch.randn(2, length, 48, 10, generator=generator, dtype=dtype)
    ppm = torch.rand(2, length, 48, generator=generator) > 0.35
    ppm[:, ::3] = False
    return x, pp, ppm


class SharedScaleMath(unittest.TestCase):
    def test_conv_values_and_all_gradients_match_independent_tap_sum(self):
        for dilation in (2, 4):
            for length in (1, 7, 19):
                for with_bias in (False, True):
                    with self.subTest(dilation=dilation, length=length, bias=with_bias):
                        conv = nn.Conv1d(
                            2, 3, 5, dilation=dilation,
                            padding=2 * dilation, bias=with_bias,
                        ).double()
                        with torch.no_grad():
                            conv.weight.copy_(
                                torch.linspace(-0.7, 0.9, 30, dtype=torch.float64)
                                .reshape_as(conv.weight)
                            )
                            if with_bias:
                                conv.bias.copy_(torch.tensor([0.3, -0.2, 0.8]))
                        wrapped = architecture.SharedScaleConv(conv)
                        x = torch.linspace(
                            -1.2, 1.5, 4 * length, dtype=torch.float64,
                        ).reshape(2, 2, length).requires_grad_()
                        reference_x = x.detach().clone().requires_grad_()
                        reference_weight = conv.weight.detach().clone().requires_grad_()
                        reference_bias = (
                            conv.bias.detach().clone().requires_grad_() if with_bias else None
                        )
                        expected = 0.5 * (
                            reference_conv(reference_x, reference_weight, reference_bias, dilation)
                            + reference_conv(reference_x, reference_weight, reference_bias, 1)
                        )
                        actual = wrapped(x)
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                        coefficients = torch.linspace(
                            -0.5, 1.3, actual.numel(), dtype=torch.float64,
                        ).reshape_as(actual)
                        actual_loss = actual.square().sum() + (actual * coefficients).sum()
                        expected_loss = expected.square().sum() + (expected * coefficients).sum()
                        actual_leaves = (x, conv.weight) + ((conv.bias,) if with_bias else ())
                        reference_leaves = (reference_x, reference_weight) + (
                            (reference_bias,) if with_bias else ()
                        )
                        actual_gradients = torch.autograd.grad(actual_loss, actual_leaves)
                        expected_gradients = torch.autograd.grad(expected_loss, reference_leaves)
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients,
                        ):
                            torch.testing.assert_close(
                                actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11,
                            )

    def test_unit_dilation_preserves_original_values_and_gradients_exactly(self):
        torch.manual_seed(4)
        conv = nn.Conv1d(2, 3, 5, padding=2).double()
        wrapped = architecture.SharedScaleConv(conv)
        x = torch.randn(2, 2, 9, dtype=torch.float64, requires_grad=True)
        actual = wrapped(x)
        expected = conv(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        leaves = (x, conv.weight, conv.bias)
        actual_gradients = torch.autograd.grad(actual.square().sum(), leaves)
        expected_gradients = torch.autograd.grad(expected.square().sum(), leaves)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        self.assertIs(wrapped.conv.weight, conv.weight)
        self.assertIs(wrapped.conv.bias, conv.bias)


class WholeModelBehavior(unittest.TestCase):
    def test_no_extra_parameters_or_random_initialization(self):
        torch.manual_seed(23)
        original = b.original._make_model(CONFIG)
        expected_rng = torch.get_rng_state()
        original_parameters = list(original.parameters())
        for family in FAMILIES:
            with self.subTest(family=family):
                torch.manual_seed(23)
                model = architecture.make_model(CONFIG, family)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                registered = list(model.named_parameters(remove_duplicate=False))
                self.assertEqual(len(registered), len(original_parameters))
                self.assertEqual(len({id(value) for _, value in registered}), len(registered))
                self.assertEqual(
                    sum(value.numel() for _, value in registered),
                    sum(value.numel() for value in original_parameters),
                )
                for (_, actual), expected in zip(registered, original_parameters):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nondual_families_preserve_original_prediction(self):
        x, pp, ppm = inputs()
        torch.manual_seed(37)
        original = b.original._make_model(CONFIG).eval()
        with torch.no_grad():
            expected = original(x, pp, ppm)
            for family in FAMILIES[:2]:
                with self.subTest(family=family):
                    torch.manual_seed(37)
                    model = architecture.make_model(CONFIG, family).eval()
                    torch.testing.assert_close(model(x, pp, ppm), expected, rtol=0, atol=0)

    def test_whole_dual_model_matches_independent_math_and_gradients(self):
        torch.manual_seed(51)
        actual_model = architecture.make_model(CONFIG, "dual_scale_tcn").double().eval()
        torch.manual_seed(51)
        expected_model = b.original._make_model(CONFIG).double().eval()
        for block in expected_model.blocks:
            if block[0].dilation != (1,):
                block[0] = ReferenceScaleConv(block[0])
        x, pp, ppm = inputs(dtype=torch.float64)
        actual_x = x.clone().requires_grad_()
        actual_pp = pp.clone().requires_grad_()
        expected_x = x.clone().requires_grad_()
        expected_pp = pp.clone().requires_grad_()
        actual = actual_model(actual_x, actual_pp, ppm)
        expected = expected_model(expected_x, expected_pp, ppm)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        coefficients = torch.linspace(-0.8, 1.4, actual.numel(), dtype=actual.dtype).reshape_as(actual)
        actual_gradients = torch.autograd.grad(
            (actual * coefficients).sum() + actual.square().mean(),
            (actual_x, actual_pp, *actual_model.parameters()),
        )
        expected_gradients = torch.autograd.grad(
            (expected * coefficients).sum() + expected.square().mean(),
            (expected_x, expected_pp, *expected_model.parameters()),
        )
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            actual_gradients[1][~ppm], torch.zeros_like(actual_gradients[1][~ppm]), rtol=0, atol=0,
        )

    def test_player_permutation_preserves_whole_model_prediction(self):
        x, pp, ppm = inputs()
        permutation = torch.randperm(48, generator=torch.Generator().manual_seed(9))
        for family in FAMILIES:
            with self.subTest(family=family):
                torch.manual_seed(61)
                model = architecture.make_model(CONFIG, family).eval()
                with torch.no_grad():
                    expected = model(x, pp, ppm)
                    permuted = model(x, pp[:, :, permutation], ppm[:, :, permutation])
                torch.testing.assert_close(permuted, expected, rtol=1e-5, atol=1e-6)

    def test_masked_slots_cannot_change_output_including_empty_players(self):
        x, pp, mixed_mask = inputs()
        for family in FAMILIES:
            torch.manual_seed(71)
            model = architecture.make_model(CONFIG, family).eval()
            for ppm in (mixed_mask, torch.zeros_like(mixed_mask)):
                with self.subTest(family=family, empty=not bool(ppm.any())):
                    changed = pp.clone()
                    changed[~ppm] = 1000.0
                    with torch.no_grad():
                        expected = model(x, pp, ppm)
                        actual = model(x, changed, ppm)
                    self.assertTrue(bool(torch.isfinite(expected).all()))
                    self.assertTrue(bool(torch.isfinite(actual).all()))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_timeline_shape_at_short_odd_and_padding_boundaries(self):
        for family in FAMILIES:
            torch.manual_seed(81)
            model = architecture.make_model(CONFIG, family).eval()
            for length in (1, 2, 3, 7, 8, 9, 17, 32):
                with self.subTest(family=family, length=length):
                    x, pp, ppm = inputs(length)
                    with torch.no_grad():
                        actual = model(x, pp, ppm)
                    self.assertEqual(tuple(actual.shape), (2, 2, length))
                    self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_training_backprop_with_dropout_is_deterministic_and_finite(self):
        results = []
        x, pp, ppm = inputs(13)
        for _ in range(2):
            torch.manual_seed(91)
            model = architecture.make_model(CONFIG, "dual_scale_tcn").train()
            train_x = x.clone().requires_grad_()
            train_pp = pp.clone().requires_grad_()
            torch.manual_seed(109)
            actual = model(train_x, train_pp, ppm)
            target = torch.linspace(-1.0, 1.0, actual.numel()).reshape_as(actual)
            (actual - target).square().mean().backward()
            gradients = [train_x.grad, train_pp.grad] + [p.grad for p in model.parameters()]
            for gradient in gradients:
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(train_x.grad.abs().sum()), 0.0)
            self.assertGreater(float(train_pp.grad[ppm].abs().sum()), 0.0)
            torch.testing.assert_close(
                train_pp.grad[~ppm], torch.zeros_like(train_pp.grad[~ppm]), rtol=0, atol=0,
            )
            for block in model.blocks:
                conv = block[0].conv if isinstance(block[0], architecture.SharedScaleConv) else block[0]
                self.assertGreater(float(conv.weight.grad.abs().sum()), 0.0)
            results.append((actual.detach(), [gradient.detach().clone() for gradient in gradients]))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for first, second in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
