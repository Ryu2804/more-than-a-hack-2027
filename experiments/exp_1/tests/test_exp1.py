"""Unit tests on synthetic arrays only (no match data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp1_arms as A  # noqa: E402
import exp1_decode as D  # noqa: E402
import exp1_tokens as T  # noqa: E402
from exp1_common import expand  # noqa: E402


def _probabilities(labels: np.ndarray, confidence: float = 0.8) -> np.ndarray:
    p = np.full((len(labels), 3), (1 - confidence) / 2)
    p[np.arange(len(labels)), labels] = confidence
    return p


def test_huge_penalty_gives_one_state_per_period():
    labels = np.array([0, 1, 0, 1, 2, 0] * 10)
    half = np.r_[np.ones(30), np.full(30, 2)].astype("int8")
    config = D.DecoderConfig(1.0, 1e6, 1e6, 0.0)
    codes = D.decode(_probabilities(labels), None, half, config)
    for period in (1, 2):
        assert len(np.unique(codes[half == period])) == 1


def test_decoder_accepts_float32_model_outputs():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(3), size=40).astype("float32")
    boundary = rng.random(40).astype("float32")
    codes = D.decode(p, boundary, np.ones(40, dtype="int8"), D.DecoderConfig(1.0, 2.0, 2.0, 1.0))
    assert codes.dtype == np.int8 and len(codes) == 40


def test_zero_penalty_equals_argmax():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=200)
    half = np.ones(200, dtype="int8")
    codes = D.decode(p, None, half, D.DecoderConfig(1.0, 0.0, 0.0, 0.0))
    assert np.array_equal(codes, p.argmax(axis=1))


def test_boundary_probability_unlocks_switch_where_constant_penalty_blocks_it():
    # 4 steps of team 1 gain 4 * log(0.7/0.15) = 6.2; two switches at lambda 4 cost 8.
    labels = np.r_[np.zeros(20, int), np.ones(4, int), np.zeros(20, int)]
    p = _probabilities(labels, 0.7)
    half = np.ones(len(labels), dtype="int8")
    boundary = np.full(len(labels), 0.01)
    boundary[[20, 24]] = 0.99
    constant = D.decode(p, boundary, half, D.DecoderConfig(1.0, 4.0, 4.0, 0.0))
    aware = D.decode(p, boundary, half, D.DecoderConfig(1.0, 4.0, 4.0, 1.0))
    assert np.all(constant == 0)
    assert np.array_equal(aware, labels)


def test_centred_decoder_with_zero_gamma_equals_constant_decoder():
    import exp1_followup as F

    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(3), size=300)
    boundary = rng.random(300)
    half = np.r_[np.ones(150), np.full(150, 2)].astype("int8")
    config = D.DecoderConfig(1.2, 2.0, 4.0, 0.0)
    assert np.array_equal(F.decode_centred(p, boundary, half, config), D.decode(p, boundary, half, config))


def test_centred_boundary_subsidises_switch_at_high_boundary():
    import exp1_followup as F

    labels = np.r_[np.zeros(20, int), np.ones(4, int), np.zeros(20, int)]
    p = _probabilities(labels, 0.7)
    half = np.ones(len(labels), dtype="int8")
    boundary = np.full(len(labels), 0.02)
    boundary[[20, 24]] = 0.9
    constant = F.decode_centred(p, boundary, half, D.DecoderConfig(1.0, 4.0, 4.0, 0.0))
    centred = F.decode_centred(p, boundary, half, D.DecoderConfig(1.0, 4.0, 4.0, 1.0))
    assert np.all(constant == 0)
    assert np.array_equal(centred, labels)


def test_touch_hold_carries_last_touch_and_merges_short_runs():
    dmin = np.full(100, 5.0)
    team = np.zeros(100, dtype="int8")
    dmin[0], team[0] = 0.5, 0
    dmin[50], team[50] = 0.5, 1
    dmin[60], team[60] = 0.5, 0  # 10-frame segment of team 1 -> merged away
    out = A.touch_hold(dmin, team)
    assert np.all(out == 0)


def test_merge_short_dead_only_removes_short_runs():
    series = np.array([0] * 10 + [2] * 5 + [0] * 10 + [2] * 200 + [1] * 5, dtype="int8")
    out = A.merge_short_dead(series, 150)
    assert np.all(out[10:15] == 0)
    assert np.all(out[25:225] == 2)


def test_mirror_is_involution_and_swaps_home_away():
    rng = np.random.default_rng(1)
    tokens = rng.normal(size=(5, T.MAX_TOKENS, len(T.TOKEN_NAMES))).astype("float32")
    glob = rng.normal(size=(5, len(T.GLOBAL_NAMES))).astype("float32")
    state = np.array([0, 1, 2, 3, 0], dtype="int8")
    m_tokens, m_glob, m_state = T.mirror(tokens, glob, state)
    assert np.array_equal(m_state, [1, 0, 2, 3, 1])
    assert np.allclose(m_glob[:, 8], glob[:, 9])
    back = T.mirror(m_tokens, m_glob, m_state)
    assert np.array_equal(back[0], tokens) and np.array_equal(back[1], glob)


def test_expand_forward_fills_and_marks_ungraded():
    half = np.array([1] * 10 + [0] + [2] * 10, dtype="int8")
    indices = np.r_[np.arange(0, 10, 5), np.arange(11, 21, 5)]
    codes = np.array([0, 1, 2, 0], dtype="int8")
    full = expand(codes, indices, half)
    assert full[10] == 3
    assert np.array_equal(full[:10], [0] * 5 + [1] * 5)
    assert np.array_equal(full[11:], [2] * 5 + [0] * 5)


@pytest.mark.parametrize(
    "labels,expected",
    [([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]), ([0, 3, 1, 1, 1], [0, 0, 0, 0, 0])],
)
def test_step_boundaries_ignore_unlabelled_and_dilate(labels, expected):
    token_set = T.TokenSet(tokens=np.zeros((5, 1, 1)), mask=np.ones((5, 1), bool),
                           glob=np.zeros((5, 1)), half=np.ones(5, dtype="int8"),
                           indices=np.arange(5))
    _, boundary = T.step_targets(np.array(labels, dtype="int8"), token_set)
    assert boundary.tolist() == expected
