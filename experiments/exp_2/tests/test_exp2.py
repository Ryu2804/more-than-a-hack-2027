"""Unit tests on synthetic arrays only (no match data)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp2_common  # noqa: E402,F401  (puts ../exp_1 on sys.path)
import exp1_decode as D  # noqa: E402
import exp2_hsmm as H  # noqa: E402
import exp2_protocol as P  # noqa: E402
import exp2_views as V  # noqa: E402
from exp1_common import Match  # noqa: E402

LENGTH, WIDTH = 105.0, 68.0
N_FRAMES = 4001
SPLIT = 2000  # frame 2000 is the ungraded row between halves


def synthetic_match(seed: int = 0) -> Match:
    rng = np.random.default_rng(seed)
    half = np.r_[np.ones(SPLIT), [0], np.full(N_FRAMES - SPLIT - 1, 2)].astype("int8")
    players = np.empty((N_FRAMES, 22, 3), dtype="float32")
    players[:, :, 0] = rng.uniform(5, LENGTH - 5, (N_FRAMES, 22))
    players[:, :, 1] = rng.uniform(5, WIDTH - 5, (N_FRAMES, 22))
    players[:, :, 2] = rng.uniform(1.5, 3.5, (N_FRAMES, 22))
    players[200:500, :, 2] = 0.3  # long slow run (300 frames)
    players[700:760, :, 2] = 0.3  # short slow run (60 frames)
    ball = np.column_stack([rng.uniform(10, LENGTH - 10, N_FRAMES),
                            rng.uniform(10, WIDTH - 10, N_FRAMES)]).astype("float32")
    ball[1000:1100, 0] = -2.0  # long off-pitch run
    ball[1500:1520, 1] = WIDTH + 3.0  # short off-pitch run
    ball[2600:2700, 0] = LENGTH + 4.0  # off-pitch run in the second half
    ball[1300:1350] = np.nan
    ball[SPLIT] = np.nan
    labels = np.zeros(N_FRAMES, dtype="int8")
    labels[600:1200] = 2  # long dead spell
    labels[1250:1330] = 2  # short dead spell (80 frames)
    labels[1400:1900] = 1
    labels[2500:2900] = 2
    labels[SPLIT] = 3
    teams = np.r_[np.zeros(11), np.ones(11)].astype("int8")
    return Match("synthetic", players, ball, teams, half, np.arange(N_FRAMES), labels, (LENGTH, WIDTH))


def _same_ball(a: Match, b: Match) -> bool:
    return bool(np.array_equal(a.ball, b.ball, equal_nan=True))


# ---------------------------------------------------------------- target_like view
def test_target_like_is_label_independent():
    match = synthetic_match()
    shuffled = replace(match, labels=np.random.default_rng(1).permutation(match.labels))
    assert _same_ball(V.target_like(match), V.target_like(shuffled))
    assert _same_ball(V.target_like(match), V.target_like(replace(match, labels=None)))


def test_target_like_keeps_only_first_k_out_frames_of_each_off_pitch_run():
    view = V.target_like(synthetic_match())
    visible = V.ball_visible(view)
    assert visible[1000:1000 + V.K_OUT].all()
    assert not visible[1000 + V.K_OUT:1100].any()
    assert visible[1500:1520].all()  # shorter than K_OUT: untouched
    assert not visible[2600 + V.K_OUT:2700].any()


def test_target_like_hides_long_slow_runs_but_not_short_ones():
    view = V.target_like(synthetic_match())
    visible = V.ball_visible(view)
    assert not visible[250:450].any()
    assert visible[700:760].all()
    assert visible[100:150].all()


def test_target_like_only_removes_ball_observations():
    match = synthetic_match()
    view = V.target_like(match)
    assert np.array_equal(view.players, match.players)
    assert np.array_equal(view.labels, match.labels)
    kept = V.ball_visible(view)
    assert np.array_equal(view.ball[kept], match.ball[kept])
    assert not (kept & ~V.ball_visible(match)).any()


def test_graded_runs_never_cross_halves():
    half = np.array([1, 1, 0, 2, 2], dtype="int8")
    mask = np.ones(5, dtype=bool)
    assert V.graded_runs(mask, half) == [(0, 2), (3, 5)]


@pytest.mark.parametrize("transform", ["target_like", "target_aug"])
def test_views_commute_with_home_away_mirror(transform):
    match = synthetic_match(3)

    def apply(m: Match) -> Match:
        if transform == "target_like":
            return V.target_like(m)
        return V.target_aug(m, np.random.default_rng(7))

    mirrored_then_viewed = apply(V.mirror_match(match))
    viewed_then_mirrored = V.mirror_match(apply(match))
    assert _same_ball(mirrored_then_viewed, viewed_then_mirrored)


def test_mirror_match_is_an_involution_and_swaps_labels():
    match = synthetic_match()
    twice = V.mirror_match(V.mirror_match(match))
    assert np.allclose(twice.players, match.players, equal_nan=True)
    assert np.array_equal(twice.labels, match.labels)
    once = V.mirror_match(match)
    assert np.array_equal(once.labels[1400:1900], np.zeros(500))


# ---------------------------------------------------------------- target_aug training transform
def test_target_aug_hides_inside_long_dead_spells_only(monkeypatch):
    monkeypatch.setattr(V, "AUG_DEAD_PROB", 1.0)
    match = synthetic_match()
    on_pitch = np.where(V.outside_mask(match)[:, None], 50.0, match.ball).astype("float32")
    augmented = V.target_aug(replace(match, ball=on_pitch), np.random.default_rng(0))
    visible = V.ball_visible(augmented)
    margin = V.AUG_DEAD_MARGIN
    assert visible[600:600 + margin].all() and visible[1200 - margin:1200].all()
    assert not visible[600 + margin:1200 - margin].any()
    assert visible[1250:1300].all()  # dead spell shorter than AUG_DEAD_MIN
    assert visible[100:200].all()  # alive frames untouched


def test_target_aug_keeps_random_prefix_of_off_pitch_runs(monkeypatch):
    monkeypatch.setattr(V, "AUG_DEAD_PROB", 0.0)
    augmented = V.target_aug(synthetic_match(), np.random.default_rng(5))
    kept = int(V.ball_visible(augmented)[1000:1100].sum())
    assert 0 <= kept <= V.AUG_K_MAX
    assert V.ball_visible(augmented)[1000:1000 + kept].all()


def test_target_aug_is_deterministic_and_needs_labels():
    match = synthetic_match()
    a = V.target_aug(match, np.random.default_rng(V.aug_seed("J03WMX", 1)))
    b = V.target_aug(match, np.random.default_rng(V.aug_seed("J03WMX", 1)))
    assert _same_ball(a, b)
    with pytest.raises(ValueError):
        V.target_aug(replace(match, labels=None), np.random.default_rng(0))


# ---------------------------------------------------------------- explicit-duration decoder
def _probabilities(labels: np.ndarray, confidence: float = 0.8) -> np.ndarray:
    p = np.full((len(labels), 3), (1 - confidence) / 2)
    p[np.arange(len(labels)), labels] = confidence
    return p


def _runs(codes: np.ndarray) -> int:
    return int(1 + np.sum(codes[1:] != codes[:-1]))


def test_zero_duration_weight_equals_constant_penalty_viterbi():
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(3), size=600)
    half = np.r_[np.ones(300), np.full(300, 2)].astype("int8")
    prior = H.DurationPrior(3.0, 1.0, 4.5, 0.9)
    for lam_team, lam_dead in ((0.5, 1.0), (2.0, 0.0), (4.0, 4.0)):
        hsmm = H.decode(p, half, prior, H.HsmmConfig(1.3, 0.0, lam_team, lam_dead))
        viterbi = D.decode(p, None, half, D.DecoderConfig(1.3, lam_team, lam_dead, 0.0))
        assert np.array_equal(hsmm, viterbi)


def test_spells_longer_than_dmax_are_decoded_by_continuation():
    logp = np.log(_probabilities(np.zeros(57, dtype=int), 0.9))
    table = np.zeros((3, 11))
    table[:, 0] = -np.inf
    path = H._hsmm(logp, table, 1.0, 1.0)
    assert np.all(path == 0)


def test_short_same_state_pieces_cannot_dodge_the_duration_prior():
    p = np.tile([0.45, 0.44, 0.11], (200, 1))
    half = np.ones(200, dtype="int8")
    tight = H.DurationPrior(np.log(20.0), 0.2, np.log(100.0), 0.5)
    flat = H.decode(p, half, tight, H.HsmmConfig(1.0, 0.0, 0.0, 2.0))
    shaped = H.decode(p, half, tight, H.HsmmConfig(1.0, 1.0, 0.0, 2.0))
    assert _runs(flat) == 1
    assert _runs(shaped) > 3


def test_duration_prior_recovers_lognormal_parameters():
    rng = np.random.default_rng(0)
    team = np.maximum(1, np.round(rng.lognormal(3.0, 0.8, 600))).astype(int)
    dead = np.maximum(1, np.round(rng.lognormal(4.5, 0.5, 600))).astype(int)
    pieces = [np.full(d, code) for pair in zip(team, dead) for d, code in zip(pair, (0, 2))]
    state = np.concatenate(pieces).astype("int8")
    prior = H.estimate_prior([(state, np.ones(len(state), dtype="int8"))])
    assert abs(prior.team_mu - 3.0) < 0.15 and abs(prior.team_sigma - 0.8) < 0.15
    assert abs(prior.dead_mu - 4.5) < 0.1 and abs(prior.dead_sigma - 0.5) < 0.1


def test_run_lengths_break_on_unlabelled_and_halves():
    state = np.array([0, 0, 3, 0, 1, 1, 2, 2, 2], dtype="int8")
    half = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2], dtype="int8")
    team, dead = H.run_lengths(state, half)
    assert sorted(team) == [1, 1, 1, 2]
    assert dead == [3]


def test_duration_term_is_entropy_centred_and_weighted():
    prior = H.DurationPrior(3.0, 1.0, 4.5, 0.9)
    table = H.log_duration_table(prior, 1.0, d_max=2000)
    for state, (mu, sigma) in ((0, (3.0, 1.0)), (2, (4.5, 0.9))):
        pmf = np.exp(H._log_pmf(mu, sigma, 2000))
        assert np.isclose(pmf.sum(), 1.0)
        assert abs(float(np.sum(pmf * table[state, 1:]))) < 1e-9  # zero mean under P
    half_weight = H.log_duration_table(prior, 0.5, d_max=2000)
    assert np.allclose(half_weight[2, 1:], 0.5 * table[2, 1:])


# ---------------------------------------------------------------- protocol helpers
def test_ensemble_averages_probabilities_and_boundaries():
    def member(value: float) -> dict:
        return {"outer_x": {"original": {"m": (np.full((4, 3), value), np.full(4, value))}}}

    out = P.ensemble_predictions({"a": member(0.1), "b": member(0.3), "c": member(0.5)})
    probability, boundary = out["outer_x"]["original"]["m"]
    assert np.allclose(probability, 0.3) and np.allclose(boundary, 0.3)


def test_prediction_cache_key_depends_on_seed_and_recipe():
    keys = {arm: P.fingerprint(arm) for arm in P.MODEL_ARMS}
    assert len(set(keys.values())) == len(keys)
