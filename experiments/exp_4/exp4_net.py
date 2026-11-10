"""SetSeq with a training-only auxiliary hidden-ball head (BallRadar-style, Kim et al. 2023).

exp_1's network is not edited. ``SetSeqAux`` subclasses ``exp1_net.SetSeq``:

* the forward pass is exp_1's, unchanged (a forward hook only keeps the BiGRU output);
* the aux head is a linear map from that output to (ball x, ball y, logit home, logit away);
* the head is initialised from a forked CPU RNG, so the main parameters get exactly the
  values exp_1 would give them and the global RNG stream is untouched.

``train`` is ``exp1_net.train`` step for step (same seeds, RNG calls, batches, mirror, base
loss ``exp1_net._loss``, optimiser, schedule and clipping) plus one extra loss term when
the aux weights are > 0. At inference the aux head is unused (``exp1_net.predict`` works as is).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

import exp4_common  # noqa: F401,I001  (first: puts ../exp_3, ../exp_2, ../exp_1 on sys.path)
import exp1_net as N

IGNORE = N.IGNORE
AUX_OUTPUTS = ("ball_x", "ball_y", "logit_home", "logit_away")


@dataclass(frozen=True)
class AuxConfig:
    """Loss weights of the two aux terms (fixed a priori, never tuned)."""

    ball_weight: float = 0.0
    team_weight: float = 0.0
    smooth_l1_beta: float = 1.0  # PyTorch default
    head_seed_offset: int = 9_000_000  # aux head init seed = TrainConfig.seed + offset

    @property
    def enabled(self) -> bool:
        return self.ball_weight > 0 or self.team_weight > 0

    def as_dict(self) -> dict:
        return asdict(self)


AUX_ON = AuxConfig(ball_weight=0.25, team_weight=0.25)
AUX_OFF = AuxConfig()


class SetSeqAux(N.SetSeq):
    """exp_1 SetSeq + a training-only aux head on the BiGRU output."""

    def __init__(self, width: int, hidden: int, head_seed: int) -> None:
        super().__init__(width, hidden)
        # Seed only the CPU generator (torch.manual_seed would also reseed CUDA, which drives
        # the GRU dropout masks) and restore it afterwards: the main RNG streams are untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(head_seed)
            self.aux_head = nn.Linear(2 * hidden, len(AUX_OUTPUTS))
        self.hidden_state: torch.Tensor | None = None
        self.gru.register_forward_hook(self._keep_hidden)

    def _keep_hidden(self, module, inputs, output) -> None:  # noqa: ARG002
        self.hidden_state = output[0]

    def aux_output(self) -> torch.Tensor:
        """Aux head on the BiGRU output of the most recent forward pass."""
        if self.hidden_state is None:
            raise RuntimeError("Run a forward pass before reading the aux head")
        return self.aux_head(self.hidden_state)


@dataclass(frozen=True)
class AuxSequence:
    xy: torch.Tensor  # (n, 2) float32, canonical normalised clean ball (0 where missing)
    ok: torch.Tensor  # (n,) bool, clean ball present
    team: torch.Tensor  # (n,) int64, team of the clean nearest player (0 home, 1 away, IGNORE)


def to_aux_sequences(arrays: list[dict]) -> list[AuxSequence]:
    """Same per-period split and order as ``exp1_net.to_sequences``."""
    out = []
    for item in arrays:
        if not {"aux_xy", "aux_ok", "aux_team"} <= set(item):
            raise ValueError("Aux training needs aux_xy / aux_ok / aux_team on every array")
        for period in np.unique(item["half"]):
            index = np.flatnonzero(item["half"] == period)
            out.append(AuxSequence(
                xy=torch.from_numpy(item["aux_xy"][index].astype("float32")).to(N.DEVICE),
                ok=torch.from_numpy(item["aux_ok"][index].astype(bool)).to(N.DEVICE),
                team=torch.from_numpy(item["aux_team"][index].astype("int64")).to(N.DEVICE),
            ))
    return out


def aux_batch(sequences: list[AuxSequence], picks: list[tuple[int, int]], window: int):
    parts = [[getattr(sequences[s], name)[start:start + window] for s, start in picks]
             for name in ("xy", "ok", "team")]
    return tuple(torch.stack(p) for p in parts)


def mirror_aux(xy: torch.Tensor, team: torch.Tensor, flip: torch.Tensor):
    """Home/away mirror of the aux targets: ball (x, y) negated, nearest team swapped."""
    xy = torch.where(flip.view(-1, 1, 1), -xy, xy)
    swapped = torch.where(team == 0, 1, torch.where(team == 1, 0, team))
    return xy, torch.where(flip.view(-1, 1), swapped, team)


def aux_loss(output: torch.Tensor, xy: torch.Tensor, ok: torch.Tensor, team: torch.Tensor,
             aux: AuxConfig) -> torch.Tensor:
    """Masked smooth-L1 on the ball position + masked cross-entropy on the nearest team."""
    ball = nn.functional.smooth_l1_loss(output[..., :2], xy, beta=aux.smooth_l1_beta, reduction="none")
    valid = ok.float()
    ball_loss = (ball.mean(dim=-1) * valid).sum() / valid.sum().clamp_min(1.0)
    team_terms = nn.functional.cross_entropy(
        output[..., 2:].reshape(-1, 2), team.reshape(-1), ignore_index=IGNORE, reduction="none")
    labelled = (team.reshape(-1) != IGNORE).float()
    team_loss = team_terms.sum() / labelled.sum().clamp_min(1.0)
    return aux.ball_weight * ball_loss + aux.team_weight * team_loss


def train(arrays: list[dict], config: N.TrainConfig, aux: AuxConfig) -> SetSeqAux:
    """``exp1_net.train`` plus the aux term (identical to it when ``aux`` is disabled)."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    every = N.to_sequences(arrays)
    keep = [len(s.state) >= config.window for s in every]
    sequences = [s for s, k in zip(every, keep) if k]
    aux_sequences = [a for a, k in zip(to_aux_sequences(arrays), keep) if k] if aux.enabled else None
    spans = np.array([len(s.state) - config.window + 1 for s in sequences], dtype="float64")
    total = int(sum(len(s.state) for s in sequences))
    per_epoch = max(config.batch, int(config.coverage * total / config.window))
    batches = per_epoch // config.batch
    model = SetSeqAux(config.width, config.hidden, config.seed + aux.head_seed_offset).to(N.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = config.epochs * batches
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, config.lr, total_steps=steps)
    model.train()
    for _ in range(steps):
        chosen = rng.choice(len(sequences), size=config.batch, p=spans / spans.sum())
        picks = [(int(s), int(rng.integers(0, spans[s]))) for s in chosen]
        tokens, mask, glob, state, boundary = N._batch(sequences, picks, config.window)
        flip = None
        if config.mirror:
            flip = torch.from_numpy(rng.random(config.batch) < 0.5).to(N.DEVICE)
            tokens, glob, state = N._mirror_batch(tokens, glob, state, flip)
        loss = N._loss(model, (tokens, mask, glob, state, boundary), config)
        if aux.enabled:
            xy, ok, team = aux_batch(aux_sequences, picks, config.window)
            if flip is not None:
                xy, team = mirror_aux(xy, team, flip)
            loss = loss + aux_loss(model.aux_output(), xy, ok, team, aux)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    model.eval()
    model.hidden_state = None
    return model


def state_nll(model: N.SetSeq, arrays: list[dict], window: int) -> tuple[float, int]:
    """Mean NLL of the raw (T = 1) p(H, A, D) on labelled 5 fps steps, eval mode, no mirror."""
    total, count = 0.0, 0
    for item in arrays:
        probability, _ = N.predict(model, item, window)
        state = item["state"].astype("int64")
        keep = state < 3
        picked = probability[np.flatnonzero(keep), state[keep]].astype("float64")
        total += float(-np.log(np.clip(picked, 1e-7, 1.0)).sum())
        count += int(keep.sum())
    if hasattr(model, "hidden_state"):
        model.hidden_state = None
    return total / max(count, 1), count
