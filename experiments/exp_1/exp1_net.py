"""SetSeq: DeepSets player encoder + bidirectional GRU with state and boundary heads.

Per 5 fps step the identity-free player set is pooled (home mean/max, away mean/max,
ball-attention pool), joined with global ball/gap features, and a 2-layer BiGRU reads
60 s windows so long ball gaps see context from both edges. Two heads: p(H, A, D) and
p(possession boundary at this step), the latter feeds the boundary-aware decoder.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA initialises

from dataclasses import dataclass, asdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from exp1_tokens import GLOBAL_NAMES, GLOBAL_NEGATE, GLOBAL_SWAP, TOKEN_NAMES, TOKEN_NEGATE  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE = 3


@dataclass(frozen=True)
class TrainConfig:
    window: int = 300  # 60 s at 5 fps
    batch: int = 32
    epochs: int = 14
    coverage: float = 2.0  # each training step is seen about twice per epoch
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 64
    hidden: int = 64
    boundary_weight: float = 0.5
    boundary_pos_weight: float = 3.0
    mirror: bool = True
    seed: int = 20261108

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Sequence:
    tokens: torch.Tensor  # (n, K, F) float16 on DEVICE
    mask: torch.Tensor  # (n, K) bool
    glob: torch.Tensor  # (n, G) float16
    state: torch.Tensor  # (n,) int64, IGNORE for unlabelled
    boundary: torch.Tensor  # (n,) float32


class SetSeq(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        token_dim, glob_dim = len(TOKEN_NAMES), len(GLOBAL_NAMES)
        self.phi = nn.Sequential(
            nn.Linear(token_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.attention = nn.Linear(width, 1)
        self.frame = nn.Sequential(
            nn.Linear(5 * width + glob_dim, 2 * width), nn.GELU(), nn.LayerNorm(2 * width)
        )
        self.gru = nn.GRU(
            2 * width, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )
        self.state_head = nn.Linear(2 * hidden, 3)
        self.boundary_head = nn.Linear(2 * hidden, 1)

    @staticmethod
    def _pool(h: torch.Tensor, weight: torch.Tensor) -> list[torch.Tensor]:
        count = weight.sum(dim=2)
        mean = (h * weight).sum(dim=2) / count.clamp_min(1.0)
        peak = torch.where(weight > 0, h, torch.full_like(h, -1e4)).amax(dim=2)
        peak = torch.where(count > 0, peak, torch.zeros_like(peak))
        return [mean, peak]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, glob: torch.Tensor):
        h = self.phi(tokens)
        present = mask.unsqueeze(-1).float()
        team = tokens[..., 5:6]
        home, away = present * (team > 0).float(), present * (team < 0).float()
        score = self.attention(h).squeeze(-1).masked_fill(~mask, -1e4)
        attention = torch.softmax(score, dim=2).unsqueeze(-1) * present
        pooled = self._pool(h, home) + self._pool(h, away) + [(attention * h).sum(dim=2)]
        z = self.frame(torch.cat(pooled + [glob], dim=-1))
        out, _ = self.gru(z)
        return self.state_head(out), self.boundary_head(out).squeeze(-1)


def to_sequences(arrays: list[dict]) -> list[Sequence]:
    """Split cached token arrays into per-period sequences on DEVICE."""
    sequences = []
    for item in arrays:
        for period in np.unique(item["half"]):
            index = np.flatnonzero(item["half"] == period)
            sequences.append(
                Sequence(
                    tokens=torch.from_numpy(item["tokens"][index]).to(DEVICE),
                    mask=torch.from_numpy(item["mask"][index]).to(DEVICE),
                    glob=torch.from_numpy(item["glob"][index]).to(DEVICE),
                    state=torch.from_numpy(item["state"][index].astype("int64")).to(DEVICE),
                    boundary=torch.from_numpy(item["boundary"][index].astype("float32")).to(DEVICE),
                )
            )
    return sequences


def _batch(sequences: list[Sequence], picks: list[tuple[int, int]], window: int):
    parts = [
        [getattr(sequences[s], name)[start : start + window] for s, start in picks]
        for name in ("tokens", "mask", "glob", "state", "boundary")
    ]
    tokens, mask, glob, state, boundary = (torch.stack(p) for p in parts)
    return tokens.float(), mask, glob.float(), state, boundary


def _mirror_batch(tokens, glob, state, flip: torch.Tensor):
    """Apply the label-consistent home/away mirror to the flagged windows."""
    token_sign = torch.ones(tokens.shape[-1], device=DEVICE)
    token_sign[list(TOKEN_NEGATE)] = -1.0
    glob_sign = torch.ones(glob.shape[-1], device=DEVICE)
    glob_sign[list(GLOBAL_NEGATE)] = -1.0
    f = flip.view(-1, 1, 1, 1)
    tokens = torch.where(f, tokens * token_sign, tokens)
    order = list(range(glob.shape[-1]))
    for left, right in GLOBAL_SWAP:
        order[left], order[right] = right, left
    mirrored = (glob * glob_sign)[..., order]
    glob = torch.where(flip.view(-1, 1, 1), mirrored, glob)
    swapped = torch.where(state == 0, 1, torch.where(state == 1, 0, state))
    state = torch.where(flip.view(-1, 1), swapped, state)
    return tokens, glob, state


def _loss(model, batch, config: TrainConfig) -> torch.Tensor:
    tokens, mask, glob, state, boundary = batch
    state_logits, boundary_logits = model(tokens, mask, glob)
    state_loss = nn.functional.cross_entropy(
        state_logits.reshape(-1, 3), state.reshape(-1), ignore_index=IGNORE
    )
    valid = (state != IGNORE).float()
    weight = torch.tensor(config.boundary_pos_weight, device=DEVICE)
    boundary_loss = nn.functional.binary_cross_entropy_with_logits(
        boundary_logits, boundary, pos_weight=weight, reduction="none"
    )
    boundary_loss = (boundary_loss * valid).sum() / valid.sum().clamp_min(1.0)
    return state_loss + config.boundary_weight * boundary_loss


def train(arrays: list[dict], config: TrainConfig) -> SetSeq:
    """Deterministic training on random 60 s windows of the given token arrays."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    sequences = [s for s in to_sequences(arrays) if len(s.state) >= config.window]
    spans = np.array([len(s.state) - config.window + 1 for s in sequences], dtype="float64")
    total = int(sum(len(s.state) for s in sequences))
    per_epoch = max(config.batch, int(config.coverage * total / config.window))
    batches = per_epoch // config.batch
    model = SetSeq(config.width, config.hidden).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = config.epochs * batches
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, config.lr, total_steps=steps)
    model.train()
    for _ in range(steps):
        chosen = rng.choice(len(sequences), size=config.batch, p=spans / spans.sum())
        picks = [(int(s), int(rng.integers(0, spans[s]))) for s in chosen]
        tokens, mask, glob, state, boundary = _batch(sequences, picks, config.window)
        if config.mirror:
            flip = torch.from_numpy(rng.random(config.batch) < 0.5).to(DEVICE)
            tokens, glob, state = _mirror_batch(tokens, glob, state, flip)
        loss = _loss(model, (tokens, mask, glob, state, boundary), config)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    model.eval()
    return model


def _windows(n: int, window: int) -> list[int]:
    if n <= window:
        return [0]
    stride = window // 3
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


@torch.no_grad()
def predict(model: SetSeq, item: dict, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Tent-weighted overlapping windows -> per-step p(H,A,D) and p(boundary)."""
    model.eval()
    n = len(item["half"])
    probability = np.zeros((n, 3), dtype="float64")
    boundary = np.zeros(n, dtype="float64")
    weight = np.zeros(n, dtype="float64")
    for period in np.unique(item["half"]):
        index = np.flatnonzero(item["half"] == period)
        length = min(window, len(index))
        tent = np.minimum(np.arange(length) + 1, length - np.arange(length)).astype("float64")
        for start in _windows(len(index), window):
            part = index[start : start + length]
            tokens = torch.from_numpy(item["tokens"][part]).to(DEVICE).float()[None]
            mask = torch.from_numpy(item["mask"][part]).to(DEVICE)[None]
            glob = torch.from_numpy(item["glob"][part]).to(DEVICE).float()[None]
            state_logits, boundary_logits = model(tokens, mask, glob)
            probability[part] += tent[:, None] * torch.softmax(state_logits[0], -1).cpu().double().numpy()
            boundary[part] += tent * torch.sigmoid(boundary_logits[0]).cpu().double().numpy()
            weight[part] += tent
    return (probability / weight[:, None]).astype("float32"), (boundary / weight).astype("float32")
