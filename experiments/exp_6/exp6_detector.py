"""3-frame heatmap U-Net ball detector (PREREG 3.3-3.4).

Network, streaming training (one sequential decode pass = one epoch, a seeded shuffle buffer kept on
the GPU, a checkpoint after every pass so a killed run resumes bit-identically) and top-k peak
inference with chunked, resumable outputs. Deterministic ops only. Frames and crops live only in
process / GPU memory; nothing per-frame is printed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import exp6_common as C  # noqa: F401  (environment first)
import exp6_data as DA

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_EVERY = 500


@dataclass(frozen=True)
class DetectorConfig:
    channels: tuple[int, ...] = (16, 32, 64, 128)
    frames: int = 3  # t-1, t, t+1
    crop: int = 128
    central: int = 96
    sigma_px: float = 2.5
    pos_weight: float = 50.0
    near_player_min_px: float = 40.0
    passes: int = 4
    buffer: int = 4096
    batch: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-5
    seed: int = 20261108
    infer_margin_px: int = 32
    top_k: int = 5
    nms: int = 7

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def stride(self) -> int:
        return 2 ** (len(self.channels) - 1)


@dataclass(frozen=True)
class TrainData:
    targets: np.ndarray  # (N,) bool: train-block frames with a kept auto-label
    label_uv: np.ndarray  # (N, 2) projected ball (pixels), NaN off-target
    bbox: np.ndarray  # (N, 4) pitch bounding box [x_min, y_min, x_max, y_max)
    players_uv: np.ndarray  # (n_targets, K, 2) projected players, one row per target frame in order
    image_size: tuple[int, int]  # (width, height)


# ---------------------------------------------------------------- network
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class UNet(nn.Module):
    """4-level U-Net: max-pool down, nearest upsampling + conv up, one logit per pixel."""

    def __init__(self, channels: tuple[int, ...], in_channels: int) -> None:
        super().__init__()
        widths = (in_channels, *channels)
        levels = range(len(channels) - 1)
        self.down = nn.ModuleList(_block(widths[i], widths[i + 1]) for i in range(len(channels)))
        self.up = nn.ModuleList(_up(channels[i + 1], channels[i]) for i in reversed(levels))
        self.merge = nn.ModuleList(_block(2 * channels[i], channels[i]) for i in reversed(levels))
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for index, block in enumerate(self.down):
            x = block(F.max_pool2d(x, 2) if index else x)
            skips.append(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1])):
            x = merge(torch.cat([up(F.interpolate(x, scale_factor=2, mode="nearest")), skip], dim=1))
        return self.head(x)


def new_model(cfg: DetectorConfig) -> UNet:
    torch.manual_seed(cfg.seed)
    return UNet(cfg.channels, 3 * cfg.frames).to(DEVICE)


def heatmap(offsets: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    """(B, 1, size, size) Gaussian targets centred at per-sample (x, y) crop offsets."""
    grid = torch.arange(size, device=offsets.device, dtype=torch.float32)
    dx = grid[None, None, :] - offsets[:, 0, None, None]
    dy = grid[None, :, None] - offsets[:, 1, None, None]
    return torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))[:, None]


def to_device(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame)).to(DEVICE)


def crop_into(out: torch.Tensor, frames: tuple, x0: int, y0: int) -> torch.Tensor:
    """Write the RGB crop [y0:y0+h, x0:x0+w] of each BGR frame into ``out`` (C, h, w); zero outside."""
    out.zero_()
    height, width = frames[0].shape[:2]
    size_y, size_x = out.shape[1], out.shape[2]
    xa, xb, ya, yb = max(x0, 0), min(x0 + size_x, width), max(y0, 0), min(y0 + size_y, height)
    if xa < xb and ya < yb:
        for k, frame in enumerate(frames):
            out[3 * k:3 * k + 3, ya - y0:yb - y0, xa - x0:xb - x0] = frame[ya:yb, xa:xb].flip(-1).permute(2, 0, 1)
    return out


# ---------------------------------------------------------------- training samples
def eligible_players(players_uv: np.ndarray, ball_uv: np.ndarray, cfg: DetectorConfig,
                     image: tuple[int, int]) -> np.ndarray:
    """(n, K) players inside the image and at least ``near_player_min_px`` from the ball."""
    u, v = players_uv[..., 0], players_uv[..., 1]
    with np.errstate(invalid="ignore"):
        inside = (u >= 0) & (u <= image[0] - 1) & (v >= 0) & (v <= image[1] - 1)
        far = np.hypot(u - ball_uv[:, None, 0], v - ball_uv[:, None, 1]) >= cfg.near_player_min_px
    return inside & far & np.isfinite(u) & np.isfinite(v)


def plan_crops(uv: np.ndarray, bbox: np.ndarray, players_uv: np.ndarray, rng: np.random.Generator,
               cfg: DetectorConfig, image: tuple[int, int]) -> list[tuple[int, int]]:
    """Crop origins for one target frame: positive, random pitch negative, near-player negative."""
    margin = (cfg.crop - cfg.central) // 2
    origins = [(int(np.floor(uv[0])) - int(rng.integers(margin, margin + cfg.central)),
                int(np.floor(uv[1])) - int(rng.integers(margin, margin + cfg.central)))]
    x_min, y_min, x_max, y_max = (int(b) for b in bbox)
    origins.append((int(rng.integers(x_min, max(x_min, x_max - cfg.crop) + 1)),
                    int(rng.integers(y_min, max(y_min, y_max - cfg.crop) + 1))))
    eligible = np.flatnonzero(eligible_players(players_uv[None], uv[None], cfg, image)[0])
    if len(eligible):
        pu, pv = players_uv[eligible[int(rng.integers(len(eligible)))]]
        origins.append((int(np.floor(pu)) - cfg.crop // 2, int(np.floor(pv)) - cfg.crop // 2))
    return origins


def samples_per_pass(data: TrainData, cfg: DetectorConfig) -> int:
    frames = np.flatnonzero(data.targets)
    near = eligible_players(data.players_uv, data.label_uv[frames], cfg, data.image_size).any(axis=1)
    return 2 * len(frames) + int(near.sum())


def lr_at(step: int, total: int, cfg: DetectorConfig) -> float:
    """Cosine decay from lr to lr_min over all training steps."""
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * step / max(1, total - 1)))


class _Stream:
    """Seeded shuffle buffer on the GPU; evicted samples fill fixed-size batches for ``step``."""

    def __init__(self, cfg: DetectorConfig, step: Callable, rng: np.random.Generator) -> None:
        shape = (3 * cfg.frames, cfg.crop, cfg.crop)
        self.store = torch.zeros((cfg.buffer, *shape), dtype=torch.uint8, device=DEVICE)
        self.store_offset = np.zeros((cfg.buffer, 2), dtype=np.float32)
        self.batch = torch.zeros((cfg.batch, *shape), dtype=torch.uint8, device=DEVICE)
        self.batch_offset = np.zeros((cfg.batch, 2), dtype=np.float32)
        self.cfg, self.step, self.rng = cfg, step, rng
        self.size = self.fill = self.emitted = 0

    def slot(self) -> int:
        """Buffer slot for one new sample (a full buffer first evicts a random sample to the batch)."""
        if self.size < self.cfg.buffer:
            self.size += 1
            return self.size - 1
        slot = int(self.rng.integers(self.cfg.buffer))
        self._emit(slot)
        return slot

    def _emit(self, slot: int) -> None:
        self.batch[self.fill].copy_(self.store[slot])
        self.batch_offset[self.fill] = self.store_offset[slot]
        self.fill += 1
        self.emitted += 1
        if self.fill == self.cfg.batch:
            self.flush()

    def flush(self) -> None:
        if self.fill:
            self.step(self.batch[:self.fill], self.batch_offset[:self.fill].copy())
            self.fill = 0

    def drain(self) -> None:
        """End of a pass: emit every buffered sample in random order, then the partial batch."""
        for slot in self.rng.permutation(self.size):
            self._emit(int(slot))
        self.size = 0
        self.flush()


def _checkpoints(folder: Path) -> list[Path]:
    return sorted(folder.glob("pass_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def _save(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(source, data: TrainData, cfg: DetectorConfig, folder: Path, log: Callable = print) -> UNet:
    """Train (or resume) the detector; returns the final model in eval mode."""
    folder.mkdir(parents=True, exist_ok=True)
    frames = np.flatnonzero(data.targets)
    row_of = {int(t): i for i, t in enumerate(frames)}
    per_pass = samples_per_pass(data, cfg)
    total = cfg.passes * math.ceil(per_pass / cfg.batch)
    model = new_model(cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    counters = {"step": 0, "loss": torch.zeros((), device=DEVICE), "since": 0, "t0": time.time()}
    start_pass = 0
    if done := _checkpoints(folder):
        state = torch.load(done[-1], map_location="cpu", weights_only=False)  # CPU RNG state stays on CPU
        if state["config"] != cfg.as_dict() or state["per_pass"] != per_pass:
            raise RuntimeError("Checkpoint belongs to a different configuration")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_cpu"])
        counters["step"], start_pass = state["step"], state["pass"] + 1
        log(f"resumed after pass {state['pass']} (step {state['step']} of {total})")
    pos_weight = torch.tensor(cfg.pos_weight, device=DEVICE)

    def step(batch: torch.Tensor, offsets: np.ndarray) -> None:
        for group in optimiser.param_groups:
            group["lr"] = lr_at(counters["step"], total, cfg)
        target = heatmap(torch.from_numpy(offsets).to(DEVICE), cfg.crop, cfg.sigma_px)
        loss = F.binary_cross_entropy_with_logits(model(batch.float() / 255.0), target, pos_weight=pos_weight)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        counters["step"] += 1
        counters["loss"] += loss.detach()
        counters["since"] += 1
        if counters["step"] % LOG_EVERY == 0:
            log(f"step {counters['step']}/{total} mean loss {counters['loss'].item() / counters['since']:.5f} "
                f"elapsed {time.time() - counters['t0']:.0f}s")
            counters["loss"].zero_()
            counters["since"] = 0

    needed = DA.dilate(data.targets)
    stream = _Stream(cfg, step, rng) if start_pass < cfg.passes else None
    model.train()
    for pass_index in range(start_pass, cfg.passes):
        start, stream.emitted, count = time.time(), 0, 0
        triples = DA.triplets(source.frames(needed, check_count=pass_index == 0), data.targets, to_device)
        for t, triple in triples:
            uv = data.label_uv[t]
            for x0, y0 in plan_crops(uv, data.bbox[t], data.players_uv[row_of[t]], rng, cfg, data.image_size):
                slot = stream.slot()
                crop_into(stream.store[slot], triple, x0, y0)
                stream.store_offset[slot] = (uv[0] - x0, uv[1] - y0)
                count += 1
        stream.drain()
        if count != per_pass or stream.emitted != per_pass:
            raise RuntimeError("Training pass produced an unexpected number of samples")
        _save({"pass": pass_index, "step": counters["step"], "model": model.state_dict(),
               "optimiser": optimiser.state_dict(), "rng": rng.bit_generator.state,
               "torch_cpu": torch.get_rng_state(), "per_pass": per_pass, "total": total,
               "config": cfg.as_dict()}, folder / f"pass_{pass_index}.pt")
        log(f"pass {pass_index + 1}/{cfg.passes} done: {count} samples, {time.time() - start:.0f}s")
    if counters["step"] != total:
        raise RuntimeError("Training ended at an unexpected step count")
    final = folder / "detector.pt"
    if not final.exists():
        _save({k: v.detach().cpu() for k, v in model.state_dict().items()}, final)
    return model.eval()


# ---------------------------------------------------------------- inference
def peaks(logits: torch.Tensor, k: int, nms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k local maxima (nms x nms max-pool suppression) of a (h, w) logit map."""
    pooled = F.max_pool2d(logits[None, None], nms, stride=1, padding=nms // 2)[0, 0]
    masked = torch.where(logits == pooled, logits, torch.full_like(logits, -torch.inf))
    values, flat = torch.topk(masked.flatten(), k)
    width = logits.shape[1]
    return ((flat // width).cpu().numpy(), (flat % width).cpu().numpy(), values.double().cpu().numpy())


@torch.no_grad()
def detect(model: UNet, frames: tuple, region: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Top-k candidate pixels (u, v) and logits inside ``region`` [x0, y0, x1, y1) of one frame."""
    x0, y0, x1, y1 = (int(r) for r in region)
    width, height = x1 - x0, y1 - y0
    pad_w, pad_h = (math.ceil(s / cfg.stride) * cfg.stride for s in (width, height))
    x = crop_into(torch.empty((3 * cfg.frames, pad_h, pad_w), dtype=torch.uint8, device=DEVICE), frames, x0, y0)
    x[:, height:, :] = 0
    x[:, :, width:] = 0
    logits = model(x[None].float() / 255.0)[0, 0]
    logits[height:, :] = -torch.inf
    logits[:, width:] = -torch.inf
    rows, cols, values = peaks(logits, cfg.top_k, cfg.nms)
    return np.stack([cols + x0, rows + y0], axis=1).astype(np.float64), values


def infer(model: UNet, source, frames_mask: np.ndarray, regions: np.ndarray, cfg: DetectorConfig,
          folder: Path, log: Callable = print, chunk: int = 4096) -> dict:
    """Candidates for every masked frame, saved in resumable chunks: frame, uv (n, k, 2), logit (n, k)."""
    folder.mkdir(parents=True, exist_ok=True)
    order = np.flatnonzero(frames_mask)
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    paths = [folder / f"chunk_{k:04d}.npz" for k in range(len(chunks))]
    remaining = np.zeros(len(frames_mask), dtype=bool)
    for frames, path in zip(chunks, paths):
        remaining[frames] = not path.exists()
    if remaining.any():
        model.eval()
        log(f"inference on {int(remaining.sum())} frames ({sum(not p.exists() for p in paths)} chunks)")
        chunk_of = np.full(len(frames_mask), -1)
        for k, frames in enumerate(chunks):
            chunk_of[frames] = k
        pending: dict[int, dict] = {}
        start = time.time()
        source_frames = source.frames(DA.dilate(remaining), check_count=False)
        for done, (t, triple) in enumerate(DA.triplets(source_frames, remaining, to_device), 1):
            uv, logit = detect(model, triple, regions[t], cfg)
            k = int(chunk_of[t])
            pending.setdefault(k, {})[t] = (uv, logit)
            if len(pending[k]) == len(chunks[k]):
                rows = pending.pop(k)
                tmp = paths[k].with_name(paths[k].stem + ".tmp.npz")
                np.savez(tmp, frame=chunks[k], uv=np.stack([rows[int(f)][0] for f in chunks[k]]),
                         logit=np.stack([rows[int(f)][1] for f in chunks[k]]))
                os.replace(tmp, paths[k])
            if done % 5000 == 0:
                log(f"inference {done}/{int(remaining.sum())} frames, {time.time() - start:.0f}s")
        if pending:
            raise RuntimeError("Inference ended with incomplete chunks")
    parts = [np.load(p) for p in paths]
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("frame", "uv", "logit")}
    if not np.array_equal(out["frame"], order):
        raise RuntimeError("Inference chunks do not cover the requested frames")
    return out
