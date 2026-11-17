from __future__ import annotations

import numpy as np
import torch


def apply(
    x: torch.Tensor,
    pp: torch.Tensor,
    ppm: torch.Tensor,
    rng: np.random.Generator,
    family: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regularize player branches, not raw tracking; no ground-truth inputs."""
    if family == "control":
        return x, pp, ppm
    if family == "pp_dropout":
        # This hides PP tokens across the crop, not pooled features or raw tracking.
        keep = torch.as_tensor(
            rng.random((x.shape[0], 48)) >= 0.2, device=ppm.device,
        )
        return x, pp, ppm & keep[:, None, :]
    if family == "modality_dropout":
        choice = rng.random(x.shape[0])
        pooled_drop = torch.as_tensor(choice < 0.2, device=x.device)
        token_drop = torch.as_tensor(
            (choice >= 0.2) & (choice < 0.4), device=ppm.device,
        )
        augmented_x = x.clone()
        augmented_x[pooled_drop, :24, :] = 0
        return augmented_x, pp, ppm & ~token_drop[:, None, None]
    raise ValueError(f"unknown dropout family: {family!r}")
