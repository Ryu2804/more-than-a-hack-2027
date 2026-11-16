from copy import deepcopy
from itertools import zip_longest

import torch


def create(model):
    """Copy initialization once; subsequent optimizer steps update only this shadow."""
    shadow = deepcopy(model)
    shadow.eval()
    for parameter in shadow.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return shadow


def _pairs(shadow_items, model_items, kind):
    pairs = []
    for shadow_item, model_item in zip_longest(shadow_items, model_items):
        if shadow_item is None or model_item is None:
            raise ValueError(f"EMA {kind} names differ")
        shadow_name, shadow_value = shadow_item
        model_name, model_value = model_item
        if shadow_name != model_name:
            raise ValueError(
                f"EMA {kind} names differ: {shadow_name!r} != {model_name!r}"
            )
        if shadow_value.shape != model_value.shape:
            raise ValueError(f"EMA {kind} shape differs for {model_name!r}")
        if shadow_value.dtype != model_value.dtype:
            raise ValueError(f"EMA {kind} dtype differs for {model_name!r}")
        if shadow_value.device != model_value.device:
            raise ValueError(f"EMA {kind} device differs for {model_name!r}")
        if shadow_value is model_value or (
            shadow_value.numel() and shadow_value.data_ptr() == model_value.data_ptr()
        ):
            raise ValueError(f"EMA {kind} shares training tensor {model_name!r}")
        pairs.append((shadow_value, model_value))
    return pairs


@torch.no_grad()
def update(shadow, model, decay=0.99):
    """Call after each optimizer step; buffers retain latest training values."""
    if shadow is model:
        raise ValueError("EMA shadow must be separate from training model")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must lie in [0, 1]")
    parameters = _pairs(
        shadow.named_parameters(), model.named_parameters(), "parameter"
    )
    buffers = _pairs(shadow.named_buffers(), model.named_buffers(), "buffer")
    for shadow_value, model_value in parameters:
        shadow_value.lerp_(model_value, 1.0 - decay)
    for shadow_value, model_value in buffers:
        shadow_value.copy_(model_value)
