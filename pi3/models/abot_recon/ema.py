"""Exponential moving average for ABot-Recon training."""

from __future__ import annotations

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class ABotReconEMA:
    """Checkpointable parameter EMA used for validation and later stages."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.averaged = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
            use_buffers=False,
        )
        self.averaged.requires_grad_(False)
        self.averaged.eval()
        # Include the initialization checkpoint in the moving average.  Without
        # this, AveragedModel replaces it with the first optimizer-step weights.
        self.averaged.n_averaged.fill_(1)

    @property
    def module(self):
        return self.averaged.module

    @torch.no_grad()
    def update(self, model):
        self.averaged.update_parameters(model)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": int(self.averaged.n_averaged.item()),
            "model": self.module.state_dict(),
        }

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.averaged.multi_avg_fn = get_ema_multi_avg_fn(self.decay)
        result = self.module.load_state_dict(state["model"], strict=True)
        self.averaged.n_averaged.fill_(int(state.get("num_updates", 1)))
        return result


__all__ = ["ABotReconEMA"]
