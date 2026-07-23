"""Learning-rate schedule used by Glob3R Appendix B."""

import math

import torch.optim.lr_scheduler as lr_scheduler


class CosineWithWarmupLR(lr_scheduler.LambdaLR):
    """Linear warm-up followed by cosine decay.

    Appendix B specifies a 2K-iteration warm-up for both the 32K coarse and
    refinement stages.  ``warmup_steps`` is expressed directly in optimizer
    steps so gradient accumulation is handled by the existing trainer.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 2000,
        accumulation_steps: int = 1,
        min_factor: float = 0.0,
        last_epoch: int = -1,
    ):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        # BaseTrainer supplies dataloader (micro-batch) steps, while Accelerate
        # advances its wrapped scheduler only when the optimizer really steps.
        optimizer_steps = total_steps // accumulation_steps
        if not 0 <= warmup_steps < optimizer_steps:
            raise ValueError("warmup_steps must be smaller than optimizer steps")

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return max(step, 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_factor + (1.0 - min_factor) * cosine

        super().__init__(optimizer, lr_lambda=schedule, last_epoch=last_epoch)
