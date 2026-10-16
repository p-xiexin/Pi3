"""Adjacent Glob3R matching predictions and supervision in TensorBoard."""

from pathlib import Path

import torch
from torchvision.transforms.functional import pil_to_tensor

from pi3.models.glob3r.geometry import sample_map_at_pixels
from pi3.models.glob3r.loss import _resize_scalar_map, _resize_warp
from utils.glob3r_visualization import _tensor_to_pil
from ..viz import _gray_panel, _heatmap, _labeled_grid


@torch.no_grad()
def render_matching_overview(images, prediction, supervision, *, batch_index=0,
                             confidence_threshold=0.5, cell_width=160,
                             epe_max=10.0):
    """Each column has its own previous reference and current target frame."""
    refs = prediction["reference_indices"].tolist()
    targets = prediction["target_indices"].tolist()
    if not refs or any(ref + 1 != target for ref, target in zip(refs, targets)):
        raise ValueError("matching visualization requires adjacent frame pairs")
    height, width = prediction["image_size"]
    size = prediction["coarse_warp"].shape[-2:]
    target_warp = _resize_warp(supervision.warp, size)[batch_index]
    positive = _resize_scalar_map(supervision.confidence, size, "nearest")[batch_index].bool()
    mask = _resize_scalar_map(supervision.mask, size, "nearest")[batch_index].bool()
    rows = [(label, []) for label in (
        "Reference", "Target", "Warp s4", "GT Warp s4", "Conf", "GT Conf",
        "Supervision", f"EPE 0..{epe_max:g}",
    )]
    for pair, (ref, target) in enumerate(zip(refs, targets)):
        warp = prediction["coarse_warp"][batch_index, pair].detach().float()
        confidence = prediction["coarse_confidence"][batch_index, pair, 0].detach().float()
        finite = torch.isfinite(warp).all(0)
        inside = (finite & (warp[0] >= 0) & (warp[0] <= width - 1)
                  & (warp[1] >= 0) & (warp[1] <= height - 1))
        valid = inside & (confidence >= confidence_threshold)
        current = images[batch_index, target].detach().float()

        def warp_rgb(coordinates, valid_pixels):
            safe = torch.nan_to_num(coordinates, nan=0.0, posinf=0.0, neginf=0.0)
            warped = sample_map_at_pixels(current[None], safe.permute(1, 2, 0)[None])[0]
            return _tensor_to_pil(warped.masked_fill(~valid_pixels[None], 0))

        error = (warp - target_warp[pair]).norm(dim=0)
        panels = [
            _tensor_to_pil(images[batch_index, ref]), _tensor_to_pil(current),
            warp_rgb(warp, valid), warp_rgb(target_warp[pair], positive[pair]),
            _gray_panel(torch.nan_to_num(confidence)),
            _gray_panel(positive[pair].float()), _gray_panel(mask[pair].float()),
            _heatmap(error, positive[pair] & finite, 0.0, epe_max),
        ]
        for (_, row), panel in zip(rows, panels):
            row.append(panel)
    return _labeled_grid(rows, [f"{ref} -> {target}" for ref, target in zip(refs, targets)],
                         cell_width, height, width)


class MatchTensorBoard:
    """One writer on rank zero, with steps supplied by the optimizer loop."""

    def __init__(self, output_dir, config, *, enabled=True, is_main_process=True,
                 resume_step=None):
        self.config = config
        self.writer = None
        if not enabled or not is_main_process:
            return
        if bool(config.get("enabled", True)):
            if int(config.get("num_frames", 8)) < 2 or int(config.get("num_samples", 1)) < 1:
                raise ValueError("visualization requires at least two frames and one sample")
            if int(config.get("cell_width", 160)) < 1 or float(config.get("epe_max", 10.0)) <= 0:
                raise ValueError("visualization cell_width and epe_max must be positive")
            if not 0 <= float(config.get("confidence_threshold", 0.5)) <= 1:
                raise ValueError("visualization confidence_threshold must lie in [0,1]")
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir=str(Path(output_dir) / "tensorboard"),
            purge_step=None if resume_step is None else int(resume_step) + 1,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.writer is not None:
            self.writer.close()

    def log_scalars(self, output, global_step, prefix="train"):
        if self.writer is None:
            return
        for name, value in output.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, global_step)

    @torch.no_grad()
    def log_images(self, accelerator, model, batch, criterion, step, mode):
        if self.writer is None or not bool(self.config.get("enabled", True)):
            return
        interval = int(self.config.get("interval_steps", 500))
        if mode == "train" and (interval <= 0 or step % interval != 0):
            return
        if mode not in ("train", "test"):
            raise ValueError("visualization mode must be train or test")
        frames = min(len(batch), int(self.config.get("num_frames", 8)))
        samples = min(batch[0]["img"].shape[0], int(self.config.get("num_samples", 1)))
        sequence = [
            {key: value[:samples] if isinstance(value, (torch.Tensor, list, tuple)) else value
             for key, value in view.items()}
            for view in batch[:frames]
        ]
        images = torch.stack([view["img"] for view in sequence], 1)
        unwrapped = accelerator.unwrap_model(model)
        was_training = unwrapped.training
        try:
            unwrapped.eval()
            # A bounded prefix preserves frame-zero initialization and causal context.
            with accelerator.autocast():
                prediction = unwrapped(images)
            with torch.autocast(device_type=images.device.type, enabled=False):
                supervision = criterion.supervision(prediction, sequence)
            for sample in range(samples):
                matching = render_matching_overview(
                    images, prediction, supervision, batch_index=sample,
                    confidence_threshold=float(self.config.get("confidence_threshold", 0.5)),
                    cell_width=int(self.config.get("cell_width", 160)),
                    epe_max=float(self.config.get("epe_max", 10.0)),
                )
                suffix = "" if samples == 1 else f"/sample_{sample}"
                prefix = "train" if mode == "train" else "val"
                self.writer.add_image(f"{prefix}/matching_overview{suffix}", pil_to_tensor(matching), step)
            self.writer.flush()
        finally:
            unwrapped.train(was_training)
