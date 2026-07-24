"""TensorBoard visualization for Glob3R matching predictions."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

from pi3.models.glob3r.geometry import build_ground_truth_warp, sample_map_at_pixels


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _resize_panel(image: torch.Tensor, height: int, width: int) -> Image.Image:
    image = F.interpolate(
        image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=True
    )[0]
    return _tensor_to_pil(image)


def _row_label(text: str, width: int, height: int) -> Image.Image:
    horizontal = Image.new("RGB", (height, width), "white")
    draw = ImageDraw.Draw(horizontal)
    box = draw.textbbox((0, 0), text)
    draw.text(
        ((height - (box[2] - box[0])) / 2, (width - (box[3] - box[1])) / 2),
        text,
        fill="black",
    )
    return horizontal.rotate(90, expand=True)


@torch.no_grad()
def render_matching_overview(
    images: torch.Tensor,
    prediction: Mapping,
    views: Sequence[Mapping],
    *,
    batch_index: int = 0,
    target_offset: int = 0,
    num_targets: int = 7,
    confidence_threshold: float = 0.5,
    cell_width: int = 160,
) -> Image.Image:
    """Render ``Images / Warp / Conf / Mask`` rows for one training sample."""

    reference_index = int(prediction.get("reference_index", 0))
    all_targets = list(prediction["target_indices"])
    selected_offsets = list(
        range(target_offset, min(target_offset + num_targets, len(all_targets)))
    )
    selected_targets = [int(all_targets[offset]) for offset in selected_offsets]
    frame_indices = [reference_index, *selected_targets]

    reference = images[batch_index, reference_index].detach().float()
    image_height, image_width = reference.shape[-2:]
    cell_height = max(round(image_height / image_width * cell_width), 1)
    blank = Image.new("RGB", (cell_width, cell_height), "white")

    image_panels = [
        _resize_panel(images[batch_index, frame].detach().float(), cell_height, cell_width)
        for frame in frame_indices
    ]
    warp_panels = [image_panels[0]]
    confidence_panels = [blank]

    for offset, target_index in zip(selected_offsets, selected_targets):
        warp = prediction["warp"][batch_index, offset].detach().float()
        confidence = prediction["warp_confidence"][batch_index, offset].detach().float()
        target = images[batch_index, target_index].detach().float()
        warped = sample_map_at_pixels(
            target.unsqueeze(0), warp.permute(1, 2, 0).unsqueeze(0)
        )[0]
        valid = (
            (confidence[0] >= confidence_threshold)
            & (warp[0] >= 0)
            & (warp[0] <= image_width - 1)
            & (warp[1] >= 0)
            & (warp[1] <= image_height - 1)
        )
        warp_panels.append(
            _resize_panel(warped * valid.unsqueeze(0), cell_height, cell_width)
        )
        confidence_panels.append(
            _resize_panel(confidence.expand(3, -1, -1), cell_height, cell_width)
        )

    depths = torch.stack([view["depthmap"][batch_index] for view in views], dim=0)
    intrinsics = torch.stack([view["camera_intrinsics"][batch_index] for view in views], dim=0)
    poses = torch.stack([view["camera_pose"][batch_index] for view in views], dim=0)
    target_from_reference = torch.linalg.inv(poses[selected_targets]) @ poses[reference_index]
    supervision = build_ground_truth_warp(
        depths[reference_index].unsqueeze(0),
        depths[selected_targets].unsqueeze(0),
        intrinsics[reference_index].unsqueeze(0),
        intrinsics[selected_targets].unsqueeze(0),
        target_from_reference.unsqueeze(0),
    )
    mask_panels = [blank] + [
        _resize_panel(mask.float().unsqueeze(0).expand(3, -1, -1), cell_height, cell_width)
        for mask in supervision.mask[0]
    ]

    rows = [image_panels, warp_panels, confidence_panels, mask_panels]
    labels = ["Images", "Warp", "Conf", "Mask"]
    label_width = 56
    canvas = Image.new(
        "RGB",
        (label_width + len(frame_indices) * cell_width, len(rows) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, panels) in enumerate(zip(labels, rows)):
        y = row_index * cell_height
        canvas.paste(_row_label(label, label_width, cell_height), (0, y))
        for column_index, panel in enumerate(panels):
            x = label_width + column_index * cell_width
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="gray")
    return canvas


class Glob3RTensorBoardVisualizer:
    """Schedule and write matching overview grids independently of training logic."""

    def __init__(self, config, gradient_accumulation_steps: int, initial_global_step: int):
        self.config = config
        accumulation = max(int(gradient_accumulation_steps), 1)
        self.step = int(initial_global_step) // accumulation
        self.last_step = -1
        self.validation_step = 0
        self.validation_pending = False

    def begin_validation(self, epoch: int) -> None:
        self.validation_step = int(epoch) + 1
        self.validation_pending = True

    def log(self, accelerator, output, mode: str) -> None:
        if not bool(self.config.get("enabled", True)) or not accelerator.is_main_process:
            return

        interval = int(self.config.get("interval_steps", 0))
        if mode == "train":
            if not accelerator.sync_gradients:
                return
            self.step += 1
            if interval <= 0 or self.step % interval != 0 or self.step == self.last_step:
                return
            tag = "train/matching_overview"
            log_step = self.step
        elif mode == "test":
            if not self.validation_pending:
                return
            tag = "val/matching_overview"
            log_step = self.validation_step
        else:
            return

        prediction, views = output
        images = torch.stack([view["img"] for view in views], dim=1)
        grids = [
            render_matching_overview(
                images,
                prediction,
                views,
                batch_index=batch_index,
                target_offset=int(self.config.get("target_offset", 0)),
                num_targets=int(self.config.get("num_targets", 7)),
                confidence_threshold=float(self.config.get("confidence_threshold", 0.5)),
                cell_width=int(self.config.get("cell_width", 160)),
            )
            for batch_index in range(min(int(self.config.get("num_samples", 1)), images.shape[0]))
        ]
        grid_batch = torch.stack([pil_to_tensor(grid) for grid in grids])
        for tracker in accelerator.trackers:
            tracker.log_images({tag: grid_batch}, step=log_step)
        if mode == "train":
            self.last_step = self.step
        else:
            self.validation_pending = False
