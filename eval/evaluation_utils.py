"""Deterministic sequence windows, pixel metrics, and evaluation rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pi3.models.glob3r.geometry import sample_map_at_pixels


@dataclass(frozen=True)
class SequenceWindow:
    """One ordered window expressed in positions of the dataset record."""

    index: int
    sampled_start: int
    sampled_end: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class TargetMetric:
    """Pixel-weighted warp statistics for one directed reference-target pair."""

    target_offset: int
    valid_pixels: int
    loss_sum: float
    mean_loss: float | None


def build_sequence_windows(
    frame_count: int,
    frame_step: int,
    chunk_size: int,
) -> list[SequenceWindow]:
    """Cover a sequence with half-overlapping windows, including its tail."""

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if frame_step < 1:
        raise ValueError("frame_step must be positive")
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least two")

    sampled_positions = list(range(0, frame_count, frame_step))
    if not sampled_positions:
        return []

    if len(sampled_positions) <= chunk_size:
        starts = [0]
    else:
        overlap = chunk_size // 2
        stride = chunk_size - overlap
        tail_start = len(sampled_positions) - chunk_size
        starts = list(range(0, tail_start + 1, stride))
        if starts[-1] != tail_start:
            starts.append(tail_start)

    windows = []
    for index, start in enumerate(starts):
        positions = tuple(sampled_positions[start : start + chunk_size])
        windows.append(
            SequenceWindow(
                index=index,
                sampled_start=start,
                sampled_end=start + len(positions) - 1,
                positions=positions,
            )
        )
    return windows


def shard_items(items: Sequence, process_index: int, process_count: int) -> list:
    """Assign each ordered item to exactly one distributed process."""

    if process_count < 1:
        raise ValueError("process_count must be positive")
    if not 0 <= process_index < process_count:
        raise ValueError("process_index must be within process_count")
    return list(items[process_index::process_count])


def shard_weighted_items(
    items: Sequence,
    weights: Sequence[int | float],
    process_index: int,
    process_count: int,
) -> list:
    """Greedily balance indivisible sequence work across distributed processes."""

    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    if process_count < 1:
        raise ValueError("process_count must be positive")
    if not 0 <= process_index < process_count:
        raise ValueError("process_index must be within process_count")

    assignments = [[] for _ in range(process_count)]
    loads = [0.0 for _ in range(process_count)]
    ordered = sorted(
        enumerate(zip(items, weights)),
        key=lambda entry: (-float(entry[1][1]), entry[0]),
    )
    for _original_index, (item, weight) in ordered:
        rank = min(range(process_count), key=lambda index: (loads[index], index))
        assignments[rank].append(item)
        loads[rank] += float(weight)
    return assignments[process_index]


def final_warp_and_confidence(output, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final dense predictions as ``[T,2,H,W]`` and ``[T,H,W]``."""

    warp = output.warp_stages[-1] if output.warp_stages else output.coarse_warp
    confidence = (
        output.confidence_stages[-1]
        if output.confidence_stages
        else output.coarse_confidence
    )
    warp = warp.squeeze(0)
    confidence = confidence.squeeze(0)
    if warp.shape[-2:] != size:
        warp = F.interpolate(warp, size=size, mode="bilinear", align_corners=True)
        confidence = F.interpolate(
            confidence, size=size, mode="bilinear", align_corners=True
        )
    return warp, confidence.squeeze(1)


def pixelwise_warp_metrics(
    predicted_warp: torch.Tensor,
    ground_truth_warp: torch.Tensor,
    positive: torch.Tensor,
    training_mask: torch.Tensor,
    epsilon: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, list[TargetMetric]]:
    """Evaluate the final warp with the training Charbonnier pixel definition."""

    ground_truth = ground_truth_warp.permute(0, 3, 1, 2)
    residual = predicted_warp.float() - ground_truth.float()
    loss_map = (residual.square().sum(dim=1) + epsilon**2).pow(alpha / 2)
    valid = positive.bool() & training_mask.bool()

    metrics = []
    for target_offset in range(loss_map.shape[0]):
        target_valid = valid[target_offset]
        valid_pixels = int(target_valid.sum().item())
        if valid_pixels:
            loss_sum = float(loss_map[target_offset][target_valid].double().sum().item())
            mean_loss = loss_sum / valid_pixels
        else:
            loss_sum = 0.0
            mean_loss = None
        metrics.append(
            TargetMetric(
                target_offset=target_offset,
                valid_pixels=valid_pixels,
                loss_sum=loss_sum,
                mean_loss=mean_loss,
            )
        )
    return loss_map, valid, metrics


def _panel(image: torch.Tensor, height: int, width: int) -> Image.Image:
    image = F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)
    array = (
        torch.nan_to_num(image.detach().float().cpu())
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _gray(value: torch.Tensor) -> torch.Tensor:
    return value.float().clamp(0, 1).unsqueeze(0).expand(3, -1, -1)


def _jet(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp(0, 1)
    return torch.stack(
        (
            (1.5 - (4 * value - 3).abs()).clamp(0, 1),
            (1.5 - (4 * value - 2).abs()).clamp(0, 1),
            (1.5 - (4 * value - 1).abs()).clamp(0, 1),
        ),
        dim=0,
    )


def _annotate(panel: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(panel)
    text = text[-48:]
    box = draw.textbbox((5, 5), text)
    draw.rectangle(
        (box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3),
        fill="black",
    )
    draw.text((5, 5), text, fill="white")


@torch.no_grad()
def render_evaluation_matrix(
    images: torch.Tensor,
    frame_instances: Sequence[str],
    reference_index: int,
    target_indices: Sequence[int],
    predicted_warp: torch.Tensor,
    predicted_confidence: torch.Tensor,
    ground_truth_warp: torch.Tensor,
    ground_truth_positive: torch.Tensor,
    training_mask: torch.Tensor,
    depths: torch.Tensor,
    depth_confidence: torch.Tensor,
    confidence_threshold: float,
    cell_width: int,
) -> Image.Image:
    """Render Image, Warp, Warp GT, Conf, Depth, and Depth Conf columns."""

    frame_count, _, image_height, image_width = images.shape
    cell_height = max(round(image_height / image_width * cell_width), 1)
    header_height = 24
    headers = ("Image", "Warp", "Warp GT", "Conf", "Depth", "Depth Conf")
    target_offsets = {int(target): offset for offset, target in enumerate(target_indices)}

    depth_valid = torch.isfinite(depths) & (depths > 0)
    if depth_valid.any():
        limits = torch.quantile(
            depths[depth_valid].float(),
            depths.new_tensor([0.02, 0.98]),
        )
        depth_min, depth_max = limits.unbind()
    else:
        depth_min = depths.new_tensor(0.0)
        depth_max = depths.new_tensor(1.0)

    canvas = Image.new(
        "RGB",
        (len(headers) * cell_width, header_height + frame_count * cell_height),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        x = column * cell_width
        box = draw.textbbox((0, 0), header)
        text_width = box[2] - box[0]
        draw.text((x + (cell_width - text_width) / 2, 5), header, fill="white")

    for frame_index in range(frame_count):
        image_panel = _panel(images[frame_index], cell_height, cell_width)
        role = "ref" if frame_index == reference_index else "target"
        _annotate(image_panel, f"{role} {frame_instances[frame_index]}")

        if frame_index == reference_index:
            predicted_panel = images[frame_index]
            ground_truth_panel = images[frame_index]
            warp_confidence = torch.zeros_like(depths[frame_index])
        else:
            offset = target_offsets[frame_index]
            predicted_xy = predicted_warp[offset].permute(1, 2, 0)
            warp_confidence = predicted_confidence[offset]
            predicted_finite = torch.isfinite(predicted_xy).all(dim=-1)
            predicted_inside = (
                (predicted_xy[..., 0] >= 0)
                & (predicted_xy[..., 0] <= image_width - 1)
                & (predicted_xy[..., 1] >= 0)
                & (predicted_xy[..., 1] <= image_height - 1)
            )
            predicted_xy = torch.where(
                predicted_finite.unsqueeze(-1), predicted_xy, 0
            )
            predicted_panel = sample_map_at_pixels(
                images[frame_index].unsqueeze(0),
                predicted_xy.unsqueeze(0),
            ).squeeze(0)
            predicted_display_valid = (
                predicted_finite
                & predicted_inside
                & (warp_confidence >= confidence_threshold)
            )
            predicted_panel *= predicted_display_valid.unsqueeze(0)

            ground_truth_xy = ground_truth_warp[offset]
            ground_truth_panel = sample_map_at_pixels(
                images[frame_index].unsqueeze(0),
                ground_truth_xy.unsqueeze(0),
            ).squeeze(0)
            ground_truth_valid = (
                ground_truth_positive[offset] & training_mask[offset]
            )
            ground_truth_panel *= ground_truth_valid.unsqueeze(0)

        normalized_depth = (
            (depths[frame_index] - depth_min)
            / (depth_max - depth_min).clamp_min(1.0e-8)
        ).clamp(0, 1)
        depth_panel = _jet(normalized_depth) * depth_valid[frame_index].unsqueeze(0)
        panels = (
            image_panel,
            predicted_panel,
            ground_truth_panel,
            _gray(warp_confidence),
            depth_panel,
            _gray(depth_confidence[frame_index]),
        )

        y = header_height + frame_index * cell_height
        for column, panel in enumerate(panels):
            x = column * cell_width
            panel_image = (
                panel
                if isinstance(panel, Image.Image)
                else _panel(panel, cell_height, cell_width)
            )
            canvas.paste(panel_image, (x, y))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="gray",
            )
    return canvas


__all__ = [
    "SequenceWindow",
    "TargetMetric",
    "build_sequence_windows",
    "final_warp_and_confidence",
    "pixelwise_warp_metrics",
    "render_evaluation_matrix",
    "shard_items",
    "shard_weighted_items",
]
