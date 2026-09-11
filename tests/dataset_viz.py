"""Visualize dataset image, depth, warp, confidence and mask with Hydra."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.timing import tic, toc


@dataclass
class WarpSupervision:
    warp: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor


def _pixel_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    homogeneous = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return homogeneous.unsqueeze(0).expand(batch, -1, -1, -1)


def _inside_image(xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= width - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= height - 1)
    )


def sample_map_at_pixels(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    height, width = value.shape[-2:]
    gx = 2.0 * xy[..., 0] / (width - 1) - 1.0
    gy = 2.0 * xy[..., 1] / (height - 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def build_ground_truth_warp(
    reference_depth: torch.Tensor,
    target_depth: torch.Tensor,
    reference_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_from_reference: torch.Tensor,
    depth_threshold: float,
) -> WarpSupervision:
    batch, targets, height, width = target_depth.shape
    pixels = _pixel_grid(
        batch,
        height,
        width,
        reference_depth.device,
        reference_depth.dtype,
    )
    pixels_flat = pixels.reshape(batch, -1, 3)
    depth_flat = reference_depth.reshape(batch, -1, 1)

    rays = torch.einsum(
        "bij,bmj->bmi", torch.linalg.inv(reference_intrinsics), pixels_flat
    )
    points_reference = rays * depth_flat
    rotation = target_from_reference[..., :3, :3]
    translation = target_from_reference[..., :3, 3]
    points_target = (
        torch.einsum("btij,bmj->btmi", rotation, points_reference)
        + translation[:, :, None]
    )
    homogeneous_target = torch.einsum(
        "btij,btmj->btmi", target_intrinsics, points_target
    )
    projected_depth = points_target[..., 2]
    projected_xy = homogeneous_target[..., :2] / homogeneous_target[..., 2:3]
    projected_xy = projected_xy.reshape(batch, targets, height, width, 2)
    projected_depth = projected_depth.reshape(batch, targets, height, width)

    inside = _inside_image(projected_xy, height, width)
    sampled_target_depth = sample_map_at_pixels(
        target_depth.reshape(batch * targets, 1, height, width),
        projected_xy.reshape(batch * targets, height, width, 2),
    ).reshape(batch, targets, height, width)
    positive_geometry = (projected_depth > 0) & (sampled_target_depth > 0)
    relative_depth_error = (
        (sampled_target_depth - projected_depth).abs() / sampled_target_depth
    )
    confidence = inside & positive_geometry & (relative_depth_error < depth_threshold)
    reference_valid = reference_depth[:, None] > 0
    mask = reference_valid & ((inside & positive_geometry) | (~inside))
    return WarpSupervision(projected_xy, confidence, mask)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        torch.nan_to_num(image.detach().float().cpu())
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _resize_panel(image: torch.Tensor, height: int, width: int) -> Image.Image:
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )[0]
    return _tensor_to_pil(resized)


def _row_label(text: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (height, width), "white")
    draw = ImageDraw.Draw(panel)
    box = draw.textbbox((0, 0), text)
    draw.text(
        ((height - (box[2] - box[0])) / 2, (width - (box[3] - box[1])) / 2),
        text,
        fill="black",
    )
    return panel.rotate(90, expand=True)


def _depth_to_rgb(depth: torch.Tensor) -> torch.Tensor:
    valid = depth > 0
    values = depth[valid]
    near = torch.quantile(values, 0.02)
    far = torch.quantile(values, 0.98)
    normalized = ((depth - near) / (far - near)).clamp(0, 1)
    anchors = torch.tensor(
        [
            [0.05, 0.10, 0.55],
            [0.00, 0.75, 1.00],
            [0.95, 0.95, 0.10],
            [0.75, 0.05, 0.00],
        ],
        dtype=depth.dtype,
        device=depth.device,
    )
    position = normalized * (len(anchors) - 1)
    lower = position.floor().long().clamp(max=len(anchors) - 2)
    fraction = (position - lower).unsqueeze(-1)
    rgb = anchors[lower] * (1 - fraction) + anchors[lower + 1] * fraction
    rgb[~valid] = 0
    return rgb.permute(2, 0, 1)


def _annotate(panel: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    box = draw.textbbox((0, 0), text)
    draw.rectangle((0, 0, box[2] + 6, box[3] + 5), fill="black")
    draw.text((3, 2), text, fill="white")
    return panel


@torch.no_grad()
def render_dataset_geometry(
    views: Sequence[Mapping],
    reference_index: int,
    depth_threshold: float,
    cell_width: int,
) -> tuple[Image.Image, list[str]]:
    images = torch.stack([view["img"].float() for view in views])
    depths = torch.stack(
        [torch.as_tensor(view["depthmap"], dtype=torch.float32) for view in views]
    )
    intrinsics = torch.stack(
        [
            torch.as_tensor(view["camera_intrinsics"], dtype=torch.float32)
            for view in views
        ]
    )
    poses = torch.stack(
        [torch.as_tensor(view["camera_pose"], dtype=torch.float32) for view in views]
    )

    target_indices = [index for index in range(len(views)) if index != reference_index]
    target_from_reference = (
        torch.linalg.inv(poses[target_indices]) @ poses[reference_index]
    )
    supervision = build_ground_truth_warp(
        depths[reference_index].unsqueeze(0),
        depths[target_indices].unsqueeze(0),
        intrinsics[reference_index].unsqueeze(0),
        intrinsics[target_indices].unsqueeze(0),
        target_from_reference.unsqueeze(0),
        depth_threshold=depth_threshold,
    )

    image_height, image_width = images.shape[-2:]
    cell_height = round(image_height / image_width * cell_width)
    blank = Image.new("RGB", (cell_width, cell_height), "white")
    order = [reference_index, *target_indices]

    image_panels = []
    depth_panels = []
    for column, view_index in enumerate(order):
        role = "ref" if column == 0 else f"target {column}"
        image_panel = _resize_panel(
            images[view_index], cell_height, cell_width
        )
        image_panels.append(
            _annotate(image_panel, f"{role}: {views[view_index]['instance']}")
        )
        depth_panels.append(
            _resize_panel(_depth_to_rgb(depths[view_index]), cell_height, cell_width)
        )

    warp_panels = [image_panels[0]]
    confidence_panels = [blank]
    mask_panels = [blank]
    statistics = []
    for view_index, warp, confidence, mask in zip(
        target_indices,
        supervision.warp[0],
        supervision.confidence[0],
        supervision.mask[0],
    ):
        warped = sample_map_at_pixels(
            images[view_index].unsqueeze(0), warp.unsqueeze(0)
        )[0]
        warp_panels.append(
            _resize_panel(warped * confidence.unsqueeze(0), cell_height, cell_width)
        )
        confidence_panels.append(
            _resize_panel(
                confidence.float().unsqueeze(0).expand(3, -1, -1),
                cell_height,
                cell_width,
            )
        )
        mask_panels.append(
            _resize_panel(
                mask.float().unsqueeze(0).expand(3, -1, -1),
                cell_height,
                cell_width,
            )
        )
        statistics.append(
            f"ref {reference_index} -> target {view_index}: "
            f"conf={confidence.float().mean():.2%}, "
            f"mask={mask.float().mean():.2%}"
        )

    for view_index in order:
        depth = depths[view_index]
        valid_depth = depth[depth > 0]
        statistics.append(
            f"view {view_index} depth: "
            f"valid={len(valid_depth) / depth.numel():.2%}, "
            f"min/median/max={valid_depth.min():.3f}/"
            f"{valid_depth.median():.3f}/{valid_depth.max():.3f} m"
        )

    rows = [image_panels, depth_panels, warp_panels, confidence_panels, mask_panels]
    labels = ["Image", "Depth", "Warp", "Conf", "Mask"]
    label_width = 56
    canvas = Image.new(
        "RGB",
        (label_width + len(order) * cell_width, len(rows) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, panels) in enumerate(zip(labels, rows)):
        y = row_index * cell_height
        canvas.paste(_row_label(label, label_width, cell_height), (0, y))
        for column_index, panel in enumerate(panels):
            x = label_width + column_index * cell_width
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1), outline="gray"
            )
    return canvas, statistics


class DatasetGeometryVisualizer:
    """Original reference-to-target warp inspection."""

    def __init__(self, reference_index: int, depth_threshold: float, cell_width: int):
        self.reference_index = int(reference_index)
        self.depth_threshold = float(depth_threshold)
        self.cell_width = int(cell_width)

    def __call__(self, views: Sequence[Mapping]):
        return render_dataset_geometry(
            views,
            reference_index=self.reference_index,
            depth_threshold=self.depth_threshold,
            cell_width=self.cell_width,
        )


def _sequence_key(sequence):
    if isinstance(sequence, list):
        return tuple(sequence)
    if hasattr(sequence, "tolist"):
        value = sequence.tolist()
        return tuple(value) if isinstance(value, list) else value
    return sequence


def _dataset_volume(dataset) -> dict[str, int]:
    sequences = list(dataset.sequences)

    for attribute in ("num_imgs", "num_image"):
        image_counts = getattr(dataset, attribute, None)
        if isinstance(image_counts, Mapping):
            num_images = sum(
                int(image_counts[_sequence_key(sequence)])
                for sequence in sequences
            )
            return {
                "num_sequences": len(sequences),
                "num_images": num_images,
            }

    records = getattr(dataset, "records", None)
    if records is not None:
        frame_fields = ("frames", "frame_ids", "view_ids", "sample_tokens")
        num_images = 0
        for record in records:
            for field in frame_fields:
                if field in record:
                    num_images += len(record[field])
                    break
            else:
                raise ValueError(
                    f"Cannot determine image count for {type(dataset).__name__}: "
                    f"record has fields {sorted(record)}"
                )
        return {
            "num_sequences": len(sequences),
            "num_images": num_images,
        }

    raise ValueError(
        f"Cannot determine sequence and image counts for {type(dataset).__name__}"
    )


def _timing_summary(samples: Sequence[float]) -> dict[str, int | float | list[float]]:
    total_seconds = sum(samples)
    return {
        "num_samples": len(samples),
        "total_seconds": round(total_seconds, 6),
        "mean_seconds": round(total_seconds / len(samples), 6) if samples else 0.0,
        "samples_seconds": list(samples),
    }


def run(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_sizes: dict[str, dict[str, int]] = {}
    sample_timings: dict[str, dict[str, int | float | list[float]]] = {}
    sample_timings_path = output_dir / "sample_timings.json"
    visualizer = hydra.utils.instantiate(cfg.visualizers[cfg.visualizer_name])

    for dataset_name, dataset_cfg in cfg.datasets.items():
        if cfg.selected_datasets is not None and dataset_name not in cfg.selected_datasets:
            continue
        dataset = hydra.utils.instantiate(dataset_cfg)
        dataset_sizes[dataset_name] = _dataset_volume(dataset)
        elapsed_samples: list[float] = []
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        for sample_index in range(cfg.sample_index, cfg.sample_index + cfg.num_samples):
            dataset_index = sample_index % len(dataset)
            tic()
            views = dataset[dataset_index]
            elapsed_samples.append(
                round(toc(f"[{dataset_name}] sample {sample_index}"), 6)
            )
            overview, statistics = visualizer(views)
            output_path = dataset_output_dir / f"sample_{sample_index:04d}.png"
            overview.save(output_path)
            print(f"\n[{sample_index}] {views[0]['label']} -> {output_path.resolve()}")
            for line in statistics:
                print(f"  {line}")
            if cfg.show:
                overview.show()

        sample_timings[dataset_name] = _timing_summary(elapsed_samples)

    dataset_sizes_path = output_dir / "dataset_sizes.json"
    dataset_sizes_path.write_text(
        json.dumps(dataset_sizes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_timings_path.write_text(
        json.dumps(sample_timings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset sizes -> {dataset_sizes_path.resolve()}")
    print(f"Sample timings -> {sample_timings_path.resolve()}")


@hydra.main(
    version_base="1.2",
    config_path="../configs",
    config_name="dataset_viz.yaml",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
