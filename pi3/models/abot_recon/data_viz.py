"""Dataset-only visualization driven by the package-local Stage I config."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image, ImageDraw


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (torch.nan_to_num(image.detach().float()).clamp(0, 1)
             .permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _resize(image: torch.Tensor, height: int, width: int) -> Image.Image:
    value = F.interpolate(
        image[None], size=(height, width), mode="bilinear", align_corners=True)[0]
    return _to_pil(value)


def _depth_rgb(depth: torch.Tensor) -> torch.Tensor:
    valid = depth > 0
    values = depth[valid]
    near, far = torch.quantile(values, torch.tensor([0.02, 0.98]))
    value = ((depth - near) / (far - near).clamp_min(1e-6)).clamp(0, 1)
    anchors = torch.tensor(
        [[0.05, 0.10, 0.55], [0.00, 0.75, 1.00],
         [0.95, 0.95, 0.10], [0.75, 0.05, 0.00]], dtype=depth.dtype)
    position = value * (len(anchors) - 1)
    lower = position.floor().long().clamp(max=len(anchors) - 2)
    fraction = (position - lower)[..., None]
    rgb = anchors[lower] * (1 - fraction) + anchors[lower + 1] * fraction
    rgb[~valid] = 0
    return rgb.permute(2, 0, 1)


def _frame_number(view: Mapping) -> int:
    value = view.get("frame_id", view["instance"])
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"(\d+)(?!.*\d)", str(value))
        if match is None:
            raise ValueError(f"ABot-Recon view has no numeric frame id: {value}")
        return int(match.group(1))


def _selected_frames(frame_count: int, maximum: int) -> list[int]:
    count = min(frame_count, int(maximum))
    if count == 1:
        return [0]
    return torch.linspace(0, frame_count - 1, count).round().long().tolist()


@torch.no_grad()
def render_abot_recon_dataset(
    views: Sequence[Mapping], cell_width: int, max_frames: int
) -> tuple[Image.Image, list[str]]:
    """Render RGB, metric depth, and valid supervision in temporal order."""

    frame_ids = [_frame_number(view) for view in views]
    if frame_ids != sorted(frame_ids):
        raise ValueError(f"ABot-Recon views are not chronological: {frame_ids}")
    selected = _selected_frames(len(views), max_frames)
    images = torch.stack([view["img"].float() for view in views])
    if images.amin() < -0.05:
        images = images * 0.5 + 0.5
    depths = torch.stack([
        torch.as_tensor(view["depthmap"], dtype=torch.float32) for view in views])
    valid = torch.stack([
        torch.as_tensor(view["valid_mask"], dtype=torch.bool) for view in views])
    poses = torch.stack([
        torch.as_tensor(view["camera_pose"], dtype=torch.float32) for view in views])
    world_points = torch.stack([
        torch.as_tensor(view["pts3d"], dtype=torch.float32) for view in views])
    homogeneous = torch.cat((world_points, torch.ones_like(world_points[..., :1])), -1)
    local_points = torch.einsum(
        "nij,nhwj->nhwi", torch.linalg.inv(poses), homogeneous)[..., :3]

    height, width = images.shape[-2:]
    cell_height = round(height / width * cell_width)
    image_panels, depth_panels, mask_panels = [], [], []
    statistics = [
        f"ordered frames: {frame_ids}",
        f"imgs={tuple(images.shape)}, depth={tuple(depths.shape)}, "
        f"valid_mask={tuple(valid.shape)}, camera_pose={tuple(poses.shape)}",
    ]
    for frame in selected:
        image_panel = _resize(images[frame], cell_height, cell_width)
        ImageDraw.Draw(image_panel).text((3, 2), f"frame {frame_ids[frame]}", fill="red")
        image_panels.append(image_panel)
        depth_panels.append(_resize(_depth_rgb(depths[frame]), cell_height, cell_width))
        mask_panels.append(_resize(
            valid[frame].float()[None].expand(3, -1, -1), cell_height, cell_width))
        values = depths[frame][valid[frame]]
        lift_error = (local_points[frame, ..., 2] - depths[frame]).abs()[valid[frame]]
        statistics.append(
            f"frame {frame_ids[frame]} valid={valid[frame].float().mean():.2%} "
            f"depth={values.min():.3f}/{values.median():.3f}/{values.max():.3f}m "
            f"local-z-max-error={lift_error.max():.3g}m")

    rows = (("Image", image_panels), ("Depth", depth_panels), ("Valid", mask_panels))
    label_width = 64
    canvas = Image.new(
        "RGB", (label_width + len(selected) * cell_width, len(rows) * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (label, panels) in enumerate(rows):
        y = row * cell_height
        draw.text((5, y + cell_height // 2), label, fill="black")
        for column, panel in enumerate(panels):
            x = label_width + column * cell_width
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="gray")
    return canvas, statistics


class ABotReconDatasetVisualizer:
    def __init__(self, cell_width: int, max_frames: int = 8):
        self.cell_width = int(cell_width)
        self.max_frames = int(max_frames)

    def __call__(self, views: Sequence[Mapping]):
        return render_abot_recon_dataset(views, self.cell_width, self.max_frames)


@hydra.main(version_base="1.2", config_path=".", config_name="stage1")
def main(cfg: DictConfig) -> None:
    options = cfg.data_viz
    dataset = hydra.utils.instantiate(
        cfg.train_dataset.KITTIABotRecon,
        resolution=cfg.train.resolution,
        frame_num=options.frame_num,
        mode="test",
    )
    visualizer = ABotReconDatasetVisualizer(options.cell_width, options.max_frames)
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_index in range(options.sample_index, options.sample_index + options.num_samples):
        views = dataset[sample_index % len(dataset)]
        overview, statistics = visualizer(views)
        path = output_dir / f"sample_{sample_index:04d}.png"
        overview.save(path)
        print(path.resolve())
        for line in statistics:
            print(line)
        if options.show:
            overview.show()


if __name__ == "__main__":
    main()
