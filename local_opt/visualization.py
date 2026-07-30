"""Image-grid visualization for local Glob3R matching inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pi3.models.glob3r.geometry import sample_map_at_pixels


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _resize_panel(image: torch.Tensor, height: int, width: int) -> Image.Image:
    resized = F.interpolate(
        image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=True
    )[0]
    return _tensor_to_pil(resized)


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
def render_keyframe_matching_overview(
    images: torch.Tensor,
    matching,
    reference_index: int,
    target_offset: int,
    num_targets: int = 7,
    cell_width: int = 160,
) -> Image.Image:
    """Render prediction-only ``Images / Warp / Conf`` rows for one reference."""

    all_targets = [int(index) for index in matching.target_indices]
    selected_targets = all_targets[target_offset : target_offset + num_targets]
    if not selected_targets:
        raise ValueError("matching overview requires at least one target")

    final_warp = matching.warp_stages[-1] if matching.warp_stages else matching.coarse_warp
    final_confidence = (
        matching.confidence_stages[-1]
        if matching.confidence_stages
        else matching.coarse_confidence
    )
    reference = images[reference_index].detach().float()
    image_height, image_width = reference.shape[-2:]
    cell_height = max(round(image_height / image_width * cell_width), 1)
    blank = Image.new("RGB", (cell_width, cell_height), "white")

    frame_indices = [reference_index, *selected_targets]
    image_panels = [
        _resize_panel(images[index].detach().float(), cell_height, cell_width)
        for index in frame_indices
    ]
    warp_panels = [image_panels[0]]
    confidence_panels = [blank]

    for target_index in selected_targets:
        matching_offset = all_targets.index(target_index)
        warp = final_warp[0, matching_offset].detach().float()
        confidence = final_confidence[0, matching_offset, 0].detach().float()
        target = images[target_index].detach().float()
        # Eq. (2): W^(a->b)(p_ref)=p_target. The sampled image is therefore
        # defined on the reference grid and should resemble the reference.
        warped = sample_map_at_pixels(
            target.unsqueeze(0), warp.permute(1, 2, 0).unsqueeze(0)
        )[0]
        valid = (
            (warp[0] >= 0)
            & (warp[0] <= image_width - 1)
            & (warp[1] >= 0)
            & (warp[1] <= image_height - 1)
            & torch.isfinite(warp).all(dim=0)
            & (confidence > 0.6)
        )
        warp_panels.append(
            _resize_panel(warped * valid.unsqueeze(0), cell_height, cell_width)
        )
        confidence_panels.append(
            _resize_panel(confidence.unsqueeze(0).expand(3, -1, -1), cell_height, cell_width)
        )

    rows = [image_panels, warp_panels, confidence_panels]
    labels = ["Images", "Warp", "Conf"]
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


@torch.no_grad()
def save_keyframe_matching_overviews(
    output_dir: str | Path,
    images: torch.Tensor,
    matches: Mapping[int, object],
    num_targets: int = 7,
    cell_width: int = 160,
) -> list[Path]:
    """Save paginated prediction-only matching grids using global frame indices."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()
    saved = []
    for reference_index, matching in matches.items():
        target_count = len(matching.target_indices)
        for target_offset in range(0, target_count, num_targets):
            page_targets = matching.target_indices[
                target_offset : target_offset + num_targets
            ]
            overview = render_keyframe_matching_overview(
                images,
                matching,
                int(reference_index),
                target_offset,
                num_targets=num_targets,
                cell_width=cell_width,
            )
            path = output_dir / (
                f"reference_{int(reference_index):04d}_targets_"
                f"{int(page_targets[0]):04d}_{int(page_targets[-1]):04d}.png"
            )
            overview.save(path)
            saved.append(path)
    return saved


__all__ = ["render_keyframe_matching_overview", "save_keyframe_matching_overviews"]
