"""Render dense Glob3R matching and Pi3 geometry as a frame matrix."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pi3.models.glob3r.geometry import sample_map_at_pixels

from .frame import Frames
from .matching import Tracks


def _panel(I: torch.Tensor, height: int, width: int) -> Image.Image:
    I = F.interpolate(
        I.unsqueeze(dim=0),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    ).squeeze(dim=0)
    array = (
        I.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _gray(value: torch.Tensor) -> torch.Tensor:
    return value.float().clamp(0, 1).unsqueeze(dim=0).expand(3, -1, -1)


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


def _track_colors(ks: torch.Tensor) -> list[tuple[int, int, int]]:
    count = int(ks.max()) + 1 if ks.numel() else 0
    values = ks.detach().float().cpu() / max(count - 1, 1)
    rgb = _jet(values)
    return [
        tuple(int(channel * 255) for channel in color)
        for color in rgb.permute(1, 0).tolist()
    ]


def _draw_tracks(
    I: torch.Tensor,
    us: torch.Tensor,
    colors: list[tuple[int, int, int]],
    height: int,
    width: int,
) -> Image.Image:
    H, W = I.shape[-2:]
    scale = 2
    image = _panel(I, height, width).resize(
        (scale * width, scale * height),
        resample=Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(image)
    us = us.detach().float().cpu().clone()
    us[:, 0] *= scale * (width - 1) / (W - 1)
    us[:, 1] *= scale * (height - 1) / (H - 1)
    radius = 2.0 * scale
    for (x, y), color in zip(us.tolist(), colors):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def _final_warp(output, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    W = output.warp_stages[-1] if output.warp_stages else output.coarse_warp
    Q = (
        output.confidence_stages[-1]
        if output.confidence_stages
        else output.coarse_confidence
    )
    # [B, T, 2, h, w] -> [T, 2, H, W]
    W = W.squeeze(dim=0)
    # [B, T, 1, h, w] -> [T, 1, H, W]
    Q = Q.squeeze(dim=0)
    if W.shape[-2:] != size:
        W = F.interpolate(W, size=size, mode="bilinear", align_corners=True)
        Q = F.interpolate(Q, size=size, mode="bilinear", align_corners=True)
    return W, Q.squeeze(dim=1)


@torch.no_grad()
def save_matching_matrix(
    output_dir: str | Path,
    frames: Frames,
    reference_index: int,
    output,
    tracks: Tracks,
    cell_width: int | None = None,
) -> Path:
    """Save rows of frames and columns of image/warp/confidence/geometry."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    S, _, H, W = frames.Is.shape
    cell_width = W if cell_width is None else cell_width
    cell_height = max(round(H / W * cell_width), 1)
    column_count = 5

    Ws, Qs = _final_warp(output, (H, W))
    colors = _track_colors(tracks.ks)
    Ds = frames.Xs_C[..., 2]
    depth_valid = torch.isfinite(Ds) & (Ds > 0)
    depth_limits = torch.quantile(Ds[depth_valid].float(), Ds.new_tensor([0.02, 0.98]))
    D_min, D_max = depth_limits.unbind()

    canvas = Image.new(
        "RGB",
        (column_count * cell_width, S * cell_height),
        "black",
    )
    draw = ImageDraw.Draw(canvas)

    for t in range(S):
        if t == reference_index:
            I_warp = frames.Is[t]
            Q = torch.zeros(H, W, device=frames.Is.device, dtype=frames.Is.dtype)
        else:
            target_offset = output.target_indices.index(t)
            W_r2t = Ws[target_offset].permute(1, 2, 0)
            Q = Qs[target_offset]
            finite = torch.isfinite(W_r2t).all(dim=-1)
            valid = (
                finite
                & (W_r2t[..., 0] >= 0)
                & (W_r2t[..., 0] <= W - 1)
                & (W_r2t[..., 1] >= 0)
                & (W_r2t[..., 1] <= H - 1)
                & (Q > 0.6)
            )
            W_r2t = torch.where(finite.unsqueeze(dim=-1), W_r2t, 0)
            I_warp = sample_map_at_pixels(
                frames.Is[t].unsqueeze(dim=0),
                W_r2t.unsqueeze(dim=0),
            ).squeeze(dim=0)
            I_warp = I_warp * valid.unsqueeze(dim=0)

        D = ((Ds[t] - D_min) / (D_max - D_min).clamp_min(1.0e-8)).clamp(0, 1)
        D_rgb = _jet(D) * depth_valid[t].unsqueeze(dim=0)
        track_mask = tracks.mask[t]
        image_colors = [
            color for color, keep in zip(colors, track_mask.tolist()) if keep
        ]
        I_tracks = _draw_tracks(
            frames.Is[t],
            tracks.us[t, track_mask],
            image_colors,
            cell_height,
            cell_width,
        )
        label = f"frame {t:04d}" + (" (ref)" if t == reference_index else "")
        label_draw = ImageDraw.Draw(I_tracks)
        label_box = label_draw.textbbox((5, 5), label)
        label_draw.rectangle(
            (
                label_box[0] - 3,
                label_box[1] - 3,
                label_box[2] + 3,
                label_box[3] + 3,
            ),
            fill="black",
        )
        label_draw.text((5, 5), label, fill="white")
        panels = (
            I_tracks,
            I_warp,
            _gray(Q),
            D_rgb,
            _gray(frames.Cs[t]),
        )

        y = t * cell_height
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

    path = output_dir / f"reference_{reference_index:04d}.png"
    canvas.save(path)
    return path


__all__ = ["save_matching_matrix"]
