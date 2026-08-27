"""Pairwise loop-closure match visualization."""

from pathlib import Path

from PIL import Image, ImageDraw
import torch


def _rgb_image(image):
    """Convert one stored CHW float image into a PIL RGB image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("loop visualization image must have shape [3,H,W]")
    array = (
        image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).mul(255)
        .round().to(torch.uint8).numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _match_color(track_id):
    """Return a deterministic bright color for one stable track identity."""
    track_id = int(track_id)
    return (
        64 + (53 * track_id) % 192,
        64 + (97 * track_id) % 192,
        64 + (193 * track_id) % 192,
    )


def save_loop_match_images(output_dir, packets, frames, max_matches=256):
    """Render loop packets as antialiased pairwise line visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("loop_*.png"):
        stale.unlink()
    saved = []
    render_scale = 3
    resampling = Image.Resampling.LANCZOS
    for packet in packets:
        if packet.get("kind") != "loop":
            continue
        for part in packet["parts"]:
            reference = int(part["reference"])
            observation_frames = part["obs_frames"].detach().cpu().long()
            observation_points = part["obs_points"].detach().cpu().long()
            if "track_ids" in part:
                observation_points = (
                    part["track_ids"].detach().cpu().long()[observation_points]
                )
            observation_uv = part["obs_uv"].detach().cpu().float()
            reference_mask = observation_frames == reference
            reference_uv = {
                int(track_id): uv
                for track_id, uv in zip(
                    observation_points[reference_mask].tolist(),
                    observation_uv[reference_mask],
                )
            }
            cached = getattr(frames, "anchors", {}).get(reference)
            cached_queries = (
                cached[1].detach().cpu().float()
                if cached is not None
                else torch.stack(list(reference_uv.values()))
            )
            targets = observation_frames[~reference_mask].unique(sorted=True).tolist()
            for target in targets:
                target_mask = observation_frames == int(target)
                matches = [
                    (int(track_id), reference_uv[int(track_id)], uv)
                    for track_id, uv in zip(
                        observation_points[target_mask].tolist(),
                        observation_uv[target_mask],
                    )
                    if int(track_id) in reference_uv
                ]
                if not matches:
                    continue
                match_count = len(matches)
                if len(matches) > int(max_matches):
                    indices = torch.linspace(
                        0, len(matches) - 1, int(max_matches)
                    ).round().long().tolist()
                    matches = [matches[index] for index in indices]
                reference_image = _rgb_image(frames.dense[reference][0])
                target_image = _rgb_image(frames.dense[int(target)][0])
                header = 28
                width = reference_image.width + target_image.width
                height = header + max(reference_image.height, target_image.height)
                canvas = Image.new(
                    "RGB", (width * render_scale, height * render_scale)
                )
                canvas.paste(
                    reference_image.resize(
                        (
                            reference_image.width * render_scale,
                            reference_image.height * render_scale,
                        ),
                        resampling,
                    ),
                    (0, header * render_scale),
                )
                canvas.paste(
                    target_image.resize(
                        (
                            target_image.width * render_scale,
                            target_image.height * render_scale,
                        ),
                        resampling,
                    ),
                    (reference_image.width * render_scale, header * render_scale),
                )
                draw = ImageDraw.Draw(canvas)
                query_radius = render_scale
                for uv in cached_queries:
                    x = float(uv[0]) * render_scale
                    y = (header + float(uv[1])) * render_scale
                    draw.ellipse(
                        (
                            x - query_radius,
                            y - query_radius,
                            x + query_radius,
                            y + query_radius,
                        ),
                        fill=(160, 160, 160),
                    )
                for track_id, source_uv, target_uv in matches:
                    source_xy = (
                        float(source_uv[0]) * render_scale,
                        (header + float(source_uv[1])) * render_scale,
                    )
                    target_xy = (
                        (reference_image.width + float(target_uv[0])) * render_scale,
                        (header + float(target_uv[1])) * render_scale,
                    )
                    color = _match_color(track_id)
                    draw.line(
                        (source_xy, target_xy), fill=color, width=render_scale
                    )
                    for x, y in (source_xy, target_xy):
                        radius = 2 * render_scale
                        draw.ellipse(
                            (x - radius, y - radius, x + radius, y + radius),
                            fill=color,
                        )
                canvas = canvas.resize((width, height), resampling)
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (5, 7),
                    f"loop {reference} -> {int(target)}  "
                    f"queries={len(cached_queries)} matches={match_count} "
                    f"shown={len(matches)}",
                    fill=(255, 255, 255),
                )
                path = output_dir / f"loop_{reference:06d}_{int(target):06d}.png"
                canvas.save(path)
                saved.append(path)
    return saved


__all__ = ["save_loop_match_images"]
