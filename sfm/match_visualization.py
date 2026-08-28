"""Five-column sliding-window match and Pi3 geometry visualization."""

import colorsys
from pathlib import Path

from PIL import Image, ImageDraw
import torch


def _rgb_image(image):
    """Convert one stored CHW float image into a PIL RGB image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("match visualization image must have shape [3,H,W]")
    array = (
        image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).mul(255)
        .round().to(torch.uint8).numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _heat_rgb(values):
    """Map normalized scalar values to a compact blue-to-red heat palette."""
    values = values.clamp(0, 1)
    red = (1.5 - (4 * values - 3).abs()).clamp(0, 1)
    green = (1.5 - (4 * values - 2).abs()).clamp(0, 1)
    blue = (1.5 - (4 * values - 1).abs()).clamp(0, 1)
    return torch.stack((red, green, blue), dim=-1)


def _scalar_panel(values, width, height, normalize=False):
    """Render a dense confidence or depth tensor as one RGB panel."""
    if values is None:
        return Image.new("RGB", (width, height), "black")
    values = values.detach().cpu().float()
    if values.ndim != 2:
        raise ValueError("scalar visualization must have shape [H,W]")
    valid = torch.isfinite(values)
    if normalize:
        valid &= values > 0
        samples = values[valid]
        if samples.numel():
            low, high = torch.quantile(
                samples, samples.new_tensor((0.02, 0.98))
            )
            scale = (high - low).clamp_min(1.0e-8)
            normalized = (values - low) / scale
        else:
            normalized = torch.zeros_like(values)
    else:
        normalized = values
    normalized = torch.where(valid, normalized.clamp(0, 1), 0)
    rgb = _heat_rgb(normalized)
    rgb = torch.where(valid[..., None], rgb, torch.zeros_like(rgb))
    array = rgb.mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB").resize(
        (width, height), Image.Resampling.BICUBIC
    )


def _sparse_confidence_panel(
    points, values, source_width, source_height, width, height, opacity=None
):
    """Rasterize sparse color values with optional per-point opacity."""
    panel = Image.new("RGB", (width, height), "black")
    if points is None or values is None:
        return panel
    points = points.detach().cpu().float()
    values = values.detach().cpu().float()
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("sparse confidence points must have shape [P,2]")
    if values.shape != points.shape[:1]:
        raise ValueError("sparse confidence values must have shape [P]")
    if opacity is not None:
        opacity = opacity.detach().cpu().float()
        if opacity.shape != points.shape[:1]:
            raise ValueError("sparse confidence opacity must have shape [P]")
    valid = (
        torch.isfinite(points).all(-1)
        & torch.isfinite(values)
        & (points[:, 0] >= 0)
        & (points[:, 0] <= source_width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= source_height - 1)
    )
    if opacity is not None:
        valid &= torch.isfinite(opacity)
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if not indices.numel():
        return panel
    strength = values[indices].clamp(0, 1)
    if opacity is not None:
        strength *= opacity[indices].clamp(0, 1)
    indices = indices[strength.argsort()]
    x_scale = (width - 1) / max(source_width - 1, 1)
    y_scale = (height - 1) / max(source_height - 1, 1)
    radius = max(round(width / 112), 2)
    draw = ImageDraw.Draw(panel, "RGBA")
    for index in indices.tolist():
        x = float(points[index, 0]) * x_scale
        y = float(points[index, 1]) * y_scale
        color = tuple(
            int(channel)
            for channel in _heat_rgb(values[index].clamp(0, 1)).mul(255).round()
        )
        alpha = (
            255
            if opacity is None
            else round(float(opacity[index].clamp(0, 1)) * 255)
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, alpha),
        )
    return panel


def _track_color(track_id):
    """Match the golden-ratio HSV colors used by the committed VGGSfM view."""
    hue = (int(track_id) * 0.618033988749895) % 1.0
    return tuple(
        round(channel * 255)
        for channel in colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    )


def _matrix_groups(packets):
    """Collect sliding windows, raw frontend tracks, and verified observations."""
    groups = {}
    for packet in packets:
        if packet.get("kind", "sliding") != "sliding":
            continue
        frame_ids = list(map(int, packet["frame_ids"].tolist()))
        for part in packet["parts"]:
            reference = int(part["reference"])
            group = groups.setdefault(
                reference,
                {
                    "frontend": packet.get("frontend", "unknown"),
                    "targets": set(),
                    "matches": {},
                },
            )
            group["targets"].update(frame_ids)
            observation_frames = part["obs_frames"].detach().cpu().long()
            observation_uv = part["obs_uv"].detach().cpu().float()
            observation_weights = part["obs_weights"].detach().cpu().float()
            local_points = part["obs_points"].detach().cpu().long()
            track_ids = part["track_ids"].detach().cpu().long()[local_points]
            reference_mask = observation_frames == reference
            reference_uv = {
                int(track_id): uv
                for track_id, uv in zip(
                    track_ids[reference_mask].tolist(),
                    observation_uv[reference_mask],
                )
            }
            for target in observation_frames.unique(sorted=True).tolist():
                target = int(target)
                target_mask = observation_frames == target
                target_matches = group["matches"].setdefault(target, {})
                for track_id, uv, weight in zip(
                    track_ids[target_mask].tolist(),
                    observation_uv[target_mask],
                    observation_weights[target_mask].tolist(),
                ):
                    track_id = int(track_id)
                    if track_id not in reference_uv:
                        continue
                    candidate = (reference_uv[track_id], uv, float(weight))
                    previous = target_matches.get(track_id)
                    if previous is None or candidate[2] > previous[2]:
                        target_matches[track_id] = candidate

        for diagnostic in packet.get("visualization", []):
            reference = int(diagnostic["reference"])
            diagnostic_frame_ids = diagnostic.get("frame_ids", packet["frame_ids"])
            diagnostic_frame_ids = list(
                map(int, diagnostic_frame_ids.detach().cpu().tolist())
            )
            query_points = diagnostic["query_points"].detach().cpu().float()
            raw_tracks = diagnostic["raw_tracks"].detach().cpu().float()
            visualization_confidence = diagnostic.get("visualization_confidence")
            visualization_confidence = (
                None
                if visualization_confidence is None
                else visualization_confidence.detach().cpu().float()
            )
            visualization_score = diagnostic.get("visualization_score")
            visualization_score = (
                None
                if visualization_score is None
                else visualization_score.detach().cpu().float()
            )
            confidence_label = diagnostic.get(
                "visualization_confidence_label", "confidence"
            )
            frontend_valid = diagnostic.get("frontend_valid")
            frontend_valid = (
                None
                if frontend_valid is None
                else frontend_valid.detach().cpu().bool()
            )
            if query_points.ndim != 2 or query_points.shape[-1] != 2:
                raise ValueError("visualization query_points must have shape [P,2]")
            expected_shape = (
                len(diagnostic_frame_ids), query_points.shape[0], 2
            )
            if tuple(raw_tracks.shape) != expected_shape:
                raise ValueError(
                    "visualization raw_tracks must have shape [S,P,2]"
                )
            if frontend_valid is not None and tuple(frontend_valid.shape) != (
                len(diagnostic_frame_ids), query_points.shape[0]
            ):
                raise ValueError(
                    "visualization frontend_valid must have shape [S,P]"
                )
            if visualization_confidence is not None:
                sparse_shape = (len(diagnostic_frame_ids), query_points.shape[0])
                dense = (
                    visualization_confidence.ndim == 3
                    and visualization_confidence.shape[0]
                    == len(diagnostic_frame_ids)
                )
                if tuple(visualization_confidence.shape) != sparse_shape and not dense:
                    raise ValueError(
                        "visualization confidence must have shape [S,P] or [S,H,W]"
                    )
            if visualization_score is not None and tuple(
                visualization_score.shape
            ) != (len(diagnostic_frame_ids), query_points.shape[0]):
                raise ValueError("visualization score must have shape [S,P]")

            group = groups.setdefault(
                reference,
                {
                    "frontend": diagnostic.get(
                        "frontend", packet.get("frontend", "unknown")
                    ),
                    "targets": set(),
                    "matches": {},
                    "raw": {},
                },
            )
            group.setdefault("raw", {})
            group["frontend"] = diagnostic.get(
                "frontend", group.get("frontend", "unknown")
            )
            group["targets"].update(diagnostic_frame_ids)
            for local_target, target in enumerate(diagnostic_frame_ids):
                candidate = {
                    "query_points": query_points,
                    "target_points": raw_tracks[local_target],
                    "frontend_valid": (
                        None
                        if frontend_valid is None
                        else frontend_valid[local_target]
                    ),
                    "confidence": (
                        None
                        if visualization_confidence is None
                        else visualization_confidence[local_target]
                    ),
                    "score": (
                        None
                        if visualization_score is None
                        else visualization_score[local_target]
                    ),
                    "confidence_label": confidence_label,
                }
                previous = group["raw"].get(target)
                if previous is None or _raw_candidate_rank(candidate) > (
                    _raw_candidate_rank(previous)
                ):
                    # Overlapping windows may contain the same pair.  Keep one
                    # complete frontend result instead of concatenating queries.
                    group["raw"][target] = candidate
    return groups


def _raw_candidate_rank(candidate):
    """Prefer the most complete duplicate diagnostic deterministically."""
    finite = torch.isfinite(candidate["target_points"]).all(dim=-1)
    frontend_valid = candidate["frontend_valid"]
    frontend_count = -1 if frontend_valid is None else int(frontend_valid.sum())
    return frontend_count, int(finite.sum()), int(finite.numel())


def _raw_masks(raw, width, height):
    """Return drawable raw queries, raw targets, and frontend-pass targets."""
    query_points = raw["query_points"]
    target_points = raw["target_points"]
    query_drawable = torch.isfinite(query_points).all(dim=-1)
    target_drawable = (
        torch.isfinite(target_points).all(dim=-1)
        & (target_points[:, 0] >= 0)
        & (target_points[:, 0] <= width - 1)
        & (target_points[:, 1] >= 0)
        & (target_points[:, 1] <= height - 1)
    )
    frontend_valid = raw["frontend_valid"]
    frontend_pass = (
        None
        if frontend_valid is None
        else target_drawable & frontend_valid
    )
    return query_drawable, target_drawable, frontend_pass


def _confidence_panel(raw, matches, source_width, source_height, width, height):
    """Render dense Glob3R confidence or sparse VGGSfM visibility."""
    if raw is not None and raw.get("confidence") is not None:
        confidence = raw["confidence"]
        if confidence.ndim == 2:
            return _scalar_panel(confidence, width, height), raw.get(
                "confidence_label", "confidence"
            )
        if confidence.ndim == 1:
            return _sparse_confidence_panel(
                raw["target_points"],
                confidence,
                source_width,
                source_height,
                width,
                height,
                opacity=raw.get("score"),
            ), raw.get("confidence_label", "confidence")
        raise ValueError("pair confidence must have shape [P] or [H,W]")
    if matches:
        points = torch.stack([value[1] for value in matches.values()])
        values = torch.tensor([value[2] for value in matches.values()])
        return _sparse_confidence_panel(
            points,
            values,
            source_width,
            source_height,
            width,
            height,
        ), "verified confidence"
    return Image.new("RGB", (width, height), "black"), "confidence"


def _panel_title(draw, panel, title, cell_width):
    """Draw a compact label inside one column without crossing panel bounds."""
    if cell_width < 16:
        return
    left = panel * cell_width + 4
    right = min(left + max(8 * len(title), 24), (panel + 1) * cell_width - 4)
    draw.rectangle((left, 4, right, 20), fill=(0, 0, 0, 210))
    draw.text((left + 4, 6), title, fill=(255, 255, 255, 255))


def _row_label(reference, target, raw_count, frontend_count, geometry_count, shown):
    if raw_count is None:
        return (
            f"reference {reference:04d}  target {target:04d}  "
            f"shown {shown} / valid {geometry_count}"
        )
    frontend = "n/a" if frontend_count is None else str(frontend_count)
    return (
        f"reference {reference:04d}  target {target:04d}  "
        f"raw {raw_count}  frontend {frontend}  "
        f"geometry {geometry_count}  shown {shown}"
    )


def save_match_images(
    output_dir,
    packets,
    frames,
    dataset,
    max_matches=128,
    cell_width=448,
):
    """Render reference, target, frontend confidence, Pi3 depth, and Pi3 confidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("match_*.png", "reference_*.png"):
        for stale in output_dir.glob(pattern):
            stale.unlink()
    cell_width = int(cell_width)
    saved = []
    for reference, group in sorted(_matrix_groups(packets).items()):
        reference_image = _rgb_image(frames.dense[reference][0])
        cell_height = max(
            round(reference_image.height / reference_image.width * cell_width), 1
        )

        def image_panel(frame_id):
            stored = frames.dense.get(int(frame_id))
            tensor = (
                dataset.read([int(frame_id)])[0]
                if stored is None
                else stored[0]
            )
            return _rgb_image(tensor).resize(
                (cell_width, cell_height), Image.Resampling.BICUBIC
            )

        targets = sorted(group["targets"])
        canvas = Image.new(
            "RGB", (5 * cell_width, len(targets) * cell_height), "black"
        )
        x_scale = (cell_width - 1) / max(reference_image.width - 1, 1)
        y_scale = (cell_height - 1) / max(reference_image.height - 1, 1)
        reference_panel = image_panel(reference)
        for row_index, target in enumerate(targets):
            raw = group.get("raw", {}).get(target)
            matches = group["matches"].get(target, {})
            confidence_panel, confidence_label = _confidence_panel(
                raw,
                matches,
                reference_image.width,
                reference_image.height,
                cell_width,
                cell_height,
            )
            stored = frames.dense.get(int(target))
            depth = None if stored is None or len(stored) < 2 else stored[1]
            pi3_confidence = (
                None if stored is None or len(stored) < 3 else stored[2]
            )
            row = Image.new("RGB", (5 * cell_width, cell_height), "black")
            row.paste(reference_panel, (0, 0))
            row.paste(image_panel(target), (cell_width, 0))
            row.paste(confidence_panel, (2 * cell_width, 0))
            row.paste(
                _scalar_panel(depth, cell_width, cell_height, normalize=True),
                (3 * cell_width, 0),
            )
            row.paste(
                _scalar_panel(
                    pi3_confidence, cell_width, cell_height, normalize=False
                ),
                (4 * cell_width, 0),
            )
            draw = ImageDraw.Draw(row, "RGBA")
            raw_count = frontend_count = None
            if raw is not None:
                query_drawable, target_drawable, frontend_pass = _raw_masks(
                    raw, reference_image.width, reference_image.height
                )
                raw_count = int(target_drawable.sum())
                frontend_count = (
                    None
                    if frontend_pass is None
                    else int(frontend_pass.sum())
                )
                for uv in raw["query_points"][query_drawable]:
                    x = float(uv[0]) * x_scale
                    y = float(uv[1]) * y_scale
                    radius = 2.0
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(180, 180, 180, 140),
                    )
                for uv in raw["target_points"][target_drawable]:
                    x = cell_width + float(uv[0]) * x_scale
                    y = float(uv[1]) * y_scale
                    radius = 2.0
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(180, 180, 180, 140),
                    )
            valid_count = len(matches)
            selected = sorted(
                matches.items(), key=lambda item: (-item[1][2], item[0])
            )[:int(max_matches)]
            for track_id, (source_uv, target_uv, _) in selected:
                color = _track_color(track_id)
                for panel, uv in ((0, source_uv), (1, target_uv)):
                    x = panel * cell_width + float(uv[0]) * x_scale
                    y = float(uv[1]) * y_scale
                    radius = 2.5
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(*color, 255),
                    )
            label = _row_label(
                reference,
                target,
                raw_count,
                frontend_count,
                valid_count,
                len(selected),
            )
            if cell_width >= 16:
                label_width = min(
                    2 * cell_width - 4,
                    4 + (445 if raw_count is not None else 375),
                )
                draw.rectangle((4, 4, label_width, 23), fill=(0, 0, 0, 210))
                draw.text((8, 7), label, fill=(255, 255, 255, 255))
            _panel_title(draw, 2, confidence_label, cell_width)
            _panel_title(draw, 3, "pi3 depth", cell_width)
            _panel_title(draw, 4, "pi3 confidence", cell_width)
            canvas.paste(row, (0, row_index * cell_height))
        path = output_dir / f"reference_{reference:04d}.png"
        canvas.save(path)
        saved.append(path)
    return saved


__all__ = ["save_match_images"]
