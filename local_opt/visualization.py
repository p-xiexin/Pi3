"""Overlay exact BA correspondences on the existing matching overview."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pi3.models.glob3r.geometry import sample_map_at_pixels

from .factor_graph import DroidFactorGraph
from .matching import PairMatch


def _tensor_to_pil(I: torch.Tensor) -> Image.Image:
    I = I.detach().float().cpu().clamp(0, 1)
    array = (I.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _resize_panel(I: torch.Tensor, height: int, width: int) -> Image.Image:
    resized = F.interpolate(
        I.unsqueeze(dim=0),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    ).squeeze(dim=0)
    return _tensor_to_pil(resized)


def _jet_colors(indices: torch.Tensor, count: int) -> list[tuple[int, int, int]]:
    values = indices.float() / max(count - 1, 1)
    red = (1.5 - (4 * values - 3).abs()).clamp(0, 1)
    green = (1.5 - (4 * values - 2).abs()).clamp(0, 1)
    blue = (1.5 - (4 * values - 1).abs()).clamp(0, 1)
    return [
        tuple(int(channel * 255) for channel in color)
        for color in torch.stack((red, green, blue), dim=-1).tolist()
    ]


def _draw_points(
    I: torch.Tensor,
    ps: torch.Tensor,
    colors: list[tuple[int, int, int]],
    height: int,
    width: int,
    supersample: int = 4,
) -> Image.Image:
    """Draw factor centers without rounding their target coordinates."""

    image_height, image_width = I.shape[-2:]
    ps = ps.clone()
    ps[:, 0] *= (width - 1) / (image_width - 1)
    ps[:, 1] *= (height - 1) / (image_height - 1)
    image = _resize_panel(I, height, width).resize(
        (width * supersample, height * supersample),
        resample=Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(image)
    radius = 1.25 * supersample
    for point, color in zip(ps.tolist(), colors):
        x, y = point
        x *= supersample
        y *= supersample
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


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
    Is: torch.Tensor,
    matches: list[PairMatch],
    graph: DroidFactorGraph,
    reference_index: int,
    target_offset: int,
    num_targets: int = 7,
    cell_width: int | None = None,
) -> Image.Image:
    """Render one reference and several outgoing BA edges."""

    reference_edges = torch.nonzero(
        graph.rs == reference_index,
        as_tuple=False,
    ).flatten()
    edge_indices = reference_edges[target_offset : target_offset + num_targets]
    image_height, image_width = Is.shape[-2:]
    cell_width = image_width if cell_width is None else cell_width
    cell_height = max(round(image_height / image_width * cell_width), 1)
    blank = Image.new("RGB", (cell_width, cell_height), "white")
    low_height, low_width = graph.target.shape[2:4]

    # All outgoing edges share the same two-dimensional source factor grid.
    # The reference panel shows the union for this page; each target panel
    # shows only its own final graph.weight > 0 support.
    page_valid = graph.weight[0, edge_indices, ..., 0] > 0
    reference_valid = page_valid.any(dim=0)
    ys_r, xs_r = torch.nonzero(reference_valid, as_tuple=True)
    reference_indices = ys_r * low_width + xs_r
    ps_r = torch.stack((xs_r, ys_r), dim=-1).float() * graph.stride
    reference_panel = _draw_points(
        Is[reference_index],
        ps_r,
        _jet_colors(reference_indices, low_height * low_width),
        cell_height,
        cell_width,
    )

    image_panels = [reference_panel]
    warp_panels = [_resize_panel(Is[reference_index], cell_height, cell_width)]
    confidence_panels = [blank]
    column_titles = [f"reference {reference_index}"]
    for edge_index in edge_indices:
        edge = int(edge_index)
        target_index = int(graph.ts[edge])
        match = matches[edge]
        valid = graph.weight[0, edge, ..., 0] > 0
        ys_r, xs_r = torch.nonzero(valid, as_tuple=True)
        source_indices = ys_r * low_width + xs_r
        ps_t = graph.target[0, edge, ys_r, xs_r] * graph.stride
        colors = _jet_colors(source_indices, low_height * low_width)
        image_panels.append(
            _draw_points(
                Is[target_index],
                ps_t,
                colors,
                cell_height,
                cell_width,
            )
        )

        # Keep the existing full-resolution Warp and Conf rows unchanged.
        W_r2t = match.W_r2t.to(device=Is.device, dtype=torch.float32)
        Q_r2t = match.Q_r2t.to(device=Is.device, dtype=torch.float32)
        I_t2r = sample_map_at_pixels(
            Is[target_index].unsqueeze(dim=0),
            W_r2t.unsqueeze(dim=0),
        ).squeeze(dim=0)
        warp_valid = (
            (W_r2t[..., 0] >= 0)
            & (W_r2t[..., 0] <= image_width - 1)
            & (W_r2t[..., 1] >= 0)
            & (W_r2t[..., 1] <= image_height - 1)
            & torch.isfinite(W_r2t).all(dim=-1)
            & (Q_r2t > 0.6)
        )
        warp_panels.append(
            _resize_panel(
                I_t2r * warp_valid.unsqueeze(dim=0),
                cell_height,
                cell_width,
            )
        )

        confidence_panels.append(
            _resize_panel(
                Q_r2t.unsqueeze(dim=0).expand(3, -1, -1),
                cell_height,
                cell_width,
            )
        )
        column_titles.append(
            f"{reference_index} -> {target_index} | factors={int(valid.sum())}"
        )

    rows = [image_panels, warp_panels, confidence_panels]
    labels = ["Images", "Warp", "Conf"]
    label_width = 72
    header_height = 28
    canvas = Image.new(
        "RGB",
        (
            label_width + len(image_panels) * cell_width,
            header_height + len(rows) * cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column_index, title in enumerate(column_titles):
        draw.text(
            (label_width + column_index * cell_width + 5, 7),
            title,
            fill="black",
        )
    for row_index, (label, panels) in enumerate(zip(labels, rows)):
        y = header_height + row_index * cell_height
        canvas.paste(_row_label(label, label_width, cell_height), (0, y))
        for column_index, panel in enumerate(panels):
            x = label_width + column_index * cell_width
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="gray",
            )
    return canvas


@torch.no_grad()
def save_keyframe_matching_overviews(
    output_dir: str | Path,
    Is: torch.Tensor,
    matches: list[PairMatch],
    graph: DroidFactorGraph,
    num_targets: int = 7,
    cell_width: int | None = None,
) -> list[Path]:
    """Save paginated overviews of the exact BA edges."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()

    Is = Is.detach().float().cpu()
    graph = DroidFactorGraph(
        rs=graph.rs.detach().cpu(),
        ts=graph.ts.detach().cpu(),
        target=graph.target.detach().float().cpu(),
        weight=graph.weight.detach().float().cpu(),
        disps=graph.disps.detach().float().cpu(),
        intrinsics=graph.intrinsics.detach().float().cpu(),
        damping=graph.damping.detach().float().cpu(),
        stride=graph.stride,
    )
    matches = [
        PairMatch(
            r=match.r,
            t=match.t,
            W_r2t=match.W_r2t.detach().float().cpu(),
            valid_r2t=match.valid_r2t.detach().cpu(),
            Q_r2t=match.Q_r2t.detach().float().cpu(),
        )
        for match in matches
    ]
    saved = []
    references = list(dict.fromkeys(graph.rs.tolist()))
    for reference_index in references:
        reference_edges = torch.nonzero(
            graph.rs == reference_index,
            as_tuple=False,
        ).flatten()
        for target_offset in range(0, reference_edges.numel(), num_targets):
            page_edges = reference_edges[target_offset : target_offset + num_targets]
            page_targets = graph.ts[page_edges]
            overview = render_keyframe_matching_overview(
                Is,
                matches,
                graph,
                reference_index,
                target_offset,
                num_targets=num_targets,
                cell_width=cell_width,
            )
            path = output_dir / (
                f"reference_{reference_index:04d}_targets_"
                f"{int(page_targets[0]):04d}_{int(page_targets[-1]):04d}.png"
            )
            overview.save(path)
            saved.append(path)
    return saved


__all__ = ["render_keyframe_matching_overview", "save_keyframe_matching_overviews"]
