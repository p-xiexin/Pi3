"""Package-local TensorBoard diagnostics for ABot-Recon training.

The reconstruction grid exposes the dense terms behind report Eq. (15), while
the trajectory grid exposes the composed camera chain behind Eqs. (5), (12),
and (14).  Rendering is deliberately detached and interval-gated so the CPU
transfer and percentile calculations do not enter the training graph.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

from pi3.models.abot_recon.loss import _rotation_angle


def _to_uint8_rgb(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _rgb_panel(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu()
    if image.amin() < -0.05:
        image = image * 0.5 + 0.5
    return _to_uint8_rgb(image)


def _heatmap(value: torch.Tensor, valid: torch.Tensor | None = None,
             minimum: float | None = None, maximum: float | None = None) -> Image.Image:
    """Map one scalar field to a compact blue-cyan-yellow-red heat map."""

    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    if valid is not None:
        finite &= valid.detach().bool().cpu()
    samples = value[finite]
    if samples.numel() == 0:
        return Image.new("RGB", (value.shape[-1], value.shape[-2]), "black")
    lower = float(torch.quantile(samples, 0.02)) if minimum is None else float(minimum)
    upper = float(torch.quantile(samples, 0.98)) if maximum is None else float(maximum)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        upper = lower + 1.0
    x = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
    red = (1.5 - (4.0 * x - 3.0).abs()).clamp(0.0, 1.0)
    green = (1.5 - (4.0 * x - 2.0).abs()).clamp(0.0, 1.0)
    blue = (1.5 - (4.0 * x - 1.0).abs()).clamp(0.0, 1.0)
    color = torch.stack((red, green, blue), 0)
    color[:, ~finite] = 0.0
    return _to_uint8_rgb(color)


def _shared_range(first: torch.Tensor, second: torch.Tensor,
                  valid: torch.Tensor) -> tuple[float, float]:
    mask = valid.detach().bool()
    values = torch.cat((first.detach()[mask], second.detach()[mask])).float().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0, 1.0
    lower = float(torch.quantile(values, 0.02))
    upper = float(torch.quantile(values, 0.98))
    return lower, upper if upper > lower else lower + 1.0


def _gray_panel(value: torch.Tensor, valid: torch.Tensor | None = None) -> Image.Image:
    value = value.detach().float().cpu().clamp(0.0, 1.0)
    if valid is not None:
        value = value * valid.detach().float().cpu()
    return _to_uint8_rgb(value.unsqueeze(0).expand(3, -1, -1))


def _fit_panel(panel: Image.Image, width: int, height: int) -> Image.Image:
    return panel.resize((width, height), resample=Image.Resampling.BILINEAR)


def _frame_indices(frame_count: int, maximum: int) -> list[int]:
    count = min(max(int(maximum), 1), frame_count)
    if count == 1:
        return [0]
    return torch.linspace(0, frame_count - 1, count).round().long().tolist()


def _display_scale(predicted: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
    """Least-squares scale used only to expose scale-invariant diagnostics."""

    pred = predicted.detach().float()
    gt = target.detach().float()
    mask = valid.bool() & torch.isfinite(pred).all(-1) & torch.isfinite(gt).all(-1)
    numerator = (pred * gt).sum(-1).masked_fill(~mask, 0.0).sum((1, 2, 3))
    denominator = pred.square().sum(-1).masked_fill(~mask, 0.0).sum((1, 2, 3))
    scale = numerator / denominator.clamp_min(1e-8)
    return torch.where(torch.isfinite(scale) & (scale > 1e-6), scale,
                       torch.ones_like(scale))


def _labeled_grid(rows: list[tuple[str, list[Image.Image]]], frame_ids: list[int],
                  cell_width: int, source_height: int, source_width: int) -> Image.Image:
    cell_height = max(round(source_height / source_width * cell_width), 1)
    label_width = 92
    header_height = 24
    canvas = Image.new(
        "RGB",
        (label_width + len(frame_ids) * cell_width,
         header_height + len(rows) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, frame in enumerate(frame_ids):
        x = label_width + column * cell_width
        draw.text((x + 4, 5), f"frame {frame}", fill="black")
    for row_index, (label, panels) in enumerate(rows):
        y = header_height + row_index * cell_height
        draw.text((5, y + max(cell_height // 2 - 6, 1)), label, fill="black")
        for column, panel in enumerate(panels):
            x = label_width + column * cell_width
            fitted = _fit_panel(panel, cell_width, cell_height)
            canvas.paste(fitted, (x, y))
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1),
                           outline="gray")
    return canvas


@torch.no_grad()
def prepare_abot_diagnostics(prediction: Mapping, sequence: Mapping, criterion):
    """Prepare GT and scale-aligned predictions in the loss coordinate system."""

    target = criterion.prepare_targets(sequence)
    predicted_points = prediction["local_points"].detach().float()
    scale = _display_scale(
        predicted_points, target["local_points"], target["valid_masks"])
    aligned_points = predicted_points * scale[:, None, None, None, None]
    predicted_poses = prediction["camera_poses"].detach().float().clone()
    predicted_poses[..., :3, 3] *= scale[:, None, None]
    point_error = (aligned_points - target["local_points"]).norm(dim=-1)
    relative_error = point_error / target["local_points"].norm(dim=-1).clamp_min(1e-3)
    return {
        "target": target,
        "aligned_points": aligned_points,
        "predicted_poses": predicted_poses,
        "point_error": point_error,
        "relative_error": relative_error,
        "display_scale": scale,
    }


@torch.no_grad()
def render_reconstruction_overview(
    prediction: Mapping,
    sequence: Mapping,
    criterion,
    *,
    batch_index: int = 0,
    num_frames: int = 8,
    cell_width: int = 160,
    relative_error_max: float = 0.2,
    diagnostics: Mapping | None = None,
) -> Image.Image:
    """Render RGB, depth, error, confidence, and validity for one clip."""

    if diagnostics is None:
        diagnostics = prepare_abot_diagnostics(prediction, sequence, criterion)
    target = diagnostics["target"]
    indices = _frame_indices(sequence["imgs"].shape[1], num_frames)
    valid = target["valid_masks"][batch_index]
    pred_depth = diagnostics["aligned_points"][batch_index, ..., 2]
    gt_depth = target["local_points"][batch_index, ..., 2]
    rel_error = diagnostics["relative_error"][batch_index]
    confidence = prediction.get("conf")
    if confidence is not None:
        confidence = confidence.detach().float().sigmoid()[batch_index, ..., 0]

    rgb_panels = [_rgb_panel(sequence["imgs"][batch_index, frame]) for frame in indices]
    depth_ranges = [
        _shared_range(pred_depth[frame], gt_depth[frame], valid[frame])
        for frame in indices
    ]
    pred_depth_panels = [
        _heatmap(pred_depth[frame], valid[frame], *limits)
        for frame, limits in zip(indices, depth_ranges)
    ]
    gt_depth_panels = [
        _heatmap(gt_depth[frame], valid[frame], *limits)
        for frame, limits in zip(indices, depth_ranges)
    ]
    error_panels = [
        _heatmap(rel_error[frame], valid[frame], 0.0, relative_error_max)
        for frame in indices
    ]
    confidence_panels = [
        (_gray_panel(confidence[frame], valid[frame]) if confidence is not None
         else Image.new("RGB", (valid.shape[-1], valid.shape[-2]), "gray"))
        for frame in indices
    ]
    valid_panels = [_gray_panel(valid[frame].float()) for frame in indices]
    rows = [
        ("RGB", rgb_panels),
        ("Pred depth", pred_depth_panels),
        ("GT depth", gt_depth_panels),
        ("Rel error", error_panels),
        ("Confidence", confidence_panels),
        ("Valid mask", valid_panels),
    ]
    height, width = sequence["imgs"].shape[-2:]
    return _labeled_grid(rows, indices, cell_width, height, width)


def _draw_camera_frustum(ax, pose: torch.Tensor, scale: float, color: str,
                         linestyle: str, linewidth: float) -> None:
    """Draw an OpenCV-style camera looking along its positive local Z axis."""

    local = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.70, -0.45, 1.0],
            [0.70, -0.45, 1.0],
            [0.70, 0.45, 1.0],
            [-0.70, 0.45, 1.0],
        ],
        dtype=pose.dtype,
    ) * scale
    world = (pose[:3, :3] @ local.T).T + pose[:3, 3]
    segments = ((0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (2, 3), (3, 4), (4, 1))
    for start, end in segments:
        segment = world[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                color=color, linestyle=linestyle, linewidth=linewidth)


def _set_3d_limits(ax, center: torch.Tensor, radius: float) -> None:
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    ax.set_box_aspect((1.0, 1.0, 1.0))


@torch.no_grad()
def render_trajectory_overview(
    prediction: Mapping,
    sequence: Mapping,
    criterion,
    *,
    batch_index: int = 0,
    canvas_width: int = 960,
    canvas_height: int = 520,
    num_frames: int = 8,
    columns: int = 4,
    camera_scale: float = 0.08,
    diagnostics: Mapping | None = None,
) -> Image.Image:
    """Render streaming 3D trajectory and camera pairs at sampled time steps.

    Every panel shows the complete GT trajectory as a dashed line and the
    predicted trajectory prefix available at that time step as a solid line.
    Dashed and solid camera frustums mark the current GT and predicted poses.
    """

    if diagnostics is None:
        diagnostics = prepare_abot_diagnostics(prediction, sequence, criterion)
    predicted = diagnostics["predicted_poses"][batch_index].detach().float().cpu()
    target = diagnostics["target"]["camera_poses"][batch_index].detach().float().cpu()
    pred_translation = predicted[:, :3, 3]
    gt_translation = target[:, :3, 3]
    translation_error = (pred_translation - gt_translation).norm(dim=-1)
    rotation_error = torch.rad2deg(_rotation_angle(
        predicted[:, :3, :3], target[:, :3, :3]))
    render_every_frame = int(num_frames) <= 0
    indices = (list(range(len(target))) if render_every_frame
               else _frame_indices(len(target), num_frames))
    columns = min(max(int(columns), 1), len(indices))
    rows = math.ceil(len(indices) / columns)
    if render_every_frame:
        canvas_width = max(canvas_width, columns * 220)
        canvas_height = rows * 200 + 50

    all_centers = torch.cat((gt_translation, pred_translation), 0)
    minimum = all_centers.amin(0)
    maximum = all_centers.amax(0)
    center = (minimum + maximum) * 0.5
    span = float((maximum - minimum).amax().clamp_min(1e-3))
    radius = span * 0.62
    frustum_scale = span * float(camera_scale)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    dpi = 100
    figure = Figure(
        figsize=(canvas_width / dpi, canvas_height / dpi), dpi=dpi,
        facecolor="white",
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.82,
                           wspace=0.05, hspace=0.18)
    for panel_index, frame_index in enumerate(indices):
        ax = figure.add_subplot(rows, columns, panel_index + 1, projection="3d")
        ax.plot(gt_translation[:, 0], gt_translation[:, 1], gt_translation[:, 2],
                color="0.35", linestyle="--", linewidth=1.5)
        prefix = pred_translation[:frame_index + 1]
        ax.plot(prefix[:, 0], prefix[:, 1], prefix[:, 2],
                color="#2468d8", linestyle="-", linewidth=2.0)
        _draw_camera_frustum(
            ax, target[frame_index], frustum_scale * 1.12,
            color="0.25", linestyle="--", linewidth=1.1)
        _draw_camera_frustum(
            ax, predicted[frame_index], frustum_scale,
            color="#d9362b", linestyle="-", linewidth=1.4)
        ax.scatter(*gt_translation[frame_index], color="0.25", s=8)
        ax.scatter(*pred_translation[frame_index], color="#d9362b", s=10)
        _set_3d_limits(ax, center, radius)
        ax.view_init(elev=24, azim=-62)
        ax.set_title(
            f"frame {frame_index}   t {translation_error[frame_index]:.3g}   "
            f"R {rotation_error[frame_index]:.2f} deg",
            fontsize=7,
            pad=1,
        )
        ax.tick_params(labelsize=5, pad=-2)
        ax.set_xlabel("X", fontsize=6, labelpad=-5)
        ax.set_ylabel("Y", fontsize=6, labelpad=-5)
        ax.set_zlabel("Z", fontsize=6, labelpad=-5)
        ax.grid(True, linewidth=0.35, alpha=0.45)
    figure.legend(
        handles=[
            Line2D([0], [0], color="0.35", linestyle="--", label="GT full trajectory / camera"),
            Line2D([0], [0], color="#2468d8", linestyle="-", label="predicted prefix"),
            Line2D([0], [0], color="#d9362b", linestyle="-", label="predicted camera"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.frombuffer(
        "RGBA", canvas.get_width_height(), canvas.buffer_rgba(), "raw", "RGBA", 0, 1
    ).convert("RGB")


class ABotReconTensorBoardVisualizer:
    """Schedule ABot-Recon image logging with optimizer-step semantics."""

    def __init__(self, config, gradient_accumulation_steps: int,
                 initial_global_step: int, train_criterion, test_criterion):
        self.config = config
        accumulation = max(int(gradient_accumulation_steps), 1)
        self.step = int(initial_global_step) // accumulation
        self.last_step = -1
        self.validation_step = 0
        self.validation_pending = False
        self.train_criterion = train_criterion
        self.test_criterion = test_criterion

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
            tag_prefix = "train"
            log_step = self.step
            criterion = self.train_criterion
        elif mode == "test":
            if not self.validation_pending:
                return
            tag_prefix = "val"
            log_step = self.validation_step
            criterion = self.test_criterion
        else:
            return

        prediction, sequence = output
        sample_count = min(int(self.config.get("num_samples", 1)),
                           sequence["imgs"].shape[0])
        diagnostics = prepare_abot_diagnostics(prediction, sequence, criterion)
        reconstruction = []
        trajectory = []
        for batch_index in range(sample_count):
            reconstruction.append(pil_to_tensor(render_reconstruction_overview(
                prediction,
                sequence,
                criterion,
                batch_index=batch_index,
                num_frames=int(self.config.get("num_frames", 8)),
                cell_width=int(self.config.get("cell_width", 160)),
                relative_error_max=float(self.config.get("relative_error_max", 0.2)),
                diagnostics=diagnostics,
            )))
            trajectory.append(pil_to_tensor(render_trajectory_overview(
                prediction,
                sequence,
                criterion,
                batch_index=batch_index,
                canvas_width=int(self.config.get("trajectory_width", 960)),
                canvas_height=int(self.config.get("trajectory_height", 520)),
                num_frames=int(self.config.get("trajectory_num_frames", 8)),
                columns=int(self.config.get("trajectory_columns", 4)),
                camera_scale=float(self.config.get("trajectory_camera_scale", 0.08)),
                diagnostics=diagnostics,
            )))
        images = {
            f"{tag_prefix}/abot_reconstruction": torch.stack(reconstruction),
            f"{tag_prefix}/abot_trajectory": torch.stack(trajectory),
        }
        for tracker in accelerator.trackers:
            tracker.log_images(images, step=log_step)
        if mode == "train":
            self.last_step = self.step
        else:
            self.validation_pending = False
