# Tracker-only setup from the Pi3 repository root.
# git clone --depth 1 https://github.com/facebookresearch/vggsfm.git ../vggsfm
# python -m pip install hydra-core omegaconf einops kornia pillow
# mkdir -p ../vggsfm/ckpt
# wget -O ../vggsfm/ckpt/vggsfm_v2_0_0.bin \
#   https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_0_0.bin
# VGGSfM compatibility patches:
# - Modified files:
#   * vggsfm/models/track_modules/refine_track.py:
#       kornia.utils.grid.create_meshgrid -> kornia.utils.create_meshgrid
#   * vggsfm/two_view_geo/utils.py:
#       kornia.core.Tensor -> torch.Tensor
#       kornia.utils._compat.torch_version_ge -> local implementation
# - Keep current Kornia version unchanged to avoid affecting other projects.

"""Validate the local SfM backend with VGGSfM tracks and Pi3 geometry."""

from __future__ import annotations

import sys
import types
import colorsys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from .frame import Frames
from .matching import Tracks
from .sfm import Glob3RSfMConfig, Glob3RSfMPipeline


# Temporary experiment settings. Edit these values directly before running.
IMAGE_DIR = "path/to/images"
BACKBONE_CHECKPOINT = "path/to/pi3/model.safetensors"
VGGSFM_ROOT = "../vggsfm"
VGGSFM_CHECKPOINT = "../vggsfm/ckpt/vggsfm_v2_0_0.bin"
CALIBRATION = "path/to/calibration.yaml"
OUTPUT = "outputs/vggsfm_tracks_sfm/result.pt"
MATCH_VIS_DIR = "outputs/vggsfm_tracks_sfm/matching"
HEIGHT = 336
WIDTH = 448
DEVICE = "cuda:0"
RANDOM_SEED = 0
KEYFRAME_THRESHOLD = 0.5
POINTS_PER_KEYFRAME = 512
DEPTH_CONFIDENCE_THRESHOLD = 0.1
TRACK_CONFIDENCE_THRESHOLD = 0.2
TRACKER_SIZE = 1024
FINE_TRACKING = True
MIXED_PRECISION = "fp16"  # "none", "fp16", or "bf16"
MATCH_VIS_MAX_TRACKS = 128
MATCH_VIS_CELL_WIDTH = 448


@dataclass(frozen=True)
class VGGSfMTrackOutput:
    reference_index: int
    query_flat_indices: torch.Tensor
    query_points: torch.Tensor
    tracks: torch.Tensor
    visibility: torch.Tensor
    score: torch.Tensor
    confidence_threshold: float


class VGGSfMTrackFrontend(nn.Module):
    """Combine Pi3 Eq. (1) geometry with only the VGGSfM track predictor."""

    def __init__(
        self,
        geometry_model: nn.Module,
        track_predictor: nn.Module,
        points_per_keyframe: int,
        depth_confidence_threshold: float,
        track_confidence_threshold: float,
        tracker_size: int,
        fine_tracking: bool,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.geometry_model = geometry_model
        self.track_predictor = track_predictor
        self.points_per_keyframe = points_per_keyframe
        self.depth_confidence_threshold = depth_confidence_threshold
        self.track_confidence_threshold = track_confidence_threshold
        self.tracker_size = tracker_size
        self.fine_tracking = fine_tracking
        self.autocast_dtype = autocast_dtype
        self._geometry = None
        self._fmaps = None
        self._tracker_images = None

    def _autocast(self, device: torch.device):
        enabled = device.type == "cuda" and self.autocast_dtype is not None
        return torch.amp.autocast(
            device_type=device.type,
            dtype=self.autocast_dtype,
            enabled=enabled,
        )

    @torch.no_grad()
    def infer_window(self, images: torch.Tensor):
        geometry, _, _ = self.geometry_model.infer_window(images)
        batch, frames, channels, _, _ = images.shape
        tracker_images = torch.nn.functional.interpolate(
            images.reshape(batch * frames, channels, *images.shape[-2:]),
            size=(self.tracker_size, self.tracker_size),
            mode="bilinear",
            align_corners=True,
        ).reshape(batch, frames, channels, self.tracker_size, self.tracker_size)
        with self._autocast(images.device):
            self._fmaps = self.track_predictor.process_images_to_fmaps(
                tracker_images
            )
        self._geometry = geometry
        self._tracker_images = tracker_images
        return geometry, None, None

    @torch.no_grad()
    def match_pair(
        self,
        patch_tokens,
        encoder_features,
        images: torch.Tensor,
        reference_index: int,
    ) -> VGGSfMTrackOutput:
        del patch_tokens, encoder_features
        if (
            self._geometry is None
            or self._fmaps is None
            or self._tracker_images is None
        ):
            raise RuntimeError("infer_window must run before VGGSfM tracking")

        _, frame_count, _, height, width = images.shape
        points = self._geometry["local_points"][0, reference_index]
        confidence = self._geometry["conf"][0, reference_index].sigmoid().squeeze(-1)
        reliable = (
            (confidence > self.depth_confidence_threshold)
            & torch.isfinite(points).all(dim=-1)
            & (points[..., 2] > 0)
        )
        flat_indices = torch.nonzero(
            reliable.reshape(-1), as_tuple=False
        ).squeeze(dim=-1)
        count = min(self.points_per_keyframe, int(flat_indices.numel()))
        if count == 0:
            raise RuntimeError(
                f"reference frame {reference_index} has no reliable Pi3 anchors"
            )
        selected = flat_indices[
            torch.randperm(flat_indices.numel(), device=images.device)[:count]
        ]
        query_points = torch.stack(
            (
                selected.remainder(width),
                torch.div(selected, width, rounding_mode="floor"),
            ),
            dim=-1,
        ).to(dtype=images.dtype)
        tracker_scale = query_points.new_tensor(
            (
                (self.tracker_size - 1) / max(width - 1, 1),
                (self.tracker_size - 1) / max(height - 1, 1),
            )
        )
        tracker_query_points = query_points * tracker_scale

        order = torch.cat(
            (
                torch.tensor([reference_index], device=images.device),
                torch.arange(frame_count, device=images.device)[
                    torch.arange(frame_count, device=images.device)
                    != reference_index
                ],
            )
        )
        images_feed = self._tracker_images[:, order]
        fmaps_feed = self._fmaps[:, order]
        with self._autocast(images.device):
            fine_tracks, coarse_tracks, visibility, score = self.track_predictor(
                images_feed,
                tracker_query_points.unsqueeze(dim=0),
                fmaps=fmaps_feed,
                fine_tracking=self.fine_tracking,
            )
        predicted = fine_tracks if self.fine_tracking else coarse_tracks
        tracks = torch.empty_like(predicted)
        tracks[:, order] = predicted
        visibility_original = torch.empty_like(visibility)
        visibility_original[:, order] = visibility
        score_original = torch.empty_like(score)
        score_original[:, order] = score
        tracks[:, reference_index] = tracker_query_points
        visibility_original[:, reference_index] = 1
        score_original[:, reference_index] = 1
        tracks /= tracker_scale
        return VGGSfMTrackOutput(
            reference_index=reference_index,
            query_flat_indices=selected,
            query_points=query_points,
            tracks=tracks.squeeze(dim=0).float(),
            visibility=visibility_original.squeeze(dim=0).float(),
            score=score_original.squeeze(dim=0).float(),
            confidence_threshold=self.track_confidence_threshold,
        )


@torch.no_grad()
def tracks_from_vggsfm(
    output: VGGSfMTrackOutput,
    frames: Frames,
    reference_index: int,
    points_per_keyframe: int = 512,
    depth_confidence_threshold: float = 0.1,
    warp_confidence_threshold: float = 0.6,
) -> Tracks:
    """Adapt one sparse VGGSfM prediction to the existing backend Tracks."""

    del points_per_keyframe, depth_confidence_threshold, warp_confidence_threshold
    if reference_index != output.reference_index:
        raise ValueError("VGGSfM output and requested reference frame differ")
    frame_count, _, height, width = frames.Is.shape
    if output.tracks.shape[:2] != (frame_count, output.query_points.shape[0]):
        raise ValueError("VGGSfM returned an incompatible track tensor")

    reference_confidence = frames.Cs[reference_index].reshape(-1)[
        output.query_flat_indices
    ]
    anchor_points = frames.Xs_C[reference_index].reshape(-1, 3)[
        output.query_flat_indices
    ]
    combined_confidence = output.visibility * output.score
    tracks = output.tracks.to(device=frames.Is.device, dtype=frames.Is.dtype)
    valid = (
        torch.isfinite(tracks).all(dim=-1)
        & (tracks[..., 0] >= 0)
        & (tracks[..., 0] <= width - 1)
        & (tracks[..., 1] >= 0)
        & (tracks[..., 1] <= height - 1)
        & (combined_confidence >= output.confidence_threshold)
    )
    valid[reference_index] = True
    weights = reference_confidence.unsqueeze(dim=0) * combined_confidence
    weights[reference_index] = reference_confidence
    weights *= valid
    keep = valid.sum(dim=0) >= 3
    return Tracks(
        rs=torch.full(
            (int(keep.sum()),),
            reference_index,
            device=frames.Is.device,
            dtype=torch.long,
        ),
        ks=output.query_flat_indices[keep],
        Xs_Cr=anchor_points[keep],
        us=tracks[:, keep],
        mask=valid[:, keep],
        ws=weights[:, keep],
    )


@torch.no_grad()
def save_vggsfm_matching_matrix(
    output_dir: str | Path,
    frames: Frames,
    reference_index: int,
    output: VGGSfMTrackOutput,
    tracks: Tracks,
    cell_width: int | None = None,
) -> Path:
    """Draw sparse VGGSfM correspondences as reference-target row pairs."""

    del output
    from PIL import Image, ImageDraw

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count, _, height, width = frames.Is.shape
    cell_width = MATCH_VIS_CELL_WIDTH if cell_width is None else cell_width
    cell_height = max(round(height / width * cell_width), 1)

    def image_panel(image: torch.Tensor) -> Image.Image:
        array = (
            image.detach()
            .float()
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
            * 255
        ).round().astype("uint8")
        return Image.fromarray(array, mode="RGB").resize(
            (cell_width, cell_height),
            resample=Image.Resampling.BICUBIC,
        )

    def track_color(track_id: int) -> tuple[int, int, int]:
        hue = (track_id * 0.618033988749895) % 1.0
        return tuple(
            round(channel * 255)
            for channel in colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        )

    canvas = Image.new(
        "RGB",
        (2 * cell_width, frame_count * cell_height),
        "black",
    )
    x_scale = (cell_width - 1) / max(width - 1, 1)
    y_scale = (cell_height - 1) / max(height - 1, 1)
    reference_us = tracks.us[reference_index]

    for target_index in range(frame_count):
        row = Image.new("RGB", (2 * cell_width, cell_height), "black")
        row.paste(image_panel(frames.Is[reference_index]), (0, 0))
        row.paste(image_panel(frames.Is[target_index]), (cell_width, 0))
        draw = ImageDraw.Draw(row, "RGBA")
        valid_indices = torch.nonzero(
            tracks.mask[reference_index] & tracks.mask[target_index],
            as_tuple=False,
        ).squeeze(dim=-1)
        valid_count = int(valid_indices.numel())
        if valid_indices.numel() > MATCH_VIS_MAX_TRACKS:
            weights = tracks.ws[target_index, valid_indices]
            valid_indices = valid_indices[
                weights.topk(MATCH_VIS_MAX_TRACKS).indices
            ]

        for track_index in valid_indices.tolist():
            reference_xy = reference_us[track_index].detach().float().cpu()
            target_xy = tracks.us[target_index, track_index].detach().float().cpu()
            x0 = float(reference_xy[0]) * x_scale
            y0 = float(reference_xy[1]) * y_scale
            x1 = cell_width + float(target_xy[0]) * x_scale
            y1 = float(target_xy[1]) * y_scale
            color = track_color(int(tracks.ks[track_index]))
            radius = 2.5
            draw.ellipse(
                (x0 - radius, y0 - radius, x0 + radius, y0 + radius),
                fill=(*color, 255),
            )
            draw.ellipse(
                (x1 - radius, y1 - radius, x1 + radius, y1 + radius),
                fill=(*color, 255),
            )

        label = (
            f"reference {reference_index:04d}  target {target_index:04d}  "
            f"shown {int(valid_indices.numel())} / valid {valid_count}"
        )
        draw.rectangle((4, 4, 375, 23), fill=(0, 0, 0, 210))
        draw.text((8, 7), label, fill=(255, 255, 255, 255))
        canvas.paste(row, (0, target_index * cell_height))

    path = output_dir / f"reference_{reference_index:04d}.png"
    canvas.save(path)
    return path


def load_pi3_geometry(
    checkpoint: str | Path,
    device: str | torch.device,
) -> nn.Module:
    """Load Pi3 geometry without loading a Glob3R matching checkpoint."""

    from pi3.models.pi3 import Pi3

    from .glob3r_sfm import Glob3RSfM
    from .inference import _read_state_dict, _strip_prefixes

    backbone = Pi3(pos_type="rope100", decoder_size="large")
    state = _strip_prefixes(
        _read_state_dict(checkpoint), ("module.", "model.", "backbone.")
    )
    result = backbone.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"incomplete Pi3 checkpoint: {result}")
    model = Glob3RSfM(
        backbone,
        encoder_layers=(5, 11, 17, 23),
        enable_refinement=False,
        matching_checkpoint=None,
    )
    model.glob3r_matching_head = nn.Identity()
    return model.to(device).eval()


def _install_tracker_only_package(vggsfm_root: Path) -> None:
    """Expose VGGSfM tracker modules without importing its backend package."""

    package_root = vggsfm_root / "vggsfm"
    models_root = package_root / "models"
    if not (models_root / "track_predictor.py").is_file():
        raise FileNotFoundError(
            f"{vggsfm_root} is not an official VGGSfM source checkout"
        )
    root_string = str(vggsfm_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    if "vggsfm" in sys.modules or "vggsfm.models" in sys.modules:
        raise RuntimeError(
            "VGGSfM was imported before tracker-only initialization"
        )
    root_package = types.ModuleType("vggsfm")
    root_package.__path__ = [str(package_root)]
    root_package.__package__ = "vggsfm"
    models_package = types.ModuleType("vggsfm.models")
    models_package.__path__ = [str(models_root)]
    models_package.__package__ = "vggsfm.models"
    sys.modules["vggsfm"] = root_package
    sys.modules["vggsfm.models"] = models_package


def _unwrap_checkpoint(state) -> Mapping[str, torch.Tensor]:
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, Mapping) and isinstance(state.get(key), Mapping):
            state = state[key]
    if not isinstance(state, Mapping):
        raise TypeError("VGGSfM checkpoint does not contain a state dict")
    return state


def load_vggsfm_tracker(
    vggsfm_root: str | Path,
    checkpoint: str | Path,
    device: str | torch.device,
) -> nn.Module:
    """Instantiate and load only the official VGGSfM TrackerPredictor."""

    vggsfm_root = Path(vggsfm_root).resolve()
    _install_tracker_only_package(vggsfm_root)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base=None,
        config_dir=str((vggsfm_root / "cfgs").resolve()),
    ):
        config = compose(config_name="demo")
    OmegaConf.set_struct(config, False)
    config.MODEL.TRACK._target_ = (
        "vggsfm.models.track_predictor.TrackerPredictor"
    )
    config.MODEL.TRACK.COARSE.FEATURENET._target_ = (
        "vggsfm.models.track_modules.blocks.BasicEncoder"
    )
    config.MODEL.TRACK.COARSE.PREDICTOR._target_ = (
        "vggsfm.models.track_modules.base_track_predictor.BaseTrackerPredictor"
    )
    config.MODEL.TRACK.FINE.FEATURENET._target_ = (
        "vggsfm.models.track_modules.blocks.ShallowEncoder"
    )
    config.MODEL.TRACK.FINE.PREDICTOR._target_ = (
        "vggsfm.models.track_modules.base_track_predictor.BaseTrackerPredictor"
    )
    tracker = instantiate(config.MODEL.TRACK, _recursive_=False, cfg=config)

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _unwrap_checkpoint(state)
    prefixes = (
        "module.track_predictor.",
        "model.track_predictor.",
        "track_predictor.",
    )
    tracker_state = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                tracker_state[key[len(prefix):]] = value
                break
    if not tracker_state:
        raise RuntimeError("checkpoint contains no track_predictor weights")
    result = tracker.load_state_dict(tracker_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"incomplete VGGSfM tracker checkpoint: {result}")
    return tracker.to(device).eval()


def main() -> None:
    from .inference import load_calibration, load_image_sequence
    from .run_glob3r_sfm import save_ply

    torch.manual_seed(RANDOM_SEED)
    output = Path(OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    if HEIGHT % 14 or WIDTH % 14:
        raise ValueError("height and width must be divisible by Pi3 patch size 14")

    images, paths = load_image_sequence(IMAGE_DIR, (HEIGHT, WIDTH))
    print("Input frame order:")
    for index, path in enumerate(paths):
        print(f"  [{index:04d}] {path.name}")
    K, calibration_width, calibration_height = load_calibration(CALIBRATION)
    K[0] *= WIDTH / calibration_width
    K[1] *= HEIGHT / calibration_height

    geometry_model = load_pi3_geometry(BACKBONE_CHECKPOINT, DEVICE)
    tracker = load_vggsfm_tracker(
        VGGSFM_ROOT,
        VGGSFM_CHECKPOINT,
        DEVICE,
    )
    autocast_dtype = {
        "none": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[MIXED_PRECISION]
    frontend = VGGSfMTrackFrontend(
        geometry_model,
        tracker,
        points_per_keyframe=POINTS_PER_KEYFRAME,
        depth_confidence_threshold=DEPTH_CONFIDENCE_THRESHOLD,
        track_confidence_threshold=TRACK_CONFIDENCE_THRESHOLD,
        tracker_size=TRACKER_SIZE,
        fine_tracking=FINE_TRACKING,
        autocast_dtype=autocast_dtype,
    ).eval()
    config = Glob3RSfMConfig(
        tracking_points_per_keyframe=POINTS_PER_KEYFRAME,
        keyframe_projection_threshold=KEYFRAME_THRESHOLD,
        depth_confidence_threshold=DEPTH_CONFIDENCE_THRESHOLD,
        warp_confidence_threshold=TRACK_CONFIDENCE_THRESHOLD,
    )
    pipeline = Glob3RSfMPipeline(frontend, config)
    with (
        patch("local_opt.sfm.match_tracks", new=tracks_from_vggsfm),
        patch(
            "local_opt.sfm.save_matching_matrix",
            new=save_vggsfm_matching_matrix,
        ),
    ):
        result = pipeline.run(
            images.to(DEVICE),
            K.to(DEVICE),
            visualization_dir=MATCH_VIS_DIR,
        )

    torch.save(
        {
            "frontend": "vggsfm_tracker_only",
            "image_paths": [str(path) for path in paths],
            "world_to_camera": result.T_CWs.cpu(),
            "camera_to_world": result.T_WCs.cpu(),
            "intrinsics": result.Ks.cpu(),
            "keyframes": result.keyframes.cpu(),
            "track_references": result.tracks.rs.cpu(),
            "track_ids": result.tracks.ks.cpu(),
            "track_anchor_points": result.tracks.Xs_Cr.cpu(),
            "track_observations": result.tracks.us.cpu(),
            "track_mask": result.tracks.mask.cpu(),
            "track_weights": result.tracks.ws.cpu(),
            "sparse_points_before_optimization": result.Xs_W0.cpu(),
            "sparse_points": result.Xs_W.cpu(),
            "track_inliers": result.track_inliers.cpu(),
            "eq5_loss": result.eq5_loss.cpu(),
            "eq6_loss": result.eq6_loss.cpu(),
        },
        output,
    )
    observation_count = result.tracks.mask.sum(dim=0)
    observation_count[~result.track_inliers] = -1
    save_ply(
        output.parent / "sparse_tracks.ply",
        result.Xs_W,
        scalar_fields={
            "reference_id": result.tracks.rs,
            "track_id": result.tracks.ks,
            "observation_count": observation_count,
        },
    )
    save_ply(
        output.parent / "pi3_sfm.ply",
        result.dense_Ps_W,
        result.dense_RGBs,
        {"frame_id": result.dense_frame_ids},
    )
    print(f"Selected keyframes: {result.keyframes.tolist()}")
    print(f"Saved VGGSfM-track validation to {output}")


if __name__ == "__main__":
    main()
