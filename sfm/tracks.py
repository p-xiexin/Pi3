"""Glob3R and VGGSfM tracks frontends with one output protocol."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def sample_map(values, points):
    """Sample an ``H x W x C`` or ``B x C x H x W`` map at ``P x 2`` pixels."""
    if values.ndim == 3:
        values = values.permute(2, 0, 1)[None]
    height, width = values.shape[-2:]
    grid = points.to(values).clone()
    grid[:, 0] = 2 * grid[:, 0] / max(width - 1, 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / max(height - 1, 1) - 1
    grid = grid[None, None].expand(values.shape[0], -1, -1, -1)
    sampled = F.grid_sample(values, grid, mode="bilinear", align_corners=True)
    return sampled[:, :, 0].transpose(1, 2).to(points)


class Glob3RTracks:
    """Sample Glob3R dense correspondence maps at shared image query points."""

    name = "glob3r"

    def __init__(self, model):
        self.model = model
        self.images = None
        self.patch_tokens = None
        self.encoder = None

    def prepare_window(self, images, state):
        """Cache images and encoded features produced by the geometry frontend."""
        if state is None:
            state = self.model.encode_matching_state(images)
        self.images = images
        self.patch_tokens, self.encoder = state

    @torch.no_grad()
    def track(self, reference, query_points):
        """Return ``[frames, points, 2]`` tracks and ``[frames, points]`` confidence."""
        output = self.model.match_pair(self.patch_tokens, self.encoder, self.images, reference)
        if not output.warp_stages or not output.confidence_stages:
            raise RuntimeError("Glob3R refinement did not return warp and confidence stages")
        warps = output.warp_stages[-1]
        confidence = output.confidence_stages[-1]
        height, width = self.images.shape[-2:]
        if warps.shape[-2:] != (height, width):
            warps = F.interpolate(
                warps.squeeze(0),
                (height, width),
                mode="bilinear",
                align_corners=True,
            ).unsqueeze(0)
            confidence = F.interpolate(
                confidence.squeeze(0),
                (height, width),
                mode="bilinear",
                align_corners=True,
            ).unsqueeze(0)
        targets = torch.as_tensor(
            output.target_indices, device=query_points.device, dtype=torch.long
        )
        tracks = torch.zeros(
            self.images.shape[1],
            query_points.shape[0],
            2,
            device=query_points.device,
            dtype=query_points.dtype,
        )
        scores = torch.zeros(
            self.images.shape[1],
            query_points.shape[0],
            device=query_points.device,
            dtype=query_points.dtype,
        )
        tracks[reference], scores[reference] = query_points, 1
        tracks[targets] = sample_map(warps.squeeze(0), query_points)
        scores[targets] = sample_map(confidence.squeeze(0), query_points).squeeze(-1)
        scores[targets] *= scores[targets] >= 0.6
        confidence_map = confidence.squeeze(0)
        if confidence_map.ndim == 4 and confidence_map.shape[1] == 1:
            confidence_map = confidence_map[:, 0]
        if confidence_map.ndim != 3:
            raise RuntimeError(
                "Glob3R confidence heatmap must have shape [targets,H,W]"
            )
        confidence_maps = confidence_map.new_zeros(
            self.images.shape[1], height, width
        )
        confidence_maps[reference] = 1
        confidence_maps[targets] = confidence_map
        return {
            "tracks": tracks,
            "confidence": scores,
            "visualization_confidence": confidence_maps,
            "visualization_confidence_label": "glob3r confidence",
        }


class VGGSfMTracks:
    """Adapt the official VGGSfM tracker to the shared tracks output protocol."""

    name = "vgg"

    def __init__(
        self,
        tracker,
        image_size,
        tracker_size=1024,
        visibility_threshold=0.05,
        score_threshold=0.5,
        fine_tracking=True,
        mixed_precision="fp16",
    ):
        self.tracker = tracker.eval()
        self.image_size = tuple(map(int, image_size))
        self.tracker_size = int(tracker_size)
        self.visibility_threshold = float(visibility_threshold)
        self.score_threshold = float(score_threshold)
        self.fine_tracking = bool(fine_tracking)
        self.autocast_dtype = {
            "none": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]
        self.images = None
        self.fmaps = None

    def _autocast(self, device):
        return torch.amp.autocast(
            device_type=device.type, dtype=self.autocast_dtype,
            enabled=device.type == "cuda" and self.autocast_dtype is not None,
        )

    @torch.no_grad()
    def prepare_window(self, images, _state):
        """Apply VGGSfM's square padding and resize before feature extraction."""
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"VGGSfM expected image_size={self.image_size}, "
                f"received={tuple(images.shape[-2:])}"
            )
        batch, frames, channels, height, width = images.shape
        square_size = max(height, width)
        vertical = square_size - height
        horizontal = square_size - width
        # Official DemoLoader centers the image in a square crop. For an odd
        # difference its negative crop origin puts the extra pixel first.
        pad_top = (vertical + 1) // 2
        pad_bottom = vertical - pad_top
        pad_left = (horizontal + 1) // 2
        pad_right = horizontal - pad_left
        padded = F.pad(
            images, (pad_left, pad_right, pad_top, pad_bottom), value=0.0
        )
        self.images = F.interpolate(
            padded.reshape(batch * frames, channels, square_size, square_size),
            size=(self.tracker_size, self.tracker_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(
            batch,
            frames,
            channels,
            self.tracker_size,
            self.tracker_size,
        )
        self.coordinate_scale = self.tracker_size / square_size
        self.coordinate_shift = images.new_tensor(
            (pad_left, pad_top)
        ) * self.coordinate_scale
        with self._autocast(images.device):
            self.fmaps = self.tracker.process_images_to_fmaps(self.images)

    @torch.no_grad()
    def track(self, reference, query_points):
        """Track shared image queries in the configured image coordinates."""
        frame_count = self.images.shape[1]
        # TrackerPredictor treats frame zero as the query frame, so preserve an
        # explicit permutation and undo it before returning graph observations.
        order = torch.cat((
            torch.tensor([reference], device=query_points.device),
            torch.arange(frame_count, device=query_points.device)[
                torch.arange(frame_count, device=query_points.device) != reference
            ],
        ))
        with self._autocast(query_points.device):
            tracker_queries = (
                query_points * self.coordinate_scale + self.coordinate_shift
            )
            fine, coarse, visibility, score = self.tracker(
                self.images[:, order], tracker_queries[None],
                fmaps=self.fmaps[:, order], fine_tracking=self.fine_tracking,
            )
        predicted = fine if self.fine_tracking else coarse
        tracks = torch.empty_like(predicted)
        tracks[:, order] = predicted
        visible = torch.empty_like(visibility)
        visible[:, order] = visibility
        if score is None:
            raise RuntimeError("VGGSfM tracker did not return track scores")
        reordered_score = torch.empty_like(score)
        reordered_score[:, order] = score
        valid = (
            (visible > self.visibility_threshold)
            & (reordered_score > self.score_threshold)
        )
        confidence = torch.where(
            valid, visible * reordered_score, torch.zeros_like(reordered_score)
        )
        tracks[:, reference] = tracker_queries[None]
        tracks = (tracks - self.coordinate_shift) / self.coordinate_scale
        visible[:, reference] = 1
        reordered_score[:, reference] = 1
        confidence[:, reference] = 1
        return {
            "tracks": tracks[0].float(),
            "confidence": confidence[0].float(),
            "visualization_confidence": visible[0].float(),
            "visualization_score": reordered_score[0].float(),
            "visualization_confidence_label": "vgg visible=color score=alpha",
        }


def _configure_vggsfm_source(root):
    """Expose a manually prepared VGGSfM checkout on the Python import path."""
    required = (
        root / "vggsfm" / "models" / "track_predictor.py",
        root / "cfgs" / "demo.yaml",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"VGGSfM source files are missing: {missing}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

# TODO: vggsfm_v2_0_0.bin->vggsfm_v2_tracker.pt
def load_vggsfm_tracker(root, checkpoint, device):
    """Build TrackerPredictor from a configured source tree and local checkpoint."""
    root = Path(root).resolve()
    checkpoint = Path(checkpoint).resolve()
    _configure_vggsfm_source(root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGSfM checkpoint does not exist: {checkpoint}")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    with initialize_config_dir(
        version_base=None, config_dir=str((root / "cfgs").resolve())
    ):
        config = compose(config_name="demo")
    OmegaConf.set_struct(config, False)
    config.MODEL.TRACK._target_ = "vggsfm.models.track_predictor.TrackerPredictor"
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
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise TypeError("VGGSfM checkpoint must be the official raw model state dict")
    prefix = "track_predictor."
    # Official checkpoints contain several networks. The tracker receives only
    # its own subtree and is loaded strictly so architecture drift fails early.
    tracker_state = {
        key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not tracker_state:
        raise RuntimeError("checkpoint contains no track_predictor weights")
    tracker.load_state_dict(tracker_state, strict=True)
    return tracker.to(device).eval()


__all__ = ["Glob3RTracks", "VGGSfMTracks", "load_vggsfm_tracker", "sample_map"]
