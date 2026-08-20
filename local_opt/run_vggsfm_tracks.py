# Tracker-only VGGSfM setup from the Pi3 repository root.
# git clone --depth 1 https://github.com/facebookresearch/vggsfm.git ../vggsfm
# python -m pip install hydra-core omegaconf einops kornia pillow opencv-python
# mkdir -p ../vggsfm/ckpt
# wget -O ../vggsfm/ckpt/vggsfm_v2_0_0.bin \
#   https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_0_0.bin
# VGGSfM compatibility patches for recent Kornia releases.
# vggsfm/models/track_modules/refine_track.py
#   kornia.utils.grid.create_meshgrid -> kornia.utils.create_meshgrid
# vggsfm/two_view_geo/utils.py
#   kornia.core.Tensor -> torch.Tensor
#   kornia.utils._compat.torch_version_ge -> a local version check

"""Validate the local SfM backend from VGGSfM tracks without Pi3 poses."""

from __future__ import annotations

import colorsys
import math
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from pi3.utils.geometry import depth_edge

from .backend import bundle_adjust, opt_pose_ray
from .frame import Frames
from .matching import Tracks
from .pose_graph import (
    PoseGraph,
    maximum_spanning_tree_initialization,
    robust_rotation_averaging,
)
from .sfm import select_keyframes_eq4
from .timing import tic, toc


# Temporary experiment settings. Edit these values directly before running.
IMAGE_DIR = "data/chunk"
BACKBONE_CHECKPOINT = "ckpts/Pi3/model.safetensors"
VGGSFM_ROOT = "../vggsfm-main"
VGGSFM_CHECKPOINT = "../vggsfm-main/ckpt/vggsfm_v2_0_0.bin"
CALIBRATION = None
MASK = None
OUTPUT = "outputs/vggsfm_tracks_sfm/result.pt"
MATCH_VIS_DIR = "outputs/vggsfm_tracks_sfm/matching"
HEIGHT = 336
WIDTH = 448
DEVICE = "cuda:0"
RANDOM_SEED = 0

# Pi3 is used once for Eq. (4) keyframe selection and post-BA dense depth.
KEYFRAME_PROJECTION_THRESHOLD = 0.5
POINTS_PER_QUERY_FRAME = 2048
QUERY_METHODS = ["sp", "sift"]
QUERY_DETECTION_THRESHOLD = 0.005
QUERY_BUCKET_GRID = (8, 6)  # columns, rows
QUERY_CANDIDATE_MULTIPLIER = 4
TRACK_CONFIDENCE_THRESHOLD = 0.2
TRACKER_SIZE = 1024
FINE_TRACKING = True
MIXED_PRECISION = "fp16"  # "none", "fp16", or "bf16"
MIN_TRACK_OBSERVATIONS = 3
MIN_PAIR_MATCHES = 16
MIN_PAIR_INLIERS = 12
ESSENTIAL_RANSAC_THRESHOLD_PX = 1.0
MIN_TRIANGULATION_ANGLE_DEG = 1.5
EQ5_ITERATIONS = 15
EQ6_ITERATIONS = 20
SECOND_BA_ITERATIONS = 10
SCALE_PRIOR_WEIGHT = 1.0e3
MAX_REPROJECTION_ERROR_PX = 4.0

# Dense reconstruction. "none" matches VGGSfM without DepthAnything.
# "pi3" replaces DepthAnything with a Pi3 local point map after sparse BA.
DENSE_RECONSTRUCTION = "pi3"  # "none" or "pi3"
DENSE_ALIGNMENT = "disparity_affine"  # "scale" or "disparity_affine"
PI3_MASK_CONFIDENCE_THRESHOLD = 0.1
DEPTH_CONFIDENCE_THRESHOLD = 0.1
SCALE_RANSAC_THRESHOLD = 0.1
DISPARITY_RANSAC_RATIO = 30.0
DENSE_RANSAC_TRIALS = 2048
MIN_DENSE_ALIGNMENT_POINTS = 8

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


@dataclass(frozen=True)
class SparseSfMResult:
    T_CWs: torch.Tensor
    Ks: torch.Tensor
    keyframes: torch.Tensor
    tracks: Tracks
    pose_graph: PoseGraph
    T_CWs_mst: torch.Tensor
    Rs_avg: torch.Tensor
    Xs_W0: torch.Tensor
    Xs_W: torch.Tensor
    eq5_loss: torch.Tensor
    eq6_loss: torch.Tensor

    @property
    def T_WCs(self) -> torch.Tensor:
        return torch.linalg.inv(self.T_CWs)


class VGGSfMTrackFrontend(nn.Module):
    """Run only the official VGGSfM tracker on image-derived query points."""

    def __init__(
        self,
        track_predictor: nn.Module,
        points_per_query_frame: int,
        track_confidence_threshold: float,
        tracker_size: int,
        fine_tracking: bool,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.track_predictor = track_predictor
        self.points_per_query_frame = points_per_query_frame
        self.track_confidence_threshold = track_confidence_threshold
        self.tracker_size = tracker_size
        self.fine_tracking = fine_tracking
        self.autocast_dtype = autocast_dtype
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
    def prepare_window(self, images: torch.Tensor) -> None:
        batch, frames, channels, _, _ = images.shape
        tracker_images = F.interpolate(
            images.reshape(batch * frames, channels, *images.shape[-2:]),
            size=(self.tracker_size, self.tracker_size),
            mode="bilinear",
            align_corners=True,
        ).reshape(batch, frames, channels, self.tracker_size, self.tracker_size)
        with self._autocast(images.device):
            self._fmaps = self.track_predictor.process_images_to_fmaps(
                tracker_images
            )
        self._tracker_images = tracker_images

    @torch.no_grad()
    def match_reference(
        self,
        images: torch.Tensor,
        reference_index: int,
        valid_mask: torch.Tensor | None = None,
    ) -> VGGSfMTrackOutput:
        if self._fmaps is None or self._tracker_images is None:
            raise RuntimeError("prepare_window must run before VGGSfM tracking")

        _, frame_count, _, height, width = images.shape
        query_points, flat_indices = _select_image_query_points(
            images[0, reference_index],
            self.points_per_query_frame,
            None if valid_mask is None else valid_mask[reference_index],
        )
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
        with self._autocast(images.device):
            fine_tracks, coarse_tracks, visibility, score = self.track_predictor(
                self._tracker_images[:, order],
                tracker_query_points.unsqueeze(dim=0),
                fmaps=self._fmaps[:, order],
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
            query_flat_indices=flat_indices,
            query_points=query_points,
            tracks=tracks.squeeze(dim=0).float(),
            visibility=visibility_original.squeeze(dim=0).float(),
            score=score_original.squeeze(dim=0).float(),
            confidence_threshold=self.track_confidence_threshold,
        )


def _select_image_query_points(
    image: torch.Tensor,
    max_points: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and spatially bucket LightGlue query points."""

    from lightglue import ALIKED, SIFT, SuperPoint

    if not QUERY_METHODS:
        raise ValueError("QUERY_METHODS must contain at least one extractor")
    invalid_mask = None if valid_mask is None else (~valid_mask)[None]
    candidate_limit = max_points * QUERY_CANDIDATE_MULTIPLIER
    point_parts = []
    score_parts = []
    for method in QUERY_METHODS:
        if method == "sp":
            extractor = SuperPoint(
                max_num_keypoints=candidate_limit,
                detection_threshold=QUERY_DETECTION_THRESHOLD,
            )
        elif method == "sift":
            extractor = SIFT(max_num_keypoints=candidate_limit)
        elif method == "aliked":
            extractor = ALIKED(
                max_num_keypoints=candidate_limit,
                detection_threshold=QUERY_DETECTION_THRESHOLD,
            )
        else:
            raise ValueError(f"unsupported query method {method}")
        features = extractor.to(image.device).eval().extract(
            image[None], invalid_mask=invalid_mask
        )
        point_parts.append(features["keypoints"][0])
        score_parts.append(features["keypoint_scores"][0])
        del features

    query_points = torch.cat(point_parts)
    scores = torch.cat(score_parts)
    finite = torch.isfinite(query_points).all(dim=-1) & torch.isfinite(scores)
    query_points, scores = query_points[finite], scores[finite]
    pixels = query_points.round().long()
    pixels[:, 0].clamp_(0, image.shape[-1] - 1)
    pixels[:, 1].clamp_(0, image.shape[-2] - 1)
    if valid_mask is not None:
        keep = valid_mask[pixels[:, 1], pixels[:, 0]]
        query_points, pixels, scores = (
            query_points[keep],
            pixels[keep],
            scores[keep],
        )
    if query_points.shape[0] == 0:
        raise RuntimeError("the reference image has no valid query points")

    columns, rows = QUERY_BUCKET_GRID
    bucket_ids = (
        pixels[:, 1] * rows // image.shape[-2] * columns
        + pixels[:, 0] * columns // image.shape[-1]
    )
    points_per_bucket, remainder = divmod(max_points, columns * rows)
    selected = []
    for bucket_id in range(columns * rows):
        indices = torch.where(bucket_ids == bucket_id)[0]
        limit = points_per_bucket + int(bucket_id < remainder)
        if indices.numel() > limit:
            indices = indices[scores[indices].topk(limit).indices]
        if limit:
            selected.append(indices)
    selected = torch.cat(selected)
    query_points, pixels = query_points[selected], pixels[selected]
    flat_indices = pixels[:, 1] * image.shape[-1] + pixels[:, 0]
    return query_points.to(dtype=image.dtype), flat_indices


def _tracks_from_vggsfm(
    output: VGGSfMTrackOutput,
    frame_count: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    valid_mask: torch.Tensor | None = None,
) -> Tracks:
    combined_confidence = output.visibility * output.score
    tracks = output.tracks.to(device=device, dtype=dtype)
    valid = (
        torch.isfinite(tracks).all(dim=-1)
        & (tracks[..., 0] >= 0)
        & (tracks[..., 0] <= width - 1)
        & (tracks[..., 1] >= 0)
        & (tracks[..., 1] <= height - 1)
        & (combined_confidence >= output.confidence_threshold)
    )
    valid[output.reference_index] = True
    if valid_mask is not None:
        pixels = tracks.nan_to_num().round().long()
        pixels[..., 0].clamp_(0, width - 1)
        pixels[..., 1].clamp_(0, height - 1)
        frame_ids = torch.arange(frame_count, device=device)[:, None]
        valid &= valid_mask[frame_ids, pixels[..., 1], pixels[..., 0]]
    weights = combined_confidence.to(device=device, dtype=dtype)
    weights[output.reference_index] = 1
    weights *= valid
    keep = (valid.sum(dim=0) >= MIN_TRACK_OBSERVATIONS) & valid[
        output.reference_index
    ]
    point_count = int(keep.sum())
    return Tracks(
        rs=torch.full(
            (point_count,),
            output.reference_index,
            device=device,
            dtype=torch.long,
        ),
        ks=output.query_flat_indices.to(device=device)[keep],
        Xs_Cr=torch.zeros(point_count, 3, device=device, dtype=dtype),
        us=tracks[:, keep],
        mask=valid[:, keep],
        ws=weights[:, keep],
    )


def _concatenate_tracks(parts: list[Tracks]) -> Tracks:
    if not parts or sum(part.us.shape[1] for part in parts) == 0:
        raise RuntimeError("VGGSfM did not produce any multi-view tracks")
    return Tracks(
        rs=torch.cat([part.rs for part in parts]),
        ks=torch.cat([part.ks for part in parts]),
        Xs_Cr=torch.cat([part.Xs_Cr for part in parts]),
        us=torch.cat([part.us for part in parts], dim=1),
        mask=torch.cat([part.mask for part in parts], dim=1),
        ws=torch.cat([part.ws for part in parts], dim=1),
    )


def _subset_tracks(tracks: Tracks, keep: torch.Tensor) -> Tracks:
    return Tracks(
        rs=tracks.rs[keep],
        ks=tracks.ks[keep],
        Xs_Cr=tracks.Xs_Cr[keep],
        us=tracks.us[:, keep],
        mask=tracks.mask[:, keep],
        ws=tracks.ws[:, keep],
    )


def _replace_track_mask(tracks: Tracks, mask: torch.Tensor) -> Tracks:
    return replace(tracks, mask=mask, ws=tracks.ws * mask)


def _normalized_points(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    rays = torch.einsum("ij,pj->pi", torch.linalg.inv(K), homogeneous)
    return rays[:, :2] / rays[:, 2:3]


def _estimate_relative_pose(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Estimate target-from-source pose with calibrated essential RANSAC."""

    import cv2
    source_normalized = _normalized_points(source_points, source_K)
    target_normalized = _normalized_points(target_points, target_K)
    source_np = source_normalized.detach().double().cpu().numpy()
    target_np = target_normalized.detach().double().cpu().numpy()
    focal = float(
        torch.stack(
            (
                source_K[0, 0],
                source_K[1, 1],
                target_K[0, 0],
                target_K[1, 1],
            )
        ).mean()
    )
    threshold = ESSENTIAL_RANSAC_THRESHOLD_PX / max(focal, 1.0)
    essential, ransac_mask = cv2.findEssentialMat(
        source_np,
        target_np,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold,
    )
    if essential is None or ransac_mask is None:
        return None

    best = None
    for start in range(0, essential.shape[0], 3):
        candidate = essential[start : start + 3]
        if candidate.shape != (3, 3):
            continue
        count, rotation, translation, pose_mask = cv2.recoverPose(
            candidate,
            source_np,
            target_np,
            np.eye(3),
            mask=ransac_mask.copy(),
        )
        if best is None or count > best[0]:
            best = count, rotation, translation[:, 0], pose_mask[:, 0] > 0
    if best is None or best[0] < MIN_PAIR_INLIERS:
        return None

    _, rotation, translation, inliers = best
    return (
        torch.from_numpy(rotation).to(source_points),
        torch.from_numpy(translation).to(source_points),
        torch.from_numpy(inliers).to(device=source_points.device),
    )


def _build_geometric_pose_graph(
    tracks: Tracks,
    Ks: torch.Tensor,
) -> tuple[PoseGraph, Tracks]:
    """Build independently measured relative-pose edges from track geometry."""

    frame_count, point_count = tracks.mask.shape
    support = torch.zeros_like(tracks.mask, dtype=torch.int32)
    sources, targets, transforms, weights = [], [], [], []
    for source in range(frame_count):
        for target in range(source + 1, frame_count):
            shared_indices = torch.nonzero(
                tracks.mask[source] & tracks.mask[target],
                as_tuple=False,
            ).squeeze(dim=-1)
            if shared_indices.numel() < MIN_PAIR_MATCHES:
                continue
            estimate = _estimate_relative_pose(
                tracks.us[source, shared_indices],
                tracks.us[target, shared_indices],
                Ks[source],
                Ks[target],
            )
            if estimate is None:
                continue
            rotation, translation, local_inliers = estimate
            inlier_indices = shared_indices[local_inliers]
            if inlier_indices.numel() < MIN_PAIR_INLIERS:
                continue
            transform = torch.eye(4, device=Ks.device, dtype=Ks.dtype)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            sources.append(source)
            targets.append(target)
            transforms.append(transform)
            weights.append(
                tracks.ws[source, inlier_indices].mul(
                    tracks.ws[target, inlier_indices]
                ).sum()
            )
            support[source, inlier_indices] += 1
            support[target, inlier_indices] += 1

    if not transforms and frame_count > 1:
        raise RuntimeError("essential-matrix estimation produced no pose edges")
    graph = PoseGraph(
        source=torch.tensor(sources, device=Ks.device, dtype=torch.long),
        target=torch.tensor(targets, device=Ks.device, dtype=torch.long),
        target_from_source=torch.stack(transforms),
        weight=torch.stack(weights).to(dtype=Ks.dtype),
    )
    mask = tracks.mask & (support > 0)
    keep = mask.sum(dim=0) >= MIN_TRACK_OBSERVATIONS
    if not keep.any():
        raise RuntimeError("geometric verification rejected every track")
    return graph, _replace_track_mask(_subset_tracks(tracks, keep), mask[:, keep])


def _triangulate_tracks(
    tracks: Tracks,
    T_CWs: torch.Tensor,
    Ks: torch.Tensor,
) -> tuple[Tracks, torch.Tensor]:
    """Linear multi-view triangulation followed by cheirality and angle tests."""

    normalized = torch.zeros_like(tracks.us)
    for frame in range(tracks.us.shape[0]):
        normalized[frame] = _normalized_points(tracks.us[frame], Ks[frame])

    points = []
    valid_observations = tracks.mask.clone()
    valid_points = torch.zeros(
        tracks.us.shape[1], device=tracks.us.device, dtype=torch.bool
    )
    centers = -torch.einsum(
        "sji,sj->si", T_CWs[:, :3, :3], T_CWs[:, :3, 3]
    )
    for point_index in range(tracks.us.shape[1]):
        cameras = torch.nonzero(
            tracks.mask[:, point_index], as_tuple=False
        ).squeeze(dim=-1)
        projection = T_CWs[cameras, :3]
        uv = normalized[cameras, point_index]
        rows = torch.cat(
            (
                uv[:, 0:1] * projection[:, 2] - projection[:, 0],
                uv[:, 1:2] * projection[:, 2] - projection[:, 1],
            ),
            dim=0,
        )
        _, _, vh = torch.linalg.svd(rows)
        homogeneous = vh[-1]
        denominator = homogeneous[3]
        if denominator.abs() <= torch.finfo(homogeneous.dtype).eps:
            points.append(torch.full_like(homogeneous[:3], torch.nan))
            valid_observations[:, point_index] = False
            continue
        point = homogeneous[:3] / denominator
        points.append(point)
        camera_points = torch.einsum(
            "cij,j->ci", T_CWs[cameras, :3, :3], point
        ) + T_CWs[cameras, :3, 3]
        positive = torch.isfinite(camera_points).all(dim=-1) & (
            camera_points[:, 2] > 0
        )
        valid_observations[cameras, point_index] &= positive
        positive_cameras = cameras[positive]
        if positive_cameras.numel() < MIN_TRACK_OBSERVATIONS:
            continue
        directions = F.normalize(
            point.unsqueeze(dim=0) - centers[positive_cameras], dim=-1
        )
        cosine = directions @ directions.transpose(0, 1)
        cosine.fill_diagonal_(1)
        max_angle = torch.rad2deg(torch.acos(cosine.min().clamp(-1, 1)))
        valid_points[point_index] = max_angle >= MIN_TRIANGULATION_ANGLE_DEG

    points = torch.stack(points)
    valid_points &= torch.isfinite(points).all(dim=-1)
    valid_points &= valid_observations.sum(dim=0) >= MIN_TRACK_OBSERVATIONS
    if not valid_points.any():
        raise RuntimeError("triangulation rejected every geometrically valid track")
    tracks = _replace_track_mask(
        _subset_tracks(tracks, valid_points),
        valid_observations[:, valid_points],
    )
    return tracks, points[valid_points]


def _filter_ba_observations(
    tracks: Tracks,
    Xs_W: torch.Tensor,
    T_CWs: torch.Tensor,
    Ks: torch.Tensor,
) -> tuple[Tracks, torch.Tensor, torch.Tensor]:
    camera_points = torch.einsum(
        "sij,pj->spi", T_CWs[:, :3, :3], Xs_W
    ) + T_CWs[:, None, :3, 3]
    projections = torch.einsum("sij,spj->spi", Ks, camera_points)
    uv = projections[..., :2] / projections[..., 2:3].clamp_min(1.0e-8)
    error = torch.linalg.vector_norm(uv - tracks.us, dim=-1)
    mask = (
        tracks.mask
        & torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(error)
        & (camera_points[..., 2] > 0)
        & (error <= MAX_REPROJECTION_ERROR_PX)
    )
    keep = mask.sum(dim=0) >= MIN_TRACK_OBSERVATIONS
    if not keep.any():
        raise RuntimeError("post-BA filtering rejected every sparse point")
    return (
        _replace_track_mask(_subset_tracks(tracks, keep), mask[:, keep]),
        Xs_W[keep],
        keep,
    )


@torch.no_grad()
def run_sparse_sfm(
    images: torch.Tensor,
    K: torch.Tensor,
    keyframes: torch.Tensor,
    frontend: VGGSfMTrackFrontend,
    visualization_dir: str | Path | None,
    valid_mask: torch.Tensor | None = None,
) -> SparseSfMResult:
    frame_count, _, height, width = images.shape
    if frame_count < 2:
        raise ValueError("sparse SfM requires at least two images")
    Ks = K.to(device=images.device, dtype=torch.float32)
    if Ks.ndim == 2:
        Ks = Ks.unsqueeze(dim=0).expand(frame_count, -1, -1).clone()
    images = images.float()
    images_window = images.unsqueeze(dim=0)
    keyframes = keyframes.to(device=images.device, dtype=torch.long)
    if keyframes.ndim != 1 or keyframes.numel() == 0:
        raise ValueError("Pi3 Eq. (4) did not select any keyframes")
    if keyframes.min() < 0 or keyframes.max() >= frame_count:
        raise ValueError("keyframe index lies outside the image window")

    if visualization_dir is not None:
        visualization_dir = Path(visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        for path in visualization_dir.glob("reference_*.png"):
            path.unlink()

    tic()
    frontend.prepare_window(images_window)
    frontend_outputs = []
    for reference_tensor in keyframes:
        reference = int(reference_tensor)
        print(f"VGGSfM tracking from reference {reference}")
        output = frontend.match_reference(images_window, reference, valid_mask)
        frontend_outputs.append(output)
    toc("VGGSfM complete forward")

    track_parts = []
    for output in frontend_outputs:
        reference = output.reference_index
        part = _tracks_from_vggsfm(
            output,
            frame_count,
            height,
            width,
            images.device,
            images.dtype,
            valid_mask,
        )
        track_parts.append(part)
        if visualization_dir is not None:
            save_vggsfm_matching_matrix(
                visualization_dir,
                images,
                reference,
                part,
            )
    tracks = _concatenate_tracks(track_parts)
    print(
        f"Tracks before geometry: P={tracks.us.shape[1]}, "
        f"observations={int(tracks.mask.sum())}"
    )

    pose_graph, tracks = _build_geometric_pose_graph(tracks, Ks)
    T_CWs_mst = maximum_spanning_tree_initialization(frame_count, pose_graph)
    Rs = robust_rotation_averaging(
        T_CWs_mst,
        pose_graph,
        iterations=15,
        robust_delta=0.1,
    )
    centers = torch.linalg.inv(T_CWs_mst)[:, :3, 3]
    T_CWs0 = torch.eye(4, device=images.device, dtype=images.dtype).repeat(
        frame_count, 1, 1
    )
    T_CWs0[:, :3, :3] = Rs
    T_CWs0[:, :3, 3] = -torch.einsum("sij,sj->si", Rs, centers)
    tracks, Xs_W0 = _triangulate_tracks(tracks, T_CWs0, Ks)
    Xs_Cr = torch.einsum(
        "pij,pj->pi", T_CWs0[tracks.rs, :3, :3], Xs_W0
    ) + T_CWs0[tracks.rs, :3, 3]
    tracks = replace(tracks, Xs_Cr=Xs_Cr)

    ii, jj = torch.nonzero(tracks.mask, as_tuple=True)
    uv1 = torch.cat(
        (tracks.us[ii, jj], torch.ones_like(tracks.us[ii, jj, :1])), dim=-1
    )
    rays = torch.einsum("oij,oj->oi", torch.linalg.inv(Ks[ii]), uv1)
    eq5 = opt_pose_ray(
        Rs,
        centers,
        Xs_W0,
        rays,
        ii,
        jj,
        tracks.ws[ii, jj],
        iterations=EQ5_ITERATIONS,
        scale_prior_weight=SCALE_PRIOR_WEIGHT,
    )
    T_CWs5 = torch.eye(4, device=images.device, dtype=images.dtype).repeat(
        frame_count, 1, 1
    )
    T_CWs5[:, :3, :3] = Rs
    T_CWs5[:, :3, 3] = -torch.einsum("sij,sj->si", Rs, eq5.cs)
    first_ba = bundle_adjust(
        T_CWs5,
        eq5.Xs_W,
        Ks,
        torch.zeros(frame_count, 4, device=images.device, dtype=images.dtype),
        tracks,
        iterations=EQ6_ITERATIONS,
        scale_prior_weight=SCALE_PRIOR_WEIGHT,
    )
    tracks, filtered_points, keep = _filter_ba_observations(
        tracks, first_ba.Xs_W, first_ba.T_CWs, Ks
    )
    Xs_W0 = Xs_W0[keep]
    second_ba = bundle_adjust(
        first_ba.T_CWs,
        filtered_points,
        Ks,
        torch.zeros(frame_count, 4, device=images.device, dtype=images.dtype),
        tracks,
        iterations=SECOND_BA_ITERATIONS,
        scale_prior_weight=SCALE_PRIOR_WEIGHT,
    )
    tracks, final_points, keep = _filter_ba_observations(
        tracks, second_ba.Xs_W, second_ba.T_CWs, Ks
    )
    Xs_W0 = Xs_W0[keep]
    print(
        f"Geometric SfM: edges={pose_graph.source.numel()}, "
        f"points={final_points.shape[0]}, "
        f"Eq5={float(eq5.loss):.6g}, Eq6={float(second_ba.loss):.6g}"
    )
    return SparseSfMResult(
        T_CWs=second_ba.T_CWs,
        Ks=Ks,
        keyframes=keyframes,
        tracks=tracks,
        pose_graph=pose_graph,
        T_CWs_mst=T_CWs_mst,
        Rs_avg=Rs,
        Xs_W0=Xs_W0,
        Xs_W=final_points,
        eq5_loss=eq5.loss,
        eq6_loss=second_ba.loss,
    )


def _sample_map_at_tracks(
    value_map: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    height, width = value_map.shape[:2]
    grid = points.clone()
    grid[:, 0] = 2 * grid[:, 0] / max(width - 1, 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / max(height - 1, 1) - 1
    channels_first = value_map.permute(2, 0, 1)[None]
    sampled = F.grid_sample(
        channels_first,
        grid.reshape(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0].transpose(0, 1)


def _fit_scale(
    predicted_depth: torch.Tensor,
    sparse_depth: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratios = sparse_depth / predicted_depth
    hypotheses = ratios[
        torch.randint(
            ratios.numel(),
            (min(DENSE_RANSAC_TRIALS, ratios.numel()),),
            device=ratios.device,
        )
    ]
    consensus = (
        ratios.log().unsqueeze(0) - hypotheses.log().unsqueeze(1)
    ).abs() < math.log1p(SCALE_RANSAC_THRESHOLD)
    inliers = consensus[(consensus.to(weights.dtype) @ weights).argmax()]
    scale = (ratios[inliers] * weights[inliers]).sum() / weights[inliers].sum()
    return scale, inliers


def _fit_disparity_affine(
    predicted_depth: torch.Tensor,
    sparse_depth: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = predicted_depth.reciprocal()
    y = sparse_depth.reciprocal()
    trials = min(DENSE_RANSAC_TRIALS, max(x.numel() * 4, 1))
    first = torch.randint(x.numel(), (trials,), device=x.device)
    second = torch.randint(x.numel(), (trials,), device=x.device)
    dx = x[first] - x[second]
    usable = dx.abs() > torch.finfo(x.dtype).eps
    safe_dx = torch.where(usable, dx, torch.ones_like(dx))
    scale = (y[first] - y[second]) / safe_dx
    shift = y[first] - scale * x[first]
    usable &= torch.isfinite(scale) & torch.isfinite(shift) & (scale > 0)
    scale = scale[usable]
    shift = shift[usable]
    if scale.numel() == 0:
        raise RuntimeError("dense disparity alignment has no valid hypotheses")
    threshold = y.median() / DISPARITY_RANSAC_RATIO
    residual = (scale[:, None] * x + shift[:, None] - y).abs()
    consensus = residual <= threshold
    inliers = consensus[(consensus.to(weights.dtype) @ weights).argmax()]
    inlier_x = x[inliers]
    inlier_y = y[inliers]
    inlier_weight = weights[inliers]
    weight_sum = inlier_weight.sum().clamp_min(torch.finfo(x.dtype).eps)
    mean_x = (inlier_weight * inlier_x).sum() / weight_sum
    mean_y = (inlier_weight * inlier_y).sum() / weight_sum
    centered_x = inlier_x - mean_x
    scale = (
        inlier_weight * centered_x * (inlier_y - mean_y)
    ).sum() / (inlier_weight * centered_x.square()).sum().clamp_min(
        torch.finfo(x.dtype).eps
    )
    shift = mean_y - scale * mean_x
    if not torch.isfinite(scale) or not torch.isfinite(shift) or scale <= 0:
        raise RuntimeError("dense disparity alignment produced an invalid fit")
    inliers = (scale * x + shift - y).abs() <= threshold
    return scale, shift, inliers


@torch.no_grad()
def reconstruct_pi3_dense(
    images: torch.Tensor,
    sparse: SparseSfMResult,
    geometry: Mapping[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align Pi3 local point maps after BA using optimized sparse depths."""

    local_points = geometry["local_points"].squeeze(dim=0).to(
        device=images.device, dtype=torch.float32
    )
    confidence = (
        geometry["conf"]
        .squeeze(dim=0)
        .to(device=images.device, dtype=torch.float32)
        .sigmoid()
        .squeeze(dim=-1)
    )
    if valid_mask is not None:
        confidence *= valid_mask
    predicted_depth_maps = local_points[..., 2]
    height, width = predicted_depth_maps.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=local_points.dtype),
        torch.arange(width, device=images.device, dtype=local_points.dtype),
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    camera_rays = torch.einsum(
        "sij,hwj->shwi", torch.linalg.inv(sparse.Ks), pixels
    )
    camera_rays = camera_rays / camera_rays[..., 2:3].clamp_min(1.0e-8)
    T_WCs = sparse.T_WCs
    dense_points, dense_rgb, dense_frame_ids = [], [], []
    track_inliers = torch.zeros(
        sparse.Xs_W.shape[0], device=images.device, dtype=torch.bool
    )

    for frame in range(images.shape[0]):
        indices = torch.nonzero(
            sparse.tracks.mask[frame], as_tuple=False
        ).squeeze(dim=-1)
        if indices.numel() < MIN_DENSE_ALIGNMENT_POINTS:
            print(f"Dense frame {frame}: skipped, only {indices.numel()} sparse depths")
            continue
        sparse_camera = torch.einsum(
            "ij,pj->pi", sparse.T_CWs[frame, :3, :3], sparse.Xs_W[indices]
        ) + sparse.T_CWs[frame, :3, 3]
        sampled_depth = _sample_map_at_tracks(
            predicted_depth_maps[frame, ..., None],
            sparse.tracks.us[frame, indices],
        )[:, 0]
        sampled_confidence = _sample_map_at_tracks(
            confidence[frame, ..., None], sparse.tracks.us[frame, indices]
        )[:, 0]
        predicted_depth = sampled_depth
        sparse_depth = sparse_camera[:, 2]
        weights = sparse.tracks.ws[frame, indices] * sampled_confidence
        valid = (
            torch.isfinite(predicted_depth)
            & torch.isfinite(sparse_depth)
            & torch.isfinite(weights)
            & (predicted_depth > 0)
            & (sparse_depth > 0)
            & (weights > 0)
        )
        if valid.sum() < MIN_DENSE_ALIGNMENT_POINTS:
            print(f"Dense frame {frame}: skipped, insufficient valid Pi3 samples")
            continue

        if DENSE_ALIGNMENT == "scale":
            scale, local_inliers = _fit_scale(
                predicted_depth[valid], sparse_depth[valid], weights[valid]
            )
            aligned_depth = predicted_depth_maps[frame] * scale
            alignment_text = f"scale={float(scale):.6g}"
        elif DENSE_ALIGNMENT == "disparity_affine":
            scale, shift, local_inliers = _fit_disparity_affine(
                predicted_depth[valid], sparse_depth[valid], weights[valid]
            )
            predicted_disparity = predicted_depth_maps[frame].reciprocal()
            aligned_disparity = scale * predicted_disparity + shift
            aligned_depth = aligned_disparity.reciprocal()
            alignment_text = (
                f"disparity_scale={float(scale):.6g}, "
                f"disparity_shift={float(shift):.6g}"
            )
        else:
            raise ValueError(f"unsupported DENSE_ALIGNMENT {DENSE_ALIGNMENT}")

        aligned_points = camera_rays[frame] * aligned_depth.unsqueeze(dim=-1)
        valid_indices = indices[valid]
        track_inliers[valid_indices[local_inliers]] = True
        world_points = torch.einsum(
            "ij,hwj->hwi", T_WCs[frame, :3, :3], aligned_points
        ) + T_WCs[frame, None, None, :3, 3]
        mask = (
            (confidence[frame] > DEPTH_CONFIDENCE_THRESHOLD)
            & torch.isfinite(world_points).all(dim=-1)
            & torch.isfinite(aligned_depth)
            & (aligned_depth > 0)
            & ~depth_edge(aligned_depth, rtol=0.03)
        )
        print(
            f"Dense frame {frame}: {alignment_text}, "
            f"inliers={int(local_inliers.sum())}/{int(valid.sum())}, "
            f"points={int(mask.sum())}"
        )
        dense_points.append(world_points[mask])
        dense_rgb.append(images[frame].permute(1, 2, 0)[mask])
        dense_frame_ids.append(
            torch.full(
                (int(mask.sum()),), frame, device=images.device, dtype=torch.long
            )
        )
    if not dense_points:
        raise RuntimeError("no frame had enough sparse support for dense alignment")
    return (
        torch.cat(dense_points),
        torch.cat(dense_rgb),
        torch.cat(dense_frame_ids),
        track_inliers,
    )


def save_vggsfm_matching_matrix(
    output_dir: str | Path,
    images: torch.Tensor,
    reference_index: int,
    tracks: Tracks,
    cell_width: int | None = None,
) -> Path:
    """Draw corresponding points without match lines."""

    from PIL import Image, ImageDraw

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count, _, height, width = images.shape
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
            (cell_width, cell_height), resample=Image.Resampling.BICUBIC
        )

    def track_color(track_id: int) -> tuple[int, int, int]:
        hue = (track_id * 0.618033988749895) % 1.0
        return tuple(
            round(channel * 255)
            for channel in colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        )

    canvas = Image.new("RGB", (2 * cell_width, frame_count * cell_height), "black")
    x_scale = (cell_width - 1) / max(width - 1, 1)
    y_scale = (cell_height - 1) / max(height - 1, 1)
    for target_index in range(frame_count):
        row = Image.new("RGB", (2 * cell_width, cell_height), "black")
        row.paste(image_panel(images[reference_index]), (0, 0))
        row.paste(image_panel(images[target_index]), (cell_width, 0))
        draw = ImageDraw.Draw(row, "RGBA")
        valid_indices = torch.nonzero(
            tracks.mask[reference_index] & tracks.mask[target_index],
            as_tuple=False,
        ).squeeze(dim=-1)
        valid_count = int(valid_indices.numel())
        if valid_indices.numel() > MATCH_VIS_MAX_TRACKS:
            pair_weights = tracks.ws[target_index, valid_indices]
            valid_indices = valid_indices[
                pair_weights.topk(MATCH_VIS_MAX_TRACKS).indices
            ]
        for track_index in valid_indices.tolist():
            color = track_color(int(tracks.ks[track_index]))
            for panel, frame in ((0, reference_index), (1, target_index)):
                xy = tracks.us[frame, track_index].detach().float().cpu()
                x = panel * cell_width + float(xy[0]) * x_scale
                y = float(xy[1]) * y_scale
                radius = 2.5
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
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
    """Load Pi3 only for the post-BA dense reconstruction stage."""

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


@torch.no_grad()
def select_pi3_keyframes(
    images: torch.Tensor,
    K: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the original Pi3 Eq. (4) coverage test before sparse SfM."""

    frame_count = images.shape[0]
    Ks = K.to(device=images.device, dtype=torch.float32)
    if Ks.ndim == 2:
        Ks = Ks.unsqueeze(dim=0).expand(frame_count, -1, -1).clone()
    local_points = geometry["local_points"].squeeze(dim=0).float()
    confidence = geometry["conf"].squeeze(dim=0).sigmoid().squeeze(dim=-1).float()
    T_WCs = geometry["camera_poses"].squeeze(dim=0).float().clone()
    T_C0W = torch.linalg.inv(T_WCs[0])
    T_WCs = T_C0W.unsqueeze(dim=0) @ T_WCs
    frames = Frames(
        Is=images.float(),
        Xs_C=local_points,
        Cs=confidence,
        Ks=Ks,
        deltas=torch.zeros(
            frame_count, 4, device=images.device, dtype=torch.float32
        ),
        T_WCs=T_WCs,
    )
    return select_keyframes_eq4(
        frames,
        proj_threshold=KEYFRAME_PROJECTION_THRESHOLD,
        conf_threshold=DEPTH_CONFIDENCE_THRESHOLD,
        valid_mask=valid_mask,
    )


def _install_tracker_only_package(vggsfm_root: Path) -> None:
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
        raise RuntimeError("VGGSfM was imported before tracker-only initialization")
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
        version_base=None, config_dir=str((vggsfm_root / "cfgs").resolve())
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

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = _unwrap_checkpoint(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    prefixes = (
        "module.track_predictor.",
        "model.track_predictor.",
        "track_predictor.",
    )
    tracker_state = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                tracker_state[key[len(prefix) :]] = value
                break
    if not tracker_state:
        raise RuntimeError("checkpoint contains no track_predictor weights")
    result = tracker.load_state_dict(tracker_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"incomplete VGGSfM tracker checkpoint: {result}")
    return tracker.to(device).eval()


def main() -> None:
    from .image_utils import crop_resize
    from .inference import load_calibration, load_image_sequence
    from .run_glob3r_sfm import save_ply

    torch.manual_seed(RANDOM_SEED)
    output = Path(OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    if HEIGHT % 14 or WIDTH % 14:
        raise ValueError("height and width must be divisible by Pi3 patch size 14")
    if DENSE_RECONSTRUCTION not in ("none", "pi3"):
        raise ValueError(f"unsupported DENSE_RECONSTRUCTION {DENSE_RECONSTRUCTION}")

    images, paths = load_image_sequence(IMAGE_DIR, (HEIGHT, WIDTH))
    print("Input frame order")
    for index, path in enumerate(paths):
        print(f"  [{index:04d}] {path.name}")
    if CALIBRATION:
        K, calibration_width, calibration_height = load_calibration(CALIBRATION)
    else:
        focal = float(max(WIDTH, HEIGHT))
        K = torch.tensor(
            [[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]],
            dtype=torch.float32,
        )
    with Image.open(paths[0]) as source:
        if CALIBRATION:
            K[0] *= source.width / calibration_width
            K[1] *= source.height / calibration_height
        mask = np.array(Image.open(MASK).convert("L")) if MASK else None
        _, mask, resized_K = crop_resize(
            source.convert("RGB"), K.numpy(), (HEIGHT, WIDTH), mask
        )
    if CALIBRATION:
        K = torch.from_numpy(resized_K).float()
    manual_mask = torch.from_numpy(mask > 0) if mask is not None else None
    images = images.to(DEVICE)
    K = K.to(DEVICE)
    manual_mask = None if manual_mask is None else manual_mask.to(DEVICE)
    pi3_images = images if manual_mask is None else images * manual_mask[None, None]

    geometry_model = load_pi3_geometry(BACKBONE_CHECKPOINT, DEVICE)
    with torch.no_grad():
        pi3_geometry, _, _ = geometry_model.infer_window(pi3_images.unsqueeze(dim=0))
    valid_mask = (
        pi3_geometry["conf"].squeeze(0).sigmoid().squeeze(-1)
        > PI3_MASK_CONFIDENCE_THRESHOLD
    )
    if manual_mask is not None:
        valid_mask &= manual_mask[None]
    model_images = images * valid_mask[:, None]
    keyframes = select_pi3_keyframes(images, K, pi3_geometry, valid_mask)
    dense_geometry = None
    if DENSE_RECONSTRUCTION == "pi3":
        dense_geometry = {
            "local_points": pi3_geometry["local_points"].detach().cpu(),
            "conf": pi3_geometry["conf"].detach().cpu(),
        }
    del geometry_model, pi3_geometry
    if images.is_cuda:
        torch.cuda.empty_cache()

    tracker = load_vggsfm_tracker(VGGSFM_ROOT, VGGSFM_CHECKPOINT, DEVICE)
    autocast_dtype = {
        "none": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[MIXED_PRECISION]
    frontend = VGGSfMTrackFrontend(
        tracker,
        points_per_query_frame=POINTS_PER_QUERY_FRAME,
        track_confidence_threshold=TRACK_CONFIDENCE_THRESHOLD,
        tracker_size=TRACKER_SIZE,
        fine_tracking=FINE_TRACKING,
        autocast_dtype=autocast_dtype,
    ).eval()
    sparse = run_sparse_sfm(
        model_images, K, keyframes, frontend, MATCH_VIS_DIR, valid_mask
    )

    dense_points = dense_rgb = dense_frame_ids = None
    track_inliers = torch.ones(
        sparse.Xs_W.shape[0], device=images.device, dtype=torch.bool
    )
    if DENSE_RECONSTRUCTION == "pi3":
        del frontend, tracker
        if images.is_cuda:
            torch.cuda.empty_cache()
        if dense_geometry is None:
            raise RuntimeError("Pi3 dense geometry was not cached")
        dense_points, dense_rgb, dense_frame_ids, track_inliers = (
            reconstruct_pi3_dense(images, sparse, dense_geometry, valid_mask)
        )

    torch.save(
        {
            "frontend": "vggsfm_tracker_only_geometric_initialization",
            "dense_source": DENSE_RECONSTRUCTION,
            "dense_alignment": DENSE_ALIGNMENT,
            "image_paths": [str(path) for path in paths],
            "valid_mask": None if manual_mask is None else manual_mask.cpu(),
            "world_to_camera": sparse.T_CWs.cpu(),
            "camera_to_world": sparse.T_WCs.cpu(),
            "intrinsics": sparse.Ks.cpu(),
            "keyframes": sparse.keyframes.cpu(),
            "track_references": sparse.tracks.rs.cpu(),
            "track_ids": sparse.tracks.ks.cpu(),
            "track_anchor_points": sparse.tracks.Xs_Cr.cpu(),
            "track_observations": sparse.tracks.us.cpu(),
            "track_mask": sparse.tracks.mask.cpu(),
            "track_weights": sparse.tracks.ws.cpu(),
            "sparse_points_before_optimization": sparse.Xs_W0.cpu(),
            "sparse_points": sparse.Xs_W.cpu(),
            "track_inliers": track_inliers.cpu(),
            "eq5_loss": sparse.eq5_loss.cpu(),
            "eq6_loss": sparse.eq6_loss.cpu(),
        },
        output,
    )
    observation_count = sparse.tracks.mask.sum(dim=0)
    observation_count[~track_inliers] = -1
    save_ply(
        output.parent / "sparse_tracks.ply",
        sparse.Xs_W,
        scalar_fields={
            "reference_id": sparse.tracks.rs,
            "track_id": sparse.tracks.ks,
            "observation_count": observation_count,
        },
    )
    if dense_points is not None:
        save_ply(
            output.parent / "pi3_sfm.ply",
            dense_points,
            dense_rgb,
            {"frame_id": dense_frame_ids},
        )
    print(f"Selected query frames {sparse.keyframes.tolist()}")
    print(f"Saved VGGSfM-track validation to {output}")


if __name__ == "__main__":
    main()
