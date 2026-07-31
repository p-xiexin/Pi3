"""Paper-aligned single-window Glob3R structure-from-motion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
import torch

from pi3.models.glob3r.geometry import bundle_adjustment_objective
from pi3.utils.geometry import depth_edge

from .optimization import BundleAdjustmentResult, bundle_adjust
from .tracker import Tracker, Tracks


@dataclass
class Glob3RSfMConfig:
    """Appendix C.1 defaults and local solver controls."""

    tracking_points_per_keyframe: int = 512
    keyframe_projection_threshold: float = 0.2
    depth_confidence_threshold: float = 0.1
    warp_confidence_threshold: float = 0.6
    optimize_intrinsics: bool = False
    optimize_distortion: bool = False
    rotation_iterations: int = 15
    translation_iterations: int = 15
    ba_iterations: int = 20
    random_seed: int = 0
    dense_depth_ransac_iterations: int = 256
    dense_depth_ransac_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.tracking_points_per_keyframe < 1:
            raise ValueError("tracking_points_per_keyframe must be positive")


@dataclass
class SfMResult:
    world_to_camera: torch.Tensor
    camera_to_world: torch.Tensor
    points_3d_before_ba: torch.Tensor
    points_3d: torch.Tensor
    intrinsics: torch.Tensor
    distortion: torch.Tensor
    keyframes: np.ndarray
    tracks: Tracks
    raw_points: torch.Tensor
    raw_colors: torch.Tensor
    raw_frame_ids: torch.Tensor
    dense_points: Optional[torch.Tensor]
    dense_colors: Optional[torch.Tensor]
    dense_frame_ids: Optional[torch.Tensor]
    motion_objective: torch.Tensor
    bundle_adjustment_objective: torch.Tensor


def select_keyframes_eq4(
    Xs: torch.Tensor,
    Cs: torch.Tensor,
    T_CWs: torch.Tensor,
    intrinsics: torch.Tensor,
    projection_threshold: float,
    confidence_threshold: float,
) -> np.ndarray:
    """Select keyframes using the valid-projection count in paper Eq. (4).

    n_t = max_{r in K} sum_u 1[pi(T_{t->r} X_t(u)) in D,
                                z_{t->r}(u) > 0, C_t(u) > tau_c].
    Candidate t is added to K when n_t < tau_proj.
    """

    keyframes = [0]
    height, width = Xs.shape[1:3]
    pixel_threshold = projection_threshold * height * width
    for candidate in range(1, Xs.shape[0]):
        # Eq. (4): transform candidate points X_t into every existing
        # keyframe r, then retain the largest valid projection count n_t.
        keyframe_from_candidate = (
            torch.linalg.inv(T_CWs[keyframes]) @ T_CWs[candidate]
        )
        # [H, W, 3] -> [M, 3]
        candidate_points = Xs[candidate].reshape(-1, 3)
        transformed = torch.einsum(
            "kij,mj->kmi",
            keyframe_from_candidate[:, :3, :3],
            candidate_points,
        ) + keyframe_from_candidate[:, None, :3, 3]
        projected = torch.einsum(
            "kij,kmj->kmi", intrinsics[keyframes], transformed
        )
        xy = projected[..., :2] / projected[..., 2:3].clamp_min(1.0e-8)
        valid = (
            (Cs[candidate].reshape(1, -1) > confidence_threshold)
            & (transformed[..., 2] > 0)
            & (xy[..., 0] >= 0)
            & (xy[..., 0] <= width - 1)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] <= height - 1)
        )
        count = valid.sum(dim=-1).max()
        if count < pixel_threshold:
            keyframes.append(candidate)
    return np.asarray(keyframes, dtype=np.int64)


def _undistort_normalized(
    distorted: torch.Tensor,
    distortion: torch.Tensor,
    iterations: int = 12,
) -> torch.Tensor:
    """Invert the Eq. (6) Brown-Conrady model for dense backprojection."""

    k1, k2, p1, p2 = distortion.unbind()
    undistorted = distorted.clone()
    for _ in range(iterations):
        x, y = undistorted.unbind(dim=-1)
        radius_squared = x.square() + y.square()
        radial = 1 + k1 * radius_squared + k2 * radius_squared.square()
        tangential = torch.stack(
            (
                2 * p1 * x * y + p2 * (radius_squared + 2 * x.square()),
                p1 * (radius_squared + 2 * y.square()) + 2 * p2 * x * y,
            ),
            dim=-1,
        )
        undistorted = (distorted - tangential) / radial[..., None]
    return undistorted


class Glob3RSfMPipeline:
    """Run the single-window pipeline in Glob3R Secs. 3.2 and 3.3.

    The frozen backbone predicts Eq. (1) once. Eq. (4) selects keyframes, and
    Eq. (2) matches every selected keyframe to all other frames using the same
    cached features. The resulting tracks feed motion averaging in Eq. (5),
    bundle adjustment in Eq. (6), and keyframe point-cloud fusion.

    Sliding-window scheduling and cross-window association are outside this
    local pipeline. Pi3 has no Pi3X metric-scale head, so absent metric scale is
    represented by one.
    """

    def __init__(self, model, config: Optional[Glob3RSfMConfig] = None):
        self.model = model.eval()
        self.config = config or Glob3RSfMConfig()
        self.tracker = Tracker(self.model, self.config)

    @torch.no_grad()
    def run(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        matching_callback: Optional[
            Callable[[torch.Tensor, Mapping[int, object]], None]
        ] = None,
    ) -> SfMResult:
        """Reconstruct one image window with the paper's local SfM flow."""

        frame_count = images.shape[0]
        images = images.float()
        intrinsics = intrinsics.to(device=images.device, dtype=images.dtype)
        if intrinsics.ndim == 2:
            # [3, 3] -> [1, 3, 3]
            intrinsics = intrinsics.unsqueeze(dim=0)
            # [1, 3, 3] -> [N, 3, 3]
            intrinsics = intrinsics.expand(frame_count, -1, -1).clone()

        # [N, 3, H, W] -> [B, N, 3, H, W]
        window_images = images.unsqueeze(dim=0)

        # Sec. 3.2: evaluate Eq. (1) once, select Eq. (4) keyframes here, then
        # evaluate Eq. (2) for those references using the cached features.
        print(f"Backbone inference: frames [0, {frame_count - 1}]")
        geometry, patch_tokens, encoder_features = self.model.infer_window(
            window_images
        )
        # [B, N, H, W, 3] -> [N, H, W, 3]
        Xs = geometry["local_points"].squeeze(dim=0)
        # [B, N, 4, 4] -> [N, 4, 4]
        T_CWs = geometry["camera_poses"].squeeze(dim=0)
        # [B, N, H, W, 1] -> [N, H, W, 1]
        Cs_logits = geometry["conf"].squeeze(dim=0)
        # [N, H, W, 1] -> [N, H, W]
        Cs = Cs_logits.sigmoid().squeeze(dim=-1)
        metric = geometry.get("metric")
        ss = torch.as_tensor(
            1.0 if metric is None else metric.reshape(-1)[0],
            device=images.device,
            dtype=images.dtype,
        )
        keyframes = select_keyframes_eq4(
            Xs,
            Cs,
            T_CWs,
            intrinsics,
            projection_threshold=self.config.keyframe_projection_threshold,
            confidence_threshold=self.config.depth_confidence_threshold,
        )
        tracking = self.tracker(
            patch_tokens,
            encoder_features,
            window_images,
            Xs,
            Cs,
            T_CWs,
            ss,
            intrinsics,
            keyframes,
            matching_callback=matching_callback,
        )

        # Sec. 3.3: motion averaging implements Eq. (5); BA implements Eq. (6).
        ba = self._bundle_adjust(
            tracking.world_to_camera,
            tracking.points_3d,
            tracking.tracks,
        )

        # Sec. 3.3: recover keyframe depth scales and fuse dense geometry.
        raw_points, raw_colors, raw_frame_ids = self._reconstruct_raw(
            images, Xs, T_CWs, Cs, ss, keyframes
        )
        dense_points, dense_colors, dense_frame_ids = self._reconstruct_dense(
            ba,
            images,
            Xs,
            Cs,
            ss,
            tracking.tracks,
            keyframes,
        )
        return SfMResult(
            world_to_camera=ba.world_to_camera,
            camera_to_world=torch.linalg.inv(ba.world_to_camera),
            points_3d_before_ba=tracking.points_3d,
            points_3d=ba.points_3d,
            intrinsics=ba.intrinsics,
            distortion=ba.distortion,
            keyframes=keyframes,
            tracks=tracking.tracks,
            raw_points=raw_points,
            raw_colors=raw_colors,
            raw_frame_ids=raw_frame_ids,
            dense_points=dense_points,
            dense_colors=dense_colors,
            dense_frame_ids=dense_frame_ids,
            motion_objective=tracking.objective,
            bundle_adjustment_objective=ba.objective,
        )

    def _bundle_adjust(
        self,
        world_to_camera: torch.Tensor,
        points_3d: torch.Tensor,
        tracks: Tracks,
    ) -> BundleAdjustmentResult:
        """Optimize poses and sparse points with Eq. (6)."""

        # Known calibration remains fixed by default. The optional flags expose
        # the shared intrinsics and distortion variables already present in Eq. (6).
        distortion = torch.zeros(
            tracks.intrinsics.shape[0],
            4,
            device=tracks.intrinsics.device,
            dtype=tracks.intrinsics.dtype,
        )
        initial_objective = bundle_adjustment_objective(
            points_3d,
            world_to_camera,
            tracks.intrinsics,
            tracks.observations,
            tracks.camera_indices,
            tracks.point_indices,
            tracks.confidence,
            distortion,
            robust_delta=2.0,
        )
        print(
            "BA input: "
            f"cameras={tracks.intrinsics.shape[0]}, "
            f"points={points_3d.shape[0]}, "
            f"observations={tracks.observations.shape[0]}, "
            f"max_iterations={self.config.ba_iterations}, "
            f"intrinsics={'optimized(shared)' if self.config.optimize_intrinsics else 'fixed'}, "
            f"distortion={'optimized(shared)' if self.config.optimize_distortion else 'fixed-zero'}"
        )
        observations_per_camera = torch.bincount(
            tracks.camera_indices, minlength=tracks.intrinsics.shape[0]
        ).tolist()
        print(f"BA observations per camera: {observations_per_camera}")
        print(
            "BA tracking confidence: "
            f"min={float(tracks.confidence.min()):.4f}, "
            f"mean={float(tracks.confidence.mean()):.4f}, "
            f"max={float(tracks.confidence.max()):.4f}"
        )
        print(f"BA initial objective: {float(initial_objective):.6g}")
        result = bundle_adjust(
            world_to_camera,
            points_3d,
            tracks.intrinsics,
            tracks.observations,
            tracks.camera_indices,
            tracks.point_indices,
            tracks.confidence,
            distortion,
            optimize_intrinsics=self.config.optimize_intrinsics,
            optimize_distortion=self.config.optimize_distortion,
            shared_intrinsics=True,
            iterations=self.config.ba_iterations,
        )
        reduction = (initial_objective - result.objective) / initial_objective.clamp_min(
            1.0e-12
        )
        print(
            f"BA final objective: {float(result.objective):.6g} "
            f"(reduction={float(reduction) * 100:.2f}%)"
        )
        return result

    def _reconstruct_raw(
        self,
        images: torch.Tensor,
        Xs: torch.Tensor,
        T_CWs: torch.Tensor,
        Cs: torch.Tensor,
        ss: torch.Tensor,
        keyframes: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse Eq. (4) keyframe point maps using predicted camera poses."""

        frame_indices = torch.arange(Xs.shape[0], device=images.device)[keyframes]
        images = images[frame_indices]
        Xs = Xs[frame_indices] * ss
        Cs = Cs[frame_indices]
        T_CWs = T_CWs[frame_indices].clone()
        T_CWs[:, :3, 3] *= ss
        X_Ws = torch.einsum(
            "nij,nhwj->nhwi", T_CWs[:, :3, :3], Xs
        ) + T_CWs[:, None, None, :3, 3]
        valid = (
            (Cs > self.config.depth_confidence_threshold)
            & ~depth_edge(Xs[..., 2], rtol=0.03)
            & torch.isfinite(X_Ws).all(dim=-1)
            & (Xs[..., 2] > 0)
        )
        colors = images.permute(0, 2, 3, 1)
        frame_ids = frame_indices[:, None, None].expand_as(valid)
        return X_Ws[valid], colors[valid], frame_ids[valid]

    def _reconstruct_dense(
        self,
        ba: BundleAdjustmentResult,
        images: torch.Tensor,
        Xs: torch.Tensor,
        Cs: torch.Tensor,
        ss: torch.Tensor,
        tracks: Tracks,
        keyframes: np.ndarray,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Apply Sec. 3.3 depth-scale RANSAC and fuse keyframe point maps."""

        fused_points, fused_colors, fused_frame_ids = [], [], []
        generator = torch.Generator(device=images.device).manual_seed(
            self.config.random_seed
        )
        for frame in keyframes:
            mask = tracks.camera_indices == frame
            if mask.sum() < 2:
                continue
            image = images[frame]
            depth = Xs[frame, ..., 2] * ss
            predicted = tracks.predicted_depth[mask]
            weight = tracks.confidence[mask].clamp_min(0)
            camera_point = (
                torch.einsum(
                    "ij,nj->ni",
                    ba.world_to_camera[frame, :3, :3],
                    ba.points_3d[tracks.point_indices[mask]],
                )
                + ba.world_to_camera[frame, :3, 3]
            )
            valid_ratio = (
                torch.isfinite(camera_point[:, 2])
                & torch.isfinite(predicted)
                & (camera_point[:, 2] > 0)
                & (predicted > 0)
                & (weight > 0)
            )
            ratio = camera_point[valid_ratio, 2] / predicted[valid_ratio]
            weight = weight[valid_ratio]
            if ratio.numel() < 2:
                print(f"Dense scale frame {frame}: skipped ({ratio.numel()} valid ratios)")
                continue
            candidate_indices = torch.multinomial(
                weight,
                self.config.dense_depth_ransac_iterations,
                replacement=True,
                generator=generator,
            )
            candidates = ratio[candidate_indices]
            relative_error = (ratio[None] - candidates[:, None]).abs() / candidates[
                :, None
            ].abs().clamp_min(1.0e-8)
            best = candidates[
                (
                    (relative_error < self.config.dense_depth_ransac_threshold)
                    * weight[None]
                ).sum(dim=1).argmax()
            ]
            inlier = (
                (ratio - best).abs() / best.abs().clamp_min(1.0e-8)
                < self.config.dense_depth_ransac_threshold
            )
            inlier_ratio = ratio[inlier]
            inlier_weight = weight[inlier]
            order = inlier_ratio.argsort()
            cumulative_weight = inlier_weight[order].cumsum(dim=0)
            median_index = torch.searchsorted(
                cumulative_weight, cumulative_weight[-1] * 0.5
            ).clamp_max(order.numel() - 1)
            scale = inlier_ratio[order[median_index]]
            relative_mad = ((inlier_ratio - scale).abs() / scale.abs()).median()
            weighted_inlier_fraction = inlier_weight.sum() / weight.sum()
            print(
                f"Dense scale frame {frame}: scale={float(scale):.6g}, "
                f"inliers={int(inlier.sum())}/{ratio.numel()}, "
                f"weighted_inliers={float(weighted_inlier_fraction) * 100:.2f}%, "
                f"relative_mad={float(relative_mad):.4f}"
            )
            depth = depth * scale
            confidence_mask = Cs[frame] > self.config.depth_confidence_threshold
            y_grid, x_grid = torch.meshgrid(
                torch.arange(depth.shape[0], device=depth.device, dtype=depth.dtype),
                torch.arange(depth.shape[1], device=depth.device, dtype=depth.dtype),
                indexing="ij",
            )
            homogeneous = torch.stack(
                (x_grid, y_grid, torch.ones_like(x_grid)), dim=-1
            )
            normalized = torch.einsum(
                "ij,hwj->hwi", torch.linalg.inv(ba.intrinsics[frame]), homogeneous
            )
            normalized_xy = _undistort_normalized(
                normalized[..., :2] / normalized[..., 2:3],
                ba.distortion[frame],
            )
            camera_points = torch.cat(
                (normalized_xy * depth[..., None], depth[..., None]), dim=-1
            )
            valid = (
                confidence_mask
                & ~depth_edge(depth, rtol=0.03)
                & torch.isfinite(camera_points).all(dim=-1)
                & (depth > 0)
            )
            camera_to_world = torch.linalg.inv(ba.world_to_camera[frame])
            world_points = torch.einsum(
                "ij,nj->ni", camera_to_world[:3, :3], camera_points[valid]
            ) + camera_to_world[:3, 3]
            fused_points.append(world_points)
            fused_colors.append(image.permute(1, 2, 0)[valid])
            fused_frame_ids.append(
                torch.full(
                    (world_points.shape[0],),
                    frame,
                    device=world_points.device,
                    dtype=torch.long,
                )
            )
        if not fused_points:
            return None, None, None
        return (
            torch.cat(fused_points),
            torch.cat(fused_colors),
            torch.cat(fused_frame_ids),
        )


__all__ = [
    "Glob3RSfMPipeline",
    "Glob3RSfMConfig",
    "SfMResult",
    "select_keyframes_eq4",
]
