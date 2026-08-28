"""Align cached Pi3 frame maps with optimized multi-view track depths."""

import torch
import torch.nn.functional as F

from pi3.utils.geometry import depth_edge


DENSE_RANSAC_TRIALS = 2048
DISPARITY_RANSAC_RATIO = 30.0
MIN_DENSE_ALIGNMENT_POINTS = 8


def sample_map_at_points(value_map, points):
    """Bilinearly sample an HWC or HW tensor at subpixel ``(x, y)`` points."""
    if value_map.ndim == 2:
        value_map = value_map[..., None]
    if value_map.ndim != 3:
        raise ValueError("value_map must have shape [H,W] or [H,W,C]")
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [N,2]")
    height, width, channels = value_map.shape
    if not points.numel():
        return value_map.new_empty((0, channels))
    grid = points.to(device=value_map.device, dtype=value_map.dtype).clone()
    grid[:, 0] = 2 * grid[:, 0] / max(width - 1, 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / max(height - 1, 1) - 1
    sampled = F.grid_sample(
        value_map.permute(2, 0, 1)[None],
        grid.reshape(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0].transpose(0, 1)


def fit_disparity_affine(
    predicted_depth,
    sparse_depth,
    weights,
    trials=DENSE_RANSAC_TRIALS,
    threshold_ratio=DISPARITY_RANSAC_RATIO,
    min_points=MIN_DENSE_ALIGNMENT_POINTS,
):
    """Fit ``1 / sparse_depth = scale / predicted_depth + shift`` by RANSAC."""
    predicted_depth = torch.as_tensor(predicted_depth)
    sparse_depth = torch.as_tensor(
        sparse_depth, device=predicted_depth.device, dtype=predicted_depth.dtype
    )
    weights = torch.as_tensor(
        weights, device=predicted_depth.device, dtype=predicted_depth.dtype
    )
    if predicted_depth.ndim != 1:
        raise ValueError("depth and weight inputs must be one-dimensional")
    if sparse_depth.shape != predicted_depth.shape or weights.shape != predicted_depth.shape:
        raise ValueError("predicted_depth, sparse_depth, and weights must have equal shape")
    if int(min_points) < 2:
        raise ValueError("min_points must be at least two")
    if int(trials) < 1:
        raise ValueError("trials must be positive")
    if float(threshold_ratio) <= 0:
        raise ValueError("threshold_ratio must be positive")

    valid = (
        torch.isfinite(predicted_depth)
        & torch.isfinite(sparse_depth)
        & torch.isfinite(weights)
        & (predicted_depth > 0)
        & (sparse_depth > 0)
        & (weights > 0)
    )
    if int(valid.sum()) < int(min_points):
        raise RuntimeError(
            f"dense disparity alignment requires at least {int(min_points)} valid points"
        )
    x = predicted_depth[valid].reciprocal()
    y = sparse_depth[valid].reciprocal()
    fit_weights = weights[valid]
    hypothesis_count = min(int(trials), max(int(x.numel()) * 4, 1))
    first = torch.randint(x.numel(), (hypothesis_count,), device=x.device)
    second = torch.randint(x.numel(), (hypothesis_count,), device=x.device)
    dx = x[first] - x[second]
    usable = dx.abs() > torch.finfo(x.dtype).eps
    safe_dx = torch.where(usable, dx, torch.ones_like(dx))
    scales = (y[first] - y[second]) / safe_dx
    shifts = y[first] - scales * x[first]
    usable &= torch.isfinite(scales) & torch.isfinite(shifts) & (scales > 0)
    scales, shifts = scales[usable], shifts[usable]
    if not scales.numel():
        raise RuntimeError("dense disparity alignment has no valid hypotheses")

    threshold = y.median() / float(threshold_ratio)
    residual = (scales[:, None] * x + shifts[:, None] - y).abs()
    consensus = residual <= threshold
    inliers = consensus[(consensus.to(fit_weights.dtype) @ fit_weights).argmax()]
    inlier_x, inlier_y, inlier_weight = x[inliers], y[inliers], fit_weights[inliers]
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
    if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(shift)) or scale <= 0:
        raise RuntimeError("dense disparity alignment produced an invalid fit")
    fit_inliers = (scale * x + shift - y).abs() <= threshold
    inliers = torch.zeros_like(valid)
    inliers[valid] = fit_inliers
    return scale, shift, inliers


def _active_observation_mask(graph, chunk_index, count, device):
    """Return the persistent graph activity mask when observation IDs exist."""
    observation_ids = getattr(graph, "observation_ids", None)
    inactive = getattr(graph, "inactive_observation_ids", None)
    if observation_ids is None or inactive is None:
        return torch.ones(count, device=device, dtype=torch.bool)
    if chunk_index >= len(observation_ids):
        raise RuntimeError("observation ID registry is shorter than graph observations")
    ids = observation_ids[chunk_index].to(device=device)
    if ids.numel() != count:
        raise RuntimeError("observation ID registry is misaligned with graph observations")
    return torch.tensor(
        [int(observation_id) not in inactive for observation_id in ids.tolist()],
        device=device,
        dtype=torch.bool,
    )


def _frame_observations(graph, frame_id, device):
    """Gather optimized track IDs, pixels, and weights observed by one frame."""
    point_parts, uv_parts, weight_parts = [], [], []
    for chunk_index, (
        _, observation_frames, point_ids, uv, weights
    ) in enumerate(graph.observations):
        observation_frames = observation_frames.to(device=device)
        point_ids = point_ids.to(device=device)
        active = _active_observation_mask(
            graph, chunk_index, point_ids.numel(), device
        )
        mask = active & (observation_frames == int(frame_id))
        if mask.any():
            selected = point_ids[mask]
            mask_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            initialized = graph.point_initialized[selected]
            if initialized.any():
                chosen = mask_indices[initialized]
                point_parts.append(selected[initialized])
                uv_parts.append(uv.to(device=device)[chosen])
                weight_parts.append(weights.to(device=device)[chosen])
    if not point_parts:
        return None
    return torch.cat(point_parts), torch.cat(uv_parts), torch.cat(weight_parts)


@torch.no_grad()
def reconstruct(graph, frames):
    """Align every stored Pi3 depth map from its subpixel sparse observations."""
    if not graph.poses:
        raise RuntimeError("reconstruction requires at least one optimized camera")
    device = next(iter(graph.poses.values())).device
    sparse_ids = torch.nonzero(
        graph.point_initialized[:graph.point_count], as_tuple=False
    ).squeeze(-1)
    sparse = graph.point_positions[sparse_ids]
    if not sparse_ids.numel():
        raise RuntimeError("reconstruction requires initialized sparse points")

    observation_count = torch.zeros(
        graph.point_count, device=device, dtype=torch.long
    )
    for chunk_index, (_, _, point_ids, _, _) in enumerate(graph.observations):
        point_ids = point_ids.to(device=device)
        active = _active_observation_mask(
            graph, chunk_index, point_ids.numel(), device
        )
        observation_count.index_add_(
            0,
            point_ids[active],
            torch.ones_like(point_ids[active], dtype=torch.long),
        )

    dense_points, dense_colors, dense_frame_ids = [], [], []
    inlier_points = torch.zeros(graph.point_count, device=device, dtype=torch.bool)
    for frame_id in sorted(frames.dense):
        if frame_id not in frames.dense or frame_id not in graph.poses:
            continue
        observations = _frame_observations(graph, frame_id, device)
        if observations is None:
            continue
        point_ids, uv, observation_weights = observations
        if point_ids.numel() < MIN_DENSE_ALIGNMENT_POINTS:
            continue

        pose = graph.poses[frame_id]
        dtype = pose.dtype
        image, stored_depth, confidence, frame_scale = frames.dense[frame_id]
        image = image.to(device=device)
        stored_depth = stored_depth.to(device=device, dtype=dtype)
        confidence = confidence.to(device=device, dtype=dtype)
        frame_scale = torch.as_tensor(frame_scale, device=device, dtype=dtype)
        if frame_scale.ndim != 0 or not bool(torch.isfinite(frame_scale)) or not bool(
            frame_scale > 0
        ):
            raise ValueError(f"frame {frame_id} Pi3 scale must be finite and positive")
        if stored_depth.ndim == 3 and stored_depth.shape[-1] == 3:
            predicted_depth = stored_depth[..., 2]
        elif stored_depth.ndim == 2:
            predicted_depth = stored_depth
        else:
            raise ValueError(
                f"frame {frame_id} Pi3 depth must have shape [H,W]"
            )
        if confidence.ndim == 3 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if confidence.shape != predicted_depth.shape:
            raise ValueError(f"frame {frame_id} confidence shape differs from its depth map")
        if image.ndim != 3 or image.shape[0] != 3 or image.shape[1:] != predicted_depth.shape:
            raise ValueError(f"frame {frame_id} image and Pi3 depth shapes differ")

        metric_depth = frame_scale * predicted_depth
        optimized = graph.point_positions[point_ids].to(dtype=dtype)
        sparse_camera = torch.einsum(
            "ij,pj->pi", pose[:3, :3], optimized
        ) + pose[:3, 3]
        sampled_metric_depth = sample_map_at_points(metric_depth, uv)[:, 0]
        sampled_confidence = sample_map_at_points(confidence, uv)[:, 0]
        sparse_depth = sparse_camera[:, 2]
        weights = observation_weights.to(dtype=dtype) * sampled_confidence
        valid = (
            torch.isfinite(sampled_metric_depth)
            & torch.isfinite(sparse_depth)
            & torch.isfinite(weights)
            & (sampled_metric_depth > 0)
            & (sparse_depth > 0)
            & (weights > 0)
        )
        if int(valid.sum()) < MIN_DENSE_ALIGNMENT_POINTS:
            continue
        try:
            disparity_scale, disparity_shift, local_inliers = fit_disparity_affine(
                sampled_metric_depth[valid], sparse_depth[valid], weights[valid]
            )
        except RuntimeError:
            continue

        predicted_disparity = metric_depth.reciprocal()
        aligned_disparity = (
            disparity_scale * predicted_disparity + disparity_shift
        )
        aligned_depth = aligned_disparity.reciprocal()
        height, width = aligned_depth.shape
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
        K = graph.intrinsics[frame_id].to(device=device, dtype=dtype)
        camera_rays = torch.einsum("ij,hwj->hwi", torch.linalg.inv(K), pixels)
        camera_rays /= camera_rays[..., 2:3].clamp_min(1.0e-8)
        camera_points = camera_rays * aligned_depth[..., None]
        world_from_camera = torch.linalg.inv(pose)
        world = torch.einsum(
            "ij,hwj->hwi", world_from_camera[:3, :3], camera_points
        ) + world_from_camera[None, None, :3, 3]
        valid_depth = torch.isfinite(aligned_depth) & (aligned_depth > 0)
        mask = (
            (confidence > 0)
            & valid_depth
            & torch.isfinite(world).all(-1)
            & ~depth_edge(aligned_depth, rtol=0.03, mask=valid_depth)
        )
        if not mask.any():
            continue
        print(
            f"dense reconstruction frame={frame_id} "
            f"relative_scale={float(frame_scale):.6g} "
            f"final_disparity_scale="
            f"{float(disparity_scale / frame_scale):.6g}"
        )
        valid_ids = point_ids[valid]
        inlier_points[valid_ids[local_inliers]] = True
        dense_points.append(world[mask].cpu())
        dense_colors.append(image.permute(1, 2, 0)[mask].cpu())
        dense_frame_ids.append(
            torch.full((int(mask.sum()),), frame_id, dtype=torch.long)
        )

    if not dense_points:
        raise RuntimeError("no frame had enough sparse support for dense alignment")
    sparse_inliers = inlier_points[sparse_ids].cpu()
    sparse_obs_cnt = observation_count[sparse_ids].cpu()
    sparse_obs_cnt[~sparse_inliers] = -1
    return {
        "sparse_points": sparse.cpu(),
        "sparse_point_ids": sparse_ids.cpu(),
        "sparse_frame_ids": graph.point_references[sparse_ids].cpu(),
        "sparse_obs_cnt": sparse_obs_cnt,
        "sparse_inliers": sparse_inliers,
        "dense_points": torch.cat(dense_points),
        "dense_colors": torch.cat(dense_colors),
        "dense_frame_ids": torch.cat(dense_frame_ids),
    }


__all__ = [
    "DENSE_RANSAC_TRIALS",
    "DISPARITY_RANSAC_RATIO",
    "MIN_DENSE_ALIGNMENT_POINTS",
    "fit_disparity_affine",
    "reconstruct",
    "sample_map_at_points",
]
