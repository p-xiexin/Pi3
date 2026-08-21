"""Scale dense Pi3 keyframe maps with optimized sparse tracks."""

import math

import torch

from pi3.utils.geometry import depth_edge


def _scale_ransac(scales, weights, threshold=0.1):
    """Estimate a weighted median scale from the strongest log-ratio consensus."""
    consensus = (scales.log()[None] - scales.log()[:, None]).abs() < math.log1p(threshold)
    inliers = consensus[(consensus.to(weights.dtype) @ weights).argmax()]
    values, order = scales[inliers].sort()
    ordered_weights = weights[inliers][order]
    index = torch.searchsorted(ordered_weights.cumsum(0), 0.5 * ordered_weights.sum())
    return values[index], inliers


@torch.no_grad()
def reconstruct(graph, frames):
    """Align dense keyframe maps to optimized tracks and return sparse and dense clouds."""
    device = next(iter(graph.poses.values())).device
    sparse_ids = torch.nonzero(
        graph.point_initialized[:graph.point_count], as_tuple=False
    ).squeeze(-1)
    sparse = graph.point_positions[sparse_ids]
    dense_points, dense_colors, dense_frame_ids = [], [], []
    inlier_points = set()
    point_weights = torch.zeros(graph.point_count, device=device)
    observation_count = torch.zeros(
        graph.point_count, device=device, dtype=torch.long
    )
    # FactorGraph removes duplicate anchor observations before storage, so this
    # count is the actual multi-view support used by the optimizer.
    for _, _, point_ids, _, weights in graph.observations:
        point_weights.index_add_(0, point_ids, weights)
        observation_count.index_add_(0, point_ids, torch.ones_like(point_ids))
    for frame_id in sorted(frames.keyframes):
        point_ids = torch.nonzero(
            (graph.point_references[:graph.point_count] == frame_id)
            & graph.point_initialized[:graph.point_count], as_tuple=False,
        ).squeeze(-1)
        if not point_ids.numel():
            continue
        anchors = graph.point_anchors[point_ids]
        optimized = graph.point_positions[point_ids]
        pose = graph.poses[frame_id]
        camera_points = torch.einsum("ij,pj->pi", pose[:3, :3], optimized) + pose[:3, 3]
        # Pi3 predicts each window up to scale. Optimized tracks determine one
        # robust positive scale for every stored keyframe point map.
        scales = camera_points[:, 2] / anchors[:, 2].clamp_min(1.0e-8)
        weights = point_weights[point_ids].to(scales.dtype)
        valid = torch.isfinite(scales) & (scales > 0) & (weights > 0)
        if not valid.any():
            continue
        scale, inliers = _scale_ransac(scales[valid], weights[valid])
        valid_ids = point_ids[valid]
        inlier_points.update(valid_ids[inliers].tolist())
        image, pointmap, confidence = frames.dense[frame_id]
        image, pointmap, confidence = image.to(device), pointmap.to(device), confidence.to(device)
        pointmap = pointmap * scale
        world_from_camera = torch.linalg.inv(pose)
        world = torch.einsum(
            "ij,hwj->hwi", world_from_camera[:3, :3], pointmap
        ) + world_from_camera[None, None, :3, 3]
        mask = (
            (confidence > 0) & ~depth_edge(pointmap[..., 2], rtol=0.03)
            & torch.isfinite(world).all(-1) & (pointmap[..., 2] > 0)
        )
        dense_points.append(world[mask].cpu())
        dense_colors.append(image.permute(1, 2, 0)[mask].cpu())
        dense_frame_ids.append(torch.full((int(mask.sum()),), frame_id, dtype=torch.long))
    if not dense_points:
        raise RuntimeError("optimized keyframes produced no dense reconstruction")
    sparse_inliers = torch.tensor(
        [point_id in inlier_points for point_id in sparse_ids.tolist()]
    )
    sparse_obs_cnt = observation_count[sparse_ids].cpu()
    # Match the local_opt export contract. Positive values retain the true
    # multi-view support, while -1 identifies scale-RANSAC outlier tracks.
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


__all__ = ["reconstruct"]
