"""Shared SIFT query features for every tracks frontend."""

import torch


QUERY_BUCKET_GRID = (8, 6)
QUERY_CANDIDATE_MULTIPLIER = 4


def _bucket_indices(points, scores, image_size, max_points):
    """Select scored features with per-cell coverage and global budget refill."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("keypoints must have shape [P,2]")
    if scores.shape != points.shape[:1]:
        raise ValueError("keypoint scores must have shape [P]")

    height, width = map(int, image_size)
    columns, rows = QUERY_BUCKET_GRID
    bucket_x = (points[:, 0] * columns / width).long().clamp(0, columns - 1)
    bucket_y = (points[:, 1] * rows / height).long().clamp(0, rows - 1)
    bucket_ids = bucket_y * columns + bucket_x
    bucket_count = columns * rows
    points_per_bucket, remainder = divmod(int(max_points), bucket_count)

    selected_mask = torch.zeros(points.shape[0], device=points.device, dtype=torch.bool)
    selected_parts = []
    for bucket_id in range(bucket_count):
        candidates = torch.where(bucket_ids == bucket_id)[0]
        limit = points_per_bucket + int(bucket_id < remainder)
        if not limit or not candidates.numel():
            continue
        order = scores[candidates].argsort(descending=True)
        chosen = candidates[order[:limit]]
        selected_parts.append(chosen)
        selected_mask[chosen] = True

    selected = (
        torch.cat(selected_parts)
        if selected_parts
        else points.new_empty(0, dtype=torch.long)
    )
    remaining_budget = min(int(max_points), points.shape[0]) - selected.numel()
    if remaining_budget > 0:
        candidates = torch.where(~selected_mask)[0]
        order = scores[candidates].argsort(descending=True)
        selected = torch.cat((selected, candidates[order[:remaining_budget]]))
    return selected


class SIFTFeatures:
    """Extract one reusable set of LightGlue SIFT queries per keyframe."""

    def __init__(self, max_points, device):
        try:
            from lightglue import SIFT
        except ImportError as error:
            raise ImportError("lightglue is required for SIFT track queries") from error
        self.max_points = int(max_points)
        candidate_count = self.max_points * QUERY_CANDIDATE_MULTIPLIER
        self.extractor = SIFT(max_num_keypoints=candidate_count).to(device).eval()

    @torch.no_grad()
    def extract(self, image, valid_mask=None):
        """Extract ``P x 2`` subpixel coordinates from a ``3 x H x W`` image."""
        if valid_mask is not None and valid_mask.shape != image.shape[-2:]:
            raise ValueError("valid_mask must have shape [H,W]")
        invalid = None if valid_mask is None else (~valid_mask.bool())[None]
        features = self.extractor.extract(image[None], invalid_mask=invalid)
        points = features["keypoints"][0]
        scores = features["keypoint_scores"][0]
        finite = torch.isfinite(points).all(-1) & torch.isfinite(scores)
        points, scores = points[finite], scores[finite]
        if valid_mask is not None and points.numel():
            pixels = points.round().long()
            pixels[:, 0].clamp_(0, image.shape[-1] - 1)
            pixels[:, 1].clamp_(0, image.shape[-2] - 1)
            keep = valid_mask[pixels[:, 1], pixels[:, 0]]
            points, scores = points[keep], scores[keep]
        if points.numel() == 0:
            raise RuntimeError("SIFT found no valid query points")
        selected = _bucket_indices(
            points, scores, image.shape[-2:], self.max_points
        )
        return points[selected].to(device=image.device, dtype=image.dtype)


__all__ = [
    "QUERY_BUCKET_GRID",
    "QUERY_CANDIDATE_MULTIPLIER",
    "SIFTFeatures",
]
