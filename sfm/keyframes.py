"""Paper Eq. (4) keyframe selection for chronological sliding windows."""

import torch


KEYFRAME_PROJECTION_THRESHOLD = 0.7
KEYFRAME_CONFIDENCE_THRESHOLD = 0.1
KEYFRAME_MAX_INTERVAL = 5


def _validate_geometry(points, confidence, poses, intrinsics, valid_mask):
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [S,H,W,3]")
    frame_count, height, width, _ = points.shape
    if confidence.shape != (frame_count, height, width):
        raise ValueError("confidence must have shape [S,H,W]")
    if poses.shape != (frame_count, 4, 4):
        raise ValueError("poses must have shape [S,4,4]")
    if intrinsics.ndim == 2:
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape [3,3] or [S,3,3]")
        intrinsics = intrinsics.unsqueeze(0).expand(frame_count, -1, -1)
    elif intrinsics.shape != (frame_count, 3, 3):
        raise ValueError("intrinsics must have shape [3,3] or [S,3,3]")
    if valid_mask is None:
        valid_mask = torch.ones_like(confidence, dtype=torch.bool)
    elif valid_mask.ndim == 2:
        if valid_mask.shape != (height, width):
            raise ValueError("valid_mask must have shape [H,W] or [S,H,W]")
        valid_mask = valid_mask.unsqueeze(0).expand(frame_count, -1, -1)
    elif valid_mask.shape != (frame_count, height, width):
        raise ValueError("valid_mask must have shape [H,W] or [S,H,W]")
    return intrinsics, valid_mask.bool()


def select_keyframes_eq4_window(
    frame_ids,
    points,
    confidence,
    poses,
    intrinsics,
    valid_mask=None,
    existing_keyframes=(),
    projection_threshold=KEYFRAME_PROJECTION_THRESHOLD,
    confidence_threshold=KEYFRAME_CONFIDENCE_THRESHOLD,
    maximum_interval=KEYFRAME_MAX_INTERVAL,
):
    """Return local keyframe indices using local_opt's Pi3 Eq. (4) rule.

    ``poses`` are camera-to-world transforms. Existing keyframes inside the
    window define its processed temporal prefix. Selection resumes after the
    latest such frame, so an overlapping window neither revisits old frames nor
    uses a later existing keyframe to evaluate an earlier target.
    """
    ids = [int(frame_id) for frame_id in frame_ids]
    if len(ids) != points.shape[0]:
        raise ValueError("frame_ids and geometry must have the same length")
    if any(current <= previous for previous, current in zip(ids, ids[1:])):
        raise ValueError("frame_ids must be strictly increasing")
    if not 0.0 <= float(projection_threshold) <= 1.0:
        raise ValueError("projection_threshold must be within [0,1]")
    if int(maximum_interval) < 1:
        raise ValueError("maximum_interval must be positive")
    intrinsics, valid_mask = _validate_geometry(
        points, confidence, poses, intrinsics, valid_mask
    )
    if not ids:
        return torch.empty(0, device=points.device, dtype=torch.long)

    existing = {int(frame_id) for frame_id in existing_keyframes}
    selected = [index for index, frame_id in enumerate(ids) if frame_id in existing]
    if selected:
        start = selected[-1] + 1
    else:
        selected = [0]
        start = 1

    height, width = points.shape[1:3]
    for target in range(start, len(ids)):
        if target - selected[-1] >= int(maximum_interval):
            selected.append(target)
            continue

        references = torch.tensor(selected, device=points.device, dtype=torch.long)
        if bool((references >= target).any()):
            raise RuntimeError("Eq. (4) references must precede the target frame")
        target_valid = valid_mask[target]
        pixel_threshold = float(projection_threshold) * target_valid.sum()
        relative = torch.linalg.inv(poses[references]) @ poses[target]
        target_points = points[target].reshape(-1, 3)
        reference_points = torch.einsum(
            "kij,mj->kmi", relative[:, :3, :3], target_points
        ) + relative[:, None, :3, 3]
        projected = torch.einsum(
            "kij,kmj->kmi", intrinsics[references], reference_points
        )
        coordinates = projected[..., :2] / projected[..., 2:3].clamp_min(1.0e-8)
        pixels = coordinates.nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0
        ).round().long()
        pixels[..., 0].clamp_(0, width - 1)
        pixels[..., 1].clamp_(0, height - 1)
        reference_rows = torch.arange(
            references.numel(), device=points.device
        )[:, None]
        projected_valid = valid_mask[references][
            reference_rows, pixels[..., 1], pixels[..., 0]
        ]
        covered = (
            target_valid.reshape(1, -1)
            & projected_valid
            & (confidence[target].reshape(1, -1) > float(confidence_threshold))
            & (reference_points[..., 2] > 0)
            & (coordinates[..., 0] >= 0)
            & (coordinates[..., 0] <= width - 1)
            & (coordinates[..., 1] >= 0)
            & (coordinates[..., 1] <= height - 1)
        )
        if covered.sum(dim=-1).max() < pixel_threshold:
            selected.append(target)

    return torch.tensor(selected, device=points.device, dtype=torch.long)


__all__ = [
    "KEYFRAME_CONFIDENCE_THRESHOLD",
    "KEYFRAME_MAX_INTERVAL",
    "KEYFRAME_PROJECTION_THRESHOLD",
    "select_keyframes_eq4_window",
]
