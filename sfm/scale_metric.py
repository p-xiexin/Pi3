"""Estimate one Pi3 chunk scale from shared high-confidence depth pixels."""

import torch


SCALE_SAMPLE_POINTS = 64
MIN_SCALE_POINTS = 8


def estimate_chunk_scale(
    current_depth,
    reference_depth,
    reference_scale,
    current_confidence,
    reference_confidence,
    sample_points=SCALE_SAMPLE_POINTS,
    minimum_points=MIN_SCALE_POINTS,
):
    """Average sampled depth-ratio scales over all usable overlap frames."""
    current_depth = torch.as_tensor(current_depth)
    reference_depth = torch.as_tensor(
        reference_depth, device=current_depth.device, dtype=current_depth.dtype
    )
    current_confidence = torch.as_tensor(
        current_confidence, device=current_depth.device, dtype=current_depth.dtype
    )
    reference_confidence = torch.as_tensor(
        reference_confidence,
        device=current_depth.device,
        dtype=current_depth.dtype,
    )
    reference_scale = torch.as_tensor(
        reference_scale, device=current_depth.device, dtype=current_depth.dtype
    )
    if current_depth.ndim != 3 or reference_depth.shape != current_depth.shape:
        raise ValueError("overlap depths must have equal shape [frames,height,width]")
    if (
        current_confidence.shape != current_depth.shape
        or reference_confidence.shape != current_depth.shape
    ):
        raise ValueError("overlap confidence maps must match the depth maps")
    if reference_scale.shape != (current_depth.shape[0],):
        raise ValueError("reference scales must contain one value per overlap frame")
    if not current_depth.shape[0]:
        raise RuntimeError("chunk scale requires at least one overlap frame")

    frame_scales = []
    for local in range(current_depth.shape[0]):
        valid = (
            torch.isfinite(current_depth[local])
            & torch.isfinite(reference_depth[local])
            & torch.isfinite(current_confidence[local])
            & torch.isfinite(reference_confidence[local])
            & (current_depth[local] > 0)
            & (reference_depth[local] > 0)
            & (current_confidence[local] > 0)
            & (reference_confidence[local] > 0)
        )
        valid_ids = torch.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        if valid_ids.numel() < int(minimum_points):
            continue
        count = min(int(sample_points), valid_ids.numel())
        confidence = (
            current_confidence[local].flatten()[valid_ids]
            * reference_confidence[local].flatten()[valid_ids]
        )
        selected = valid_ids[torch.topk(confidence, count).indices]
        frame_scales.append(
            (
                reference_scale[local]
                * reference_depth[local].flatten()[selected]
                / current_depth[local].flatten()[selected]
            ).mean()
        )
    if not frame_scales:
        raise RuntimeError(
            "chunk scale has no overlap frame with enough high-confidence Pi3 depth pixels"
        )
    scale = torch.stack(frame_scales).mean()
    if not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise RuntimeError("chunk scale estimation produced an invalid scale")
    return scale


__all__ = ["MIN_SCALE_POINTS", "SCALE_SAMPLE_POINTS", "estimate_chunk_scale"]
