"""Adjacent-frame GT and coarse losses for the streaming Glob3R head."""

import torch
from torch import nn
import torch.nn.functional as F

from pi3.models.glob3r.geometry import (
    build_ground_truth_warp, normalize_warp, sample_map_at_pixels, WarpSupervision,
)
from pi3.models.glob3r.loss import _resize_scalar_map, _resize_warp, patch_nll_targets


def mean_over_pairs(value, mask):
    """Pixel mean, then nonempty pair mean, then nonempty sequence mean."""
    value, mask = value.flatten(2), mask.flatten(2)
    counts = mask.sum(-1)
    pair_mean = value.masked_fill(~mask, 0).sum(-1) / counts.clamp_min(1)
    nonempty = counts > 0
    sequence_mean = pair_mean.sum(-1) / nonempty.sum(-1).clamp_min(1)
    return sequence_mean.sum() / nonempty.any(-1).sum().clamp_min(1)


class AdjacentMatchingLoss(nn.Module):
    """Patch CE, normalized-coordinate Charbonnier, and confidence BCE.

    Input is the repository list-of-views format, with camera-to-world poses.
    Optional image_valid_mask and static_mask exclude padding/dynamic regions.
    valid_mask denotes reliable GT depth, so target holes remain unknown.
    """

    def __init__(self, depth_threshold=0.05, epsilon=1e-3, alpha=0.5):
        super().__init__()
        if depth_threshold <= 0 or epsilon <= 0 or not 0 < alpha <= 2:
            raise ValueError("invalid matching loss parameters")
        self.depth_threshold, self.epsilon, self.alpha = depth_threshold, epsilon, alpha

    @torch.no_grad()
    def supervision(self, predictions, batch):
        refs = predictions["reference_indices"]
        targets = predictions["target_indices"]
        if not torch.equal(refs + 1, targets):
            raise ValueError("supervision requires reference=t-1 and target=t")
        depths = torch.stack([v["depthmap"] for v in batch], 1).float()
        intrinsics = torch.stack([v["camera_intrinsics"] for v in batch], 1).float()
        poses = torch.stack([v["camera_pose"] for v in batch], 1).float()
        b, _, height, width = depths.shape
        if (height, width) != predictions["image_size"]:
            raise ValueError("GT depth and matching image sizes differ")
        pairs = targets.numel()
        depth_valid = torch.isfinite(depths) & (depths > 0)
        allowed = torch.ones_like(depth_valid)
        for i, view in enumerate(batch):
            if "valid_mask" in view:
                depth_valid[:, i] &= view["valid_mask"].bool()
            for key in ("image_valid_mask", "static_mask"):
                if key in view:
                    allowed[:, i] &= view[key].bool()
        depths = torch.where(depth_valid, depths, 0)
        finite_K = torch.isfinite(intrinsics).all(dim=(-2, -1))
        finite_T = torch.isfinite(poses).all(dim=(-2, -1))
        safe_K = torch.where(finite_K[..., None, None], intrinsics, torch.eye(3, device=depths.device))
        safe_T = torch.where(finite_T[..., None, None], poses, torch.eye(4, device=depths.device))
        _, info_K = torch.linalg.inv_ex(safe_K)
        inverse_T, info_T = torch.linalg.inv_ex(safe_T)
        valid_camera = finite_K & finite_T & (info_K == 0) & (info_T == 0)
        valid_camera &= (safe_K[..., 0, 0] > 0) & (safe_K[..., 1, 1] > 0)
        safe_K = torch.where(valid_camera[..., None, None], safe_K, torch.eye(3, device=depths.device))
        safe_T = torch.where(valid_camera[..., None, None], safe_T, torch.eye(4, device=depths.device))
        inverse_T = torch.where(valid_camera[..., None, None], inverse_T, torch.eye(4, device=depths.device))
        transform = inverse_T[:, targets] @ safe_T[:, refs]
        result = build_ground_truth_warp(
            depths[:, refs].flatten(0, 1), depths[:, targets].flatten(0, 1)[:, None],
            safe_K[:, refs].flatten(0, 1), safe_K[:, targets].flatten(0, 1)[:, None],
            transform.flatten(0, 1)[:, None], self.depth_threshold)
        warp = result.warp[:, 0]
        finite = torch.isfinite(warp).all(-1) & torch.isfinite(result.projected_depth[:, 0])
        safe_warp = torch.where(finite[..., None], warp, 0)
        inside = ((safe_warp[..., 0] >= 0) & (safe_warp[..., 0] <= width - 1)
                  & (safe_warp[..., 1] >= 0) & (safe_warp[..., 1] <= height - 1))
        def target_support(value):
            return sample_map_at_pixels(value[:, targets].flatten(0, 1)[:, None].float(), safe_warp)[:, 0] >= 1 - 1e-5
        known_depth = target_support(depth_valid)
        target_allowed = target_support(allowed)
        source_valid = (depth_valid[:, refs] & allowed[:, refs]).flatten(0, 1)
        pair_valid = (valid_camera[:, refs] & valid_camera[:, targets]).flatten()
        mask = (source_valid & pair_valid[:, None, None] & finite
                & (result.projected_depth[:, 0] > 0)
                & ((inside & known_depth & target_allowed) | ~inside))
        positive = result.confidence[:, 0] & mask
        return WarpSupervision(
            safe_warp.reshape(b, pairs, height, width, 2),
            positive.reshape(b, pairs, height, width), mask.reshape(b, pairs, height, width),
            result.projected_depth.reshape(b, pairs, height, width))

    def forward(self, predictions, batch):
        # Geometry and reductions are FP32 even under the trainer's autocast.
        device_type = predictions["coarse_warp"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            gt = self.supervision(predictions, batch)
            height, width = predictions["image_size"]
            patch = predictions["patch_size"]
            labels = patch_nll_targets(gt.warp, gt.confidence, height // patch, width // patch)
            logits = predictions["similarity"].float()
            ce = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="none", ignore_index=-1)
            patch_loss = mean_over_pairs(ce.reshape_as(labels), labels >= 0)
            predicted = predictions["normalized_coarse_warp"].float()
            size = predicted.shape[-2:]
            target_pixels = _resize_warp(gt.warp, size)
            target = normalize_warp(target_pixels, (height, width))
            positive = _resize_scalar_map(gt.confidence, size, "nearest").bool()
            mask = _resize_scalar_map(gt.mask, size, "nearest").bool()
            penalty = ((predicted - target).square().sum(2) + self.epsilon ** 2).pow(self.alpha / 2)
            warp_loss = mean_over_pairs(penalty, positive)
            confidence_logits = predictions["coarse_confidence_logits"].float().squeeze(2)
            bce = F.binary_cross_entropy_with_logits(confidence_logits, positive.float(), reduction="none")
            conf_loss = mean_over_pairs(bce, mask)
            loss = patch_loss + warp_loss + conf_loss
            error = (predictions["coarse_warp"].float() - target_pixels).square().sum(2).sqrt()
            return loss, {
                "loss_nll": patch_loss.detach(), "loss_warp": warp_loss.detach(),
                "loss_match_confidence": conf_loss.detach(),
                "epe": mean_over_pairs(error, positive).detach(),
                "positive_fraction": positive.float().mean(),
                "supervised_fraction": mask.float().mean(),
            }
