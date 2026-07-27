"""Consolidated Glob3R matching supervision and Appendix B losses."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import build_ground_truth_warp, relative_camera_transform


def _resize_scalar_map(value: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    """Resize ``[B,T,H,W]`` confidence or mask maps to the prediction resolution.

    These are scalar labels rather than pixel coordinates, so camera intrinsics are irrelevant.
    """

    batch, targets = value.shape[:2]
    flat = value.reshape(batch * targets, 1, *value.shape[-2:]).float()
    resized = F.interpolate(flat, size=size, mode=mode)
    return resized.reshape(batch, targets, *size)


def _resize_warp(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize ``[B,T,H,W,2]`` GT warp grids to the prediction resolution.

    Nearest resizing keeps coordinates aligned with validity labels. Values stay
    in full-image pixels, so resize/crop-adjusted intrinsics need no scaling here.
    """

    batch, targets = value.shape[:2]
    channels_first = value.permute(0, 1, 4, 2, 3).reshape(batch * targets, 2, *value.shape[2:4])
    resized = F.interpolate(channels_first, size=size, mode="nearest")
    return resized.reshape(batch, targets, 2, *size)


def patch_nll_targets(
    ground_truth_warp: torch.Tensor,
    positive: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Invert reference-to-target patch matches into target-row labels for Eq. (31)."""

    batch, targets, height, width = ground_truth_warp.shape[:4]
    warp = _resize_warp(ground_truth_warp, (patch_height, patch_width))
    valid = _resize_scalar_map(positive, (patch_height, patch_width), "nearest").bool()
    labels = torch.full(
        (batch, targets, patch_height * patch_width),
        -1,
        device=ground_truth_warp.device,
        dtype=torch.long,
    )
    target_x = (warp[:, :, 0] / max(width - 1, 1) * (patch_width - 1)).round().long()
    target_y = (warp[:, :, 1] / max(height - 1, 1) * (patch_height - 1)).round().long()
    inside = (
        (target_x >= 0) & (target_x < patch_width) & (target_y >= 0) & (target_y < patch_height)
    )
    reference_index = torch.arange(patch_height * patch_width, device=labels.device)
    # Collisions are deterministic; the last valid reference patch becomes n*_m.
    for batch_index in range(batch):
        for target_index in range(targets):
            selected = (valid[batch_index, target_index] & inside[batch_index, target_index]).flatten()
            target_flat = (
                target_y[batch_index, target_index].flatten() * patch_width
                + target_x[batch_index, target_index].flatten()
            )
            labels[batch_index, target_index, target_flat[selected]] = reference_index[selected]
    return labels


def auxiliary_nll_loss(similarity_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Glob3R Eq. (31) using RoMaV2's row-wise cosine logits."""

    valid = labels >= 0
    if not valid.any():
        return similarity_logits.sum() * 0.0
    # The literal Eq. (31) applies Softmax to S=exp(cos/tau), which would
    # exponentiate twice. RoMaV2 [15] instead applies cross entropy directly
    # to cos/tau logits, consistently with the probabilities used in Eq. (16).
    return F.cross_entropy(similarity_logits[valid].float(), labels[valid])


def generalized_charbonnier_loss(
    predicted_warp: torch.Tensor,
    ground_truth_warp: torch.Tensor,
    valid: torch.Tensor,
    epsilon: float = 1e-3,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Glob3R Eqs. (32)-(33): robust dense-warp regression."""

    # Glob3R Eq. (32): r^(a->b)(x) = W_hat^(a->b)(x) - W*^(a->b)(x).
    residual = predicted_warp - ground_truth_warp
    # Glob3R Eq. (33): mean ((||r||_2^2 + epsilon^2) ** (alpha/2)) over Omega.
    penalty = (residual.square().sum(dim=2) + epsilon**2).pow(alpha / 2)
    if not valid.any():
        return predicted_warp.sum() * 0.0
    return penalty[valid].mean()


def confidence_loss(
    predicted_confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    training_mask: torch.Tensor,
) -> torch.Tensor:
    """Glob3R Eq. (34): masked binary cross entropy for match confidence."""

    prediction = predicted_confidence.squeeze(2).clamp(1e-6, 1 - 1e-6)
    if not training_mask.any():
        return predicted_confidence.sum() * 0.0
    return F.binary_cross_entropy(prediction[training_mask], target_confidence.float()[training_mask])


class Glob3RMatchingLoss(nn.Module):
    """Complete Glob3R coarse/refinement training objective.

    The class accepts the repository's existing ``list[view_dict]`` batch
    format so no dataset implementation has to be modified.
    """

    def __init__(
        self,
        lambda_nll: float = 1.0,
        lambda_warp: float = 1.0,
        lambda_confidence: float = 1.0,
        depth_threshold: float = 0.05,
        charbonnier_epsilon: float = 1e-3,
        charbonnier_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_warp = lambda_warp
        self.lambda_confidence = lambda_confidence
        self.depth_threshold = depth_threshold
        self.charbonnier_epsilon = charbonnier_epsilon
        self.charbonnier_alpha = charbonnier_alpha

    @staticmethod
    def _stack_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
        depths = torch.stack([view["depthmap"] for view in batch], dim=1)
        intrinsics = torch.stack([view["camera_intrinsics"] for view in batch], dim=1)
        world_from_camera = torch.stack([view["camera_pose"] for view in batch], dim=1)
        return depths, intrinsics, world_from_camera

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Sequence[Dict[str, torch.Tensor]]):
        depths, intrinsics, world_from_camera = self._stack_batch(batch)
        reference_index = int(predictions.get("reference_index", 0))
        target_indices = predictions["target_indices"]
        target_from_reference = relative_camera_transform(world_from_camera, reference_index)
        supervision = build_ground_truth_warp(
            depths[:, reference_index],
            depths[:, target_indices],
            intrinsics[:, reference_index],
            intrinsics[:, target_indices],
            target_from_reference,
            self.depth_threshold,
        )

        similarity_logits = predictions["match_similarity"]
        image_h, image_w = depths.shape[-2:]
        patch_h = image_h // 14
        patch_w = image_w // 14
        labels = patch_nll_targets(supervision.warp, supervision.confidence, patch_h, patch_w)
        if labels.shape[-1] != similarity_logits.shape[-2]:
            raise ValueError("similarity matrix does not match the Pi3X patch grid")
        nll = auxiliary_nll_loss(similarity_logits, labels)

        warp_predictions = [predictions["coarse_warp"], *predictions.get("warp_stages", [])]
        confidence_predictions = [
            predictions["coarse_match_confidence"],
            *predictions.get("match_confidence_stages", []),
        ]
        warp_terms = []
        confidence_terms = []
        for predicted_warp, predicted_confidence in zip(warp_predictions, confidence_predictions):
            size = predicted_warp.shape[-2:]
            ground_truth_warp = _resize_warp(supervision.warp, size)
            positive = _resize_scalar_map(supervision.confidence, size, "nearest").bool()
            mask = _resize_scalar_map(supervision.mask, size, "nearest").bool()
            # Appendix B: warp regression is evaluated only where y=1 and m=1.
            warp_terms.append(
                generalized_charbonnier_loss(
                    predicted_warp,
                    ground_truth_warp,
                    positive & mask,
                    self.charbonnier_epsilon,
                    self.charbonnier_alpha,
                )
            )
            confidence_terms.append(confidence_loss(predicted_confidence, positive, mask))

        warp = torch.stack(warp_terms).mean()
        confidence = torch.stack(confidence_terms).mean()
        # Glob3R Eq. (35), also summarized by main-paper Eq. (3):
        # L = sum_b(lambda_NLL L_NLL + lambda_warp L_warp + lambda_conf L_conf).
        total = self.lambda_nll * nll + self.lambda_warp * warp + self.lambda_confidence * confidence
        return total, {
            "loss_nll": nll.detach(),
            "loss_warp": warp.detach(),
            "loss_match_confidence": confidence.detach(),
        }
