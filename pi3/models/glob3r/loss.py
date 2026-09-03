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
    """Build reference-row, target-column labels for corrected Eq. (31)."""

    batch, targets, height, width = ground_truth_warp.shape[:4]
    warp = _resize_warp(ground_truth_warp, (patch_height, patch_width))
    valid = _resize_scalar_map(positive, (patch_height, patch_width), "nearest").bool()
    target_x = (warp[:, :, 0] / max(width - 1, 1) * (patch_width - 1)).round().long()
    target_y = (warp[:, :, 1] / max(height - 1, 1) * (patch_height - 1)).round().long()
    inside = (
        (target_x >= 0) & (target_x < patch_width) & (target_y >= 0) & (target_y < patch_height)
    )
    # Each reference-grid row n is supervised by its projected target patch m_n*.
    labels = (target_y * patch_width + target_x).flatten(-2)
    valid = (valid & inside).flatten(-2)
    labels = labels.masked_fill(~valid, -1)
    return labels


def auxiliary_nll_loss(similarity_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Corrected Eq. (31): target label per reference-row cosine logits."""

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
    predicted_confidence_logits: torch.Tensor,
    target_confidence: torch.Tensor,
    training_mask: torch.Tensor,
) -> torch.Tensor:
    """Glob3R Eq. (34): stable masked BCE on FP32 confidence logits."""

    logits = predicted_confidence_logits.squeeze(2).float()
    if not training_mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[training_mask], target_confidence.float()[training_mask]
    )


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
        train_sky: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_warp = lambda_warp
        self.lambda_confidence = lambda_confidence
        self.depth_threshold = depth_threshold
        self.charbonnier_epsilon = charbonnier_epsilon
        self.charbonnier_alpha = charbonnier_alpha
        self.train_sky = train_sky
        if self.train_sky:
            self.prepare_segformer()

    def prepare_segformer(self):
        """
        Load the same frozen ADE20K SegFormer used by Pi3 confidence training.
        wget -O ckpts/segformer.b0.512x512.ade.160k.pth \
        https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth
        """

        from pi3.models.segformer.model import EncoderDecoder

        self.segformer = EncoderDecoder()
        checkpoint = torch.load(
            "ckpts/segformer.b0.512x512.ade.160k.pth",
            map_location=torch.device("cpu"),
            weights_only=False,
        )["state_dict"]
        self.segformer.load_state_dict(checkpoint)
        self.segformer = self.segformer.cuda().eval()
        self.segformer.requires_grad_(False)

    def predict_sky_mask(self, imgs):
        """Match Pi3's ADE20K class-2 sky mask construction."""

        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

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

        reference_sky = None
        if self.train_sky:
            images = torch.stack([view["img"] for view in batch], dim=1)
            reference_sky = self.predict_sky_mask(images[:, reference_index])
            reference_sky = reference_sky[:, None].expand(
                -1, len(target_indices), -1, -1
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
        confidence_logits_predictions = [
            predictions["coarse_match_confidence_logits"],
            *predictions.get("match_confidence_logits_stages", []),
        ]
        if len(warp_predictions) != len(confidence_logits_predictions):
            raise ValueError("warp and confidence-logit stages must have equal length")
        warp_terms = []
        confidence_terms = []
        sky_confidence_terms = []
        for predicted_warp, predicted_confidence_logits in zip(
            warp_predictions, confidence_logits_predictions
        ):
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
            confidence_terms.append(
                confidence_loss(predicted_confidence_logits, positive, mask)
            )
            if reference_sky is not None:
                sky = _resize_scalar_map(reference_sky, size, "nearest").bool()
                # Mirror Pi3 by adding explicit zero-confidence supervision only
                # where the geometry loss would otherwise ignore semantic sky.
                sky_confidence_terms.append(
                    confidence_loss(
                        predicted_confidence_logits,
                        torch.zeros_like(positive),
                        sky & ~mask,
                    )
                )

        warp = torch.stack(warp_terms).mean()
        geometry_confidence = torch.stack(confidence_terms).mean()
        sky_confidence = (
            torch.stack(sky_confidence_terms).mean()
            if sky_confidence_terms
            else geometry_confidence.new_zeros(())
        )
        confidence = geometry_confidence + sky_confidence
        # Glob3R Eq. (35), also summarized by main-paper Eq. (3):
        # L = sum_b(lambda_NLL L_NLL + lambda_warp L_warp + lambda_conf L_conf).
        total = self.lambda_nll * nll + self.lambda_warp * warp + self.lambda_confidence * confidence
        return total, {
            "loss_nll": nll.detach(),
            "loss_warp": warp.detach(),
            "loss_match_confidence": confidence.detach(),
            "loss_match_confidence_geometry": geometry_confidence.detach(),
            "loss_match_confidence_sky": sky_confidence.detach(),
        }
