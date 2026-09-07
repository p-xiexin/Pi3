"""Composition-aware ABot-Recon training losses from Sec. 3.4.

The report gives the supervised pair set, gap weighting, pose objective, and
five-term total objective in Eqs. (12)-(15).  It refers the local point, normal,
and confidence objectives to Pi3 and describes ``L_smooth`` verbally.  Comments
below distinguish those stated equations from the explicit choices required to
turn the public description into executable training code.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi3.utils.geometry import homogenize_points, se3_inverse
from pi3.utils.alignment import align_points_scale

from .data import prepare_abot_batch, validate_training_batch


def _masked_mean(value, mask, eps=1e-8):
    """Average only valid geometric elements while preserving a zero-loss graph."""

    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(eps)


def _rotation_angle(prediction, target):
    """SO(3) geodesic angle used as the Eq. (14) rotation error."""

    residual = prediction.transpose(-2, -1) @ target
    skew = torch.stack((
        residual[..., 2, 1] - residual[..., 1, 2],
        residual[..., 0, 2] - residual[..., 2, 0],
        residual[..., 1, 0] - residual[..., 0, 1]), -1)
    sine = 0.5 * skew.norm(dim=-1)
    cosine = 0.5 * (residual.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


class ABotReconLoss(nn.Module):
    """Executable form of the ABot-Recon objective in Eqs. (12)-(15).

    The report specifies the five loss families but omits its scalar weights,
    gamma, confidence labels, and exact smoothness formula.  Those quantities
    are constructor arguments here.  Their defaults are training assumptions,
    not values claimed by the paper.
    """
    def __init__(
        self,
        point_weight=1.0,
        normal_weight=1.0,
        pose_weight=0.1,
        smooth_weight=0.1,
        confidence_weight=0.05,
        translation_weight=100.0,
        rotation_weight=1.0,
        max_pair_gap=12,
        gap_gamma=0.5,
        smooth_temporal_weight=1.0,
        confidence_error_threshold=0.02,
        huber_delta=0.1,
        local_align_res=4096,
    ):
        super().__init__()
        self.point_weight = float(point_weight)
        self.normal_weight = float(normal_weight)
        self.pose_weight = float(pose_weight)
        self.smooth_weight = float(smooth_weight)
        self.confidence_weight = float(confidence_weight)
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.max_pair_gap = int(max_pair_gap)
        self.gap_gamma = float(gap_gamma)
        self.smooth_temporal_weight = float(smooth_temporal_weight)
        self.confidence_error_threshold = float(confidence_error_threshold)
        self.huber_delta = float(huber_delta)
        self.local_align_res = int(local_align_res)

    def prepare_targets(self, batch):
        """Express GT geometry in the first-camera gauge used by Pi3 supervision.

        Local point maps remain in each current camera frame, matching ``P_i``
        in Eq. (1).  Camera-to-world poses are normalized to frame zero so the
        relative transforms used by Eq. (14) are gauge independent.
        """

        if isinstance(batch, (list, tuple)):
            batch = prepare_abot_batch(batch)
        validate_training_batch(batch)
        points = batch["world_points"].float()
        masks = batch["valid_masks"].bool() & torch.isfinite(points).all(-1)
        poses = batch["camera_poses"].float()
        # Remove the arbitrary dataset world frame.  This transformation leaves
        # all relative poses in Eqs. (12)-(14) unchanged.
        first_w2c = se3_inverse(poses[:, 0])
        global_points = torch.einsum(
            "bij,bnhwj->bnhwi", first_w2c, homogenize_points(points))[..., :3]
        poses = first_w2c[:, None] @ poses
        # Pi3-style scene-scale normalization gives point and translation losses
        # comparable magnitude across datasets with different metric units.
        distance = global_points.norm(dim=-1)
        scale = (distance * masks).sum((1, 2, 3)) / masks.sum((1, 2, 3)).clamp_min(1)
        scale = torch.where(torch.isfinite(scale) & (scale > 1e-6), scale,
                            torch.ones_like(scale))
        global_points = global_points / scale[:, None, None, None, None]
        poses = poses.clone()
        poses[..., :3, 3] /= scale[:, None, None]
        # Eq. (1) supervises P_i in the current-camera coordinate system.
        local_points = torch.einsum(
            "bnij,bnhwj->bnhwi", se3_inverse(poses),
            homogenize_points(global_points))[..., :3]
        return {
            "local_points": local_points,
            "global_points": global_points,
            "valid_masks": masks,
            "camera_poses": poses,
        }

    def _sample_valid(self, value, mask):
        """Deterministically reduce valid pixels for Pi3 robust scale alignment."""

        sampled = []
        for batch_index in range(value.shape[0]):
            valid = value[batch_index][mask[batch_index]]
            if valid.numel() == 0:
                valid = value.new_ones((1, value.shape[-1]))
            valid = valid.transpose(0, 1)[None]
            valid = F.interpolate(
                valid, size=self.local_align_res, mode="nearest")
            sampled.append(valid[0].transpose(0, 1))
        return torch.stack(sampled)

    def _point_scale(self, predicted, target, mask, depth_weight):
        """Solve the scale ambiguity of the local point maps without gradients."""

        with torch.no_grad():
            pred_sample = self._sample_valid(predicted, mask).contiguous()
            target_sample = self._sample_valid(target, mask).contiguous()
            weight_sample = self._sample_valid(depth_weight[..., None], mask)[..., 0]
            scale = align_points_scale(
                pred_sample, target_sample, weight_sample.contiguous())
            return scale.abs().clamp_min(1e-4)

    def _point_and_normal_loss(self, prediction, target):
        """Compute ``L_pts`` and ``L_normal`` appearing in Eq. (15).

        The report inherits these terms from Pi3.  This implementation therefore
        uses Pi3's robust scalar alignment, inverse-depth point weighting, and
        finite-difference surface normals.
        """

        pred_points = prediction["local_points"].float()
        gt_points = target["local_points"]
        mask = target["valid_masks"]
        depth_weight = 1.0 / gt_points[..., 2].abs().clamp_min(0.1)
        scale = self._point_scale(
            pred_points, gt_points, mask, depth_weight)
        aligned = pred_points * scale[:, None, None, None, None]
        point_error = (aligned - gt_points).abs() * depth_weight[..., None]
        point_loss = _masked_mean(point_error, mask)

        # Local horizontal and vertical tangents define the normal supervision
        # used by the Eq. (15) L_normal term.
        pred_dx = aligned[..., :, 1:, :] - aligned[..., :, :-1, :]
        pred_dy = aligned[..., 1:, :, :] - aligned[..., :-1, :, :]
        gt_dx = gt_points[..., :, 1:, :] - gt_points[..., :, :-1, :]
        gt_dy = gt_points[..., 1:, :, :] - gt_points[..., :-1, :, :]
        pred_normal = torch.cross(pred_dx[..., :-1, :, :], pred_dy[..., :, :-1, :], -1)
        gt_normal = torch.cross(gt_dx[..., :-1, :, :], gt_dy[..., :, :-1, :], -1)
        normal_mask = (mask[..., :-1, :-1] & mask[..., :-1, 1:] &
                       mask[..., 1:, :-1])
        normal_error = 1.0 - F.cosine_similarity(pred_normal, gt_normal, dim=-1, eps=1e-6)
        normal_loss = _masked_mean(normal_error, normal_mask)
        return point_loss, normal_loss, scale, aligned, point_error.mean(-1)

    def _pose_loss(self, prediction, target, point_scale):
        """Compute the multi-gap composition-aware pose loss in Eqs. (12)-(14)."""

        pred = prediction["camera_poses"].float().clone()
        pred[..., :3, 3] *= point_scale[:, None, None]
        gt = target["camera_poses"]
        frames = pred.shape[1]
        # Eq. (12): P = {(i,j) | i < j and j-i <= K-1}.  ``max_pair_gap``
        # represents K-1 and is clipped for shorter training sequences.
        maximum = min(self.max_pair_gap, frames - 1)
        if maximum < 1:
            zero = pred.sum() * 0.0
            return zero, zero, zero
        pair_count = sum(frames - gap for gap in range(1, maximum + 1))
        # Eq. (13): normalize gap^gamma by its mean over every pair in P.
        # There are N-gap pairs at a given temporal gap.
        alpha_denominator = sum(
            (frames - gap) * gap ** self.gap_gamma
            for gap in range(1, maximum + 1)) / pair_count
        translation_terms, rotation_terms, alpha_terms = [], [], []
        for offset in range(1, maximum + 1):
            # Eq. (5) is already represented by the composed c2w trajectory.
            # inv(T_0<-i) T_0<-j recovers each T_(i<-j) requested by Eq. (12).
            pred_relative = se3_inverse(pred[:, :-offset]) @ pred[:, offset:]
            gt_relative = se3_inverse(gt[:, :-offset]) @ gt[:, offset:]
            translation_terms.append(F.huber_loss(
                pred_relative[..., :3, 3], gt_relative[..., :3, 3],
                delta=self.huber_delta, reduction="none").mean(-1))
            rotation_terms.append(_rotation_angle(
                pred_relative[..., :3, :3], gt_relative[..., :3, :3]))
            alpha_terms.append(torch.full_like(
                rotation_terms[-1],
                offset ** self.gap_gamma / alpha_denominator))
        translation = torch.cat(translation_terms, 1)
        rotation = torch.cat(rotation_terms, 1)
        alpha = torch.cat(alpha_terms, 1)
        weighted_rotation = alpha * rotation
        # Eq. (14): alpha_ij multiplies only the rotation term in the printed
        # equation; translation retains a uniform pair weight.
        loss = (self.translation_weight * translation +
                self.rotation_weight * weighted_rotation).mean()
        return loss, translation.mean(), weighted_rotation.mean()

    def _smooth_loss(self, prediction):
        """Instantiate the report's verbally specified ``L_smooth`` term.

        The paper states that magnitude and temporal variation are penalized but
        does not print the formula.  We use mean squared residual magnitude plus
        a configurable mean squared first difference.  This is an explicit
        reconstruction choice rather than a recovered unpublished equation.
        """

        residual = prediction.get("rotation_residual")
        if residual is None or residual.numel() == 0:
            return prediction["local_points"].sum() * 0.0
        magnitude = residual.float().square().mean()
        variation = (residual[:, 1:] - residual[:, :-1]).float().square().mean()
        if residual.shape[1] < 2:
            variation = magnitude * 0.0
        return magnitude + self.smooth_temporal_weight * variation

    def _confidence_loss(self, prediction, target, point_error):
        """Instantiate the ``L_conf`` term referenced by Eq. (15).

        Public material does not expose confidence labels.  The target below is
        a detached threshold on aligned point error, following the reliability
        semantics of Pi3 while avoiding a separate semantic-segmentation model.
        """

        logits = prediction.get("conf")
        if logits is None:
            return point_error.sum() * 0.0
        labels = (point_error.detach() < self.confidence_error_threshold).float()
        valid = target["valid_masks"]
        if not valid.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits[..., 0][valid].float(), labels[valid], reduction="mean")

    def forward(self, prediction: Dict[str, torch.Tensor], batch):
        """Combine the five terms exactly in the structure of report Eq. (15)."""

        target = self.prepare_targets(batch)
        point, normal, scale, _, point_error = self._point_and_normal_loss(
            prediction, target)
        pose, translation, rotation = self._pose_loss(prediction, target, scale)
        smooth = self._smooth_loss(prediction)
        confidence = self._confidence_loss(prediction, target, point_error)
        # Eq. (15): L = lambda_pose L_pose + lambda_smooth L_smooth
        #              + lambda_pts L_pts + lambda_normal L_normal
        #              + lambda_conf L_conf.
        total = (self.point_weight * point + self.normal_weight * normal +
                 self.pose_weight * pose + self.smooth_weight * smooth +
                 self.confidence_weight * confidence)
        details = {
            "local_pts_loss": point,
            "normal_loss": normal,
            "pose_loss": pose,
            "trans_loss": translation,
            "rot_loss": rotation,
            "smooth_loss": smooth,
            "confidence_loss": confidence,
            "point_scale": scale.mean(),
        }
        return total, details
