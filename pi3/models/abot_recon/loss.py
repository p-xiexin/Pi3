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

from pi3.models.loss import PointLoss
from pi3.utils.geometry import homogenize_points, se3_inverse

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


_ABOT_TO_PI3_DATASET = {
    "TartanAirABotRecon": "TarTanAir",
    "ScanNetABotRecon": "ScanNet",
    "BlendedMVSABotRecon": "BlendedMVS",
    "KITTIABotRecon": "KITTI",
    "WaymoABotRecon": "Waymo",
}


def _pi3_dataset_names(names, batch_size):
    """Translate adapter labels to the names used by Pi3 loss routing."""

    if names is None:
        values = ["Unknown"] * int(batch_size)
    elif isinstance(names, str):
        values = [names] * int(batch_size)
    elif torch.is_tensor(names):
        values = names.detach().cpu().tolist()
    else:
        values = list(names)
    if len(values) != int(batch_size):
        raise ValueError(
            f"dataset_names has {len(values)} entries for batch size {batch_size}"
        )
    return [_ABOT_TO_PI3_DATASET.get(str(name), str(name)) for name in values]


def _stack_training_views(views):
    """Stack Pi3 view dictionaries without changing their dataset order."""

    if not isinstance(views, (list, tuple)) or not views:
        raise TypeError("ABotReconLoss expects a non-empty list of Pi3 views")
    required = ("pts3d", "valid_mask", "camera_pose")
    for frame_index, view in enumerate(views):
        missing = [key for key in required if key not in view]
        if missing:
            raise KeyError(
                f"ABot-Recon view {frame_index} is missing " + ", ".join(missing)
            )

    points = torch.stack([view["pts3d"] for view in views], dim=1)
    masks = torch.stack([view["valid_mask"] for view in views], dim=1)
    poses = torch.stack([view["camera_pose"] for view in views], dim=1)
    if points.ndim != 5 or points.shape[-1] != 3:
        raise ValueError("pts3d must stack to [B,N,H,W,3]")
    if masks.shape != points.shape[:-1]:
        raise ValueError("valid_mask must stack to [B,N,H,W]")
    if poses.ndim != 4 or poses.shape[:2] != points.shape[:2] or poses.shape[-2:] != (4, 4):
        raise ValueError("camera_pose must stack to [B,N,4,4]")
    return points, masks, poses, views[0].get("dataset")


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
        max_pair_gap=11,
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
        # Reuse Pi3's implementation directly so inverse-depth weighting,
        # robust scale alignment, depth-edge rejection, four-triangle normal
        # construction and dataset-quality routing stay exactly synchronized.
        self.pi3_point_loss = PointLoss(
            local_align_res=self.local_align_res,
            train_conf=False,
        )

    def prepare_targets(self, views):
        """Express GT geometry in the first-camera gauge used by Pi3 supervision.

        Local point maps remain in each current camera frame, matching ``P_i``
        in Eq. (1).  Camera-to-world poses are normalized to frame zero so the
        relative transforms used by Eq. (14) are gauge independent.
        """

        points, masks, poses, dataset_names = _stack_training_views(views)
        points = points.float()
        masks = masks.bool() & torch.isfinite(points).all(-1)
        poses = poses.float()
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
            "dataset_names": _pi3_dataset_names(
                dataset_names, points.shape[0]
            ),
        }

    def _point_and_normal_loss(self, prediction, target):
        """Compute ``L_pts`` and ``L_normal`` appearing in Eq. (15).

        The report inherits these terms from Pi3.  Delegate their computation
        to ``PointLoss`` instead of maintaining a numerically different copy.
        """

        pred_points = prediction["local_points"].float()
        gt_points = target["local_points"]
        mask = target["valid_masks"]
        _, pi3_details, scale = self.pi3_point_loss(
            {"local_points": pred_points}, target
        )
        point_loss = pi3_details["local_pts_loss"]
        normal_loss = pi3_details["normal_loss"]

        # Stage III confidence labels use the same aligned, inverse-depth
        # weighted point residual produced inside Pi3's PointLoss.
        depth_weight = gt_points[..., 2]
        valid_weight = mask.to(depth_weight.dtype)
        weighted_depth_mean = (
            (depth_weight * valid_weight).mean(dim=(-2, -1), keepdim=True)
            / valid_weight.mean(dim=(-2, -1), keepdim=True).add(1e-7)
        )
        depth_weight = depth_weight.clamp_min(0.1 * weighted_depth_mean)
        depth_weight = 1.0 / (depth_weight + 1e-6)
        aligned = pred_points * scale[:, None, None, None, None]
        point_error = (aligned - gt_points).abs() * depth_weight[..., None]
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
