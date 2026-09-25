"""ABot-Recon training losses described in Sec. 3.4 of the report."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi3.models.loss import PointLoss
from pi3.utils.geometry import homogenize_points, se3_inverse


__NO_NORMAL_DATASETS__ = [
    "KITTIABotRecon",
    "KITTIPi3X",
    "WaymoABotRecon",
    "WaymoPi3X",
    "NuScenesPi3X",
    "ARKitScenesPi3X",
    "Unknown",
]

__NO_POINT_DATASETS__ = ["ARKitScenesPi3X"]


def _rotation_angle(pred, gt):
    residual = pred.transpose(-2, -1) @ gt
    skew = torch.stack((
        residual[..., 2, 1] - residual[..., 1, 2],
        residual[..., 0, 2] - residual[..., 2, 0],
        residual[..., 1, 0] - residual[..., 0, 1],
    ), dim=-1)
    sine = 0.5 * skew.norm(dim=-1)
    cosine = 0.5 * (residual.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


def _stack_training_views(views):
    points = torch.stack([view["pts3d"] for view in views], dim=1)
    masks = torch.stack([view["valid_mask"] for view in views], dim=1)
    poses = torch.stack([view["camera_pose"] for view in views], dim=1)
    return points, masks, poses, views[0].get("dataset")


class ABotPointLoss(nn.Module):
    """Pi3 point, normal and point-confidence supervision."""

    def __init__(self, local_align_res=4096, confidence_error_threshold=0.02):
        super().__init__()
        self.pi3_loss = PointLoss(
            local_align_res=local_align_res,
            train_conf=False,
        )
        self.confidence_error_threshold = confidence_error_threshold

    def forward(self, pred, gt):
        pred_points = pred["local_points"].float()
        gt_points = gt["local_points"]
        valid_masks = gt["valid_masks"]
        dataset_names = gt["dataset_names"]

        point_batch = [
            i for i, name in enumerate(dataset_names)
            if name not in __NO_POINT_DATASETS__
        ]
        normal_batch = [
            i for i, name in enumerate(dataset_names)
            if name not in __NO_NORMAL_DATASETS__
        ]

        zero = pred_points.sum() * 0.0
        scale = pred_points.new_ones(pred_points.shape[0])
        point_loss = zero

        if point_batch:
            point_gt = {
                "local_points": gt_points[point_batch],
                "valid_masks": valid_masks[point_batch],
                "dataset_names": ["Unknown"] * len(point_batch),
            }
            _, details, point_scale = self.pi3_loss(
                {"local_points": pred_points[point_batch]}, point_gt
            )
            point_loss = details["local_pts_loss"]
            scale[point_batch] = point_scale

        aligned_points = pred_points * scale[:, None, None, None, None]
        normal_loss = zero
        if normal_batch:
            normal_loss = self.pi3_loss.noraml_loss(
                aligned_points[normal_batch],
                gt_points[normal_batch],
                valid_masks[normal_batch],
            )

        logits = pred.get("conf")
        if logits is None:
            confidence_loss = zero
        elif not point_batch:
            confidence_loss = logits.sum() * 0.0
        else:
            points = gt_points[point_batch]
            masks = valid_masks[point_batch]
            depth_weight = points[..., 2]
            depth_weight = depth_weight.clamp_min(
                0.1 * (
                    (depth_weight * masks).mean((-2, -1), keepdim=True)
                    / masks.float().mean((-2, -1), keepdim=True).add(1e-7)
                )
            )
            point_error = (
                (aligned_points[point_batch] - points).abs()
                / (depth_weight[..., None] + 1e-6)
            ).mean(-1)
            labels = (
                point_error.detach() < self.confidence_error_threshold
            ).float()
            confidence_loss = F.binary_cross_entropy_with_logits(
                logits[point_batch][..., 0][masks].float(),
                labels[masks],
                reduction="mean",
            )

        return {
            "local_pts_loss": point_loss,
            "normal_loss": normal_loss,
            "confidence_loss": confidence_loss,
        }, scale


class ABotPoseLoss(nn.Module):
    """Composition-aware pose and rotation-refiner supervision."""

    def __init__(
        self,
        translation_weight=100.0,
        rotation_weight=1.0,
        max_pair_gap=11,
        gap_gamma=0.5,
        huber_delta=0.1,
        smooth_temporal_weight=1.0,
    ):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.max_pair_gap = max_pair_gap
        self.gap_gamma = gap_gamma
        self.huber_delta = huber_delta
        self.smooth_temporal_weight = smooth_temporal_weight

    def smooth_loss(self, pred):
        residual = pred.get("rotation_residual")
        if residual is None or residual.numel() == 0:
            return pred["local_points"].sum() * 0.0

        magnitude = residual.float().square().mean()
        variation = magnitude * 0.0
        if residual.shape[1] > 1:
            variation = (
                residual[:, 1:] - residual[:, :-1]
            ).float().square().mean()
        return magnitude + self.smooth_temporal_weight * variation

    def forward(self, pred, gt, scale):
        pred_poses = pred["camera_poses"].float().clone()
        pred_poses[..., :3, 3] *= scale[:, None, None]
        gt_poses = gt["camera_poses"]

        frames = pred_poses.shape[1]
        maximum = min(self.max_pair_gap, frames - 1)
        pair_count = sum(frames - gap for gap in range(1, maximum + 1))
        alpha_mean = sum(
            (frames - gap) * gap ** self.gap_gamma
            for gap in range(1, maximum + 1)
        ) / pair_count

        translation, rotation = [], []
        for gap in range(1, maximum + 1):
            pred_relative = (
                se3_inverse(pred_poses[:, :-gap]) @ pred_poses[:, gap:]
            )
            gt_relative = se3_inverse(gt_poses[:, :-gap]) @ gt_poses[:, gap:]

            translation.append(F.huber_loss(
                pred_relative[..., :3, 3],
                gt_relative[..., :3, 3],
                delta=self.huber_delta,
                reduction="none",
            ).mean(-1))
            rotation.append(
                gap ** self.gap_gamma / alpha_mean
                * _rotation_angle(
                    pred_relative[..., :3, :3],
                    gt_relative[..., :3, :3],
                )
            )

        translation = torch.cat(translation, dim=1)
        rotation = torch.cat(rotation, dim=1)
        point_batch = [
            i for i, name in enumerate(gt["dataset_names"])
            if name not in __NO_POINT_DATASETS__
        ]

        translation_loss = (
            translation[point_batch].mean()
            if point_batch else pred_poses.sum() * 0.0
        )
        rotation_loss = rotation.mean()
        pose_loss = (
            self.translation_weight * translation_loss
            + self.rotation_weight * rotation_loss
        )

        return {
            "pose_loss": pose_loss,
            "trans_loss": translation_loss,
            "rot_loss": rotation_loss,
            "smooth_loss": self.smooth_loss(pred),
        }


class ABotReconLoss(nn.Module):
    """Combine the five loss terms in report Eq. (15)."""

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
        self.point_weight = point_weight
        self.normal_weight = normal_weight
        self.pose_weight = pose_weight
        self.smooth_weight = smooth_weight
        self.confidence_weight = confidence_weight
        self.max_pair_gap = max_pair_gap

        self.point_loss = ABotPointLoss(
            local_align_res=local_align_res,
            confidence_error_threshold=confidence_error_threshold,
        )
        self.pose_loss = ABotPoseLoss(
            translation_weight=translation_weight,
            rotation_weight=rotation_weight,
            max_pair_gap=max_pair_gap,
            gap_gamma=gap_gamma,
            huber_delta=huber_delta,
            smooth_temporal_weight=smooth_temporal_weight,
        )

    def prepare_gt(self, gt_raw):
        points, masks, poses, dataset_names = _stack_training_views(gt_raw)
        points = points.float()
        masks = masks.bool() & torch.isfinite(points).all(-1)
        poses = poses.float()

        batch_size, frames = points.shape[:2]

        first_w2c = se3_inverse(poses[:, 0])
        global_points = torch.einsum(
            "bij,bnhwj->bnhwi",
            first_w2c,
            homogenize_points(points),
        )[..., :3]
        poses = first_w2c[:, None] @ poses

        valid_batch = masks.sum((-1, -2, -3)) > 0
        if valid_batch.any():
            valid_points = global_points[valid_batch].clone()
            valid_points[~masks[valid_batch]] = 0
            valid_points = valid_points.reshape(valid_batch.sum(), frames, -1, 3)
            norm_factor = valid_points.norm(dim=-1).sum((-1, -2)) / (
                masks[valid_batch].float().sum((-1, -2, -3)) + 1e-8
            )
            global_points[valid_batch] /= norm_factor[:, None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[:, None, None]

        local_points = torch.einsum(
            "bnij,bnhwj->bnhwi",
            se3_inverse(poses),
            homogenize_points(global_points),
        )[..., :3]

        if dataset_names is None:
            dataset_names = ["Unknown"] * batch_size
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names] * batch_size
        else:
            dataset_names = list(dataset_names)

        return {
            "global_points": global_points,
            "local_points": local_points,
            "valid_masks": masks,
            "camera_poses": poses,
            "dataset_names": dataset_names,
        }

    def prepare_targets(self, gt_raw):
        return self.prepare_gt(gt_raw)

    def normalize_pred(self, pred, gt):
        local_points = pred["local_points"]
        camera_poses = pred["camera_poses"]
        batch_size, frames = local_points.shape[:2]
        masks = gt["valid_masks"]

        valid_points = local_points.clone()
        valid_points[~masks] = 0
        valid_points = valid_points.reshape(batch_size, frames, -1, 3)
        norm_factor = valid_points.norm(dim=-1).sum((-1, -2)) / (
            masks.float().sum((-1, -2, -3)) + 1e-8
        )

        pred["local_points"] = (
            local_points / norm_factor[:, None, None, None, None]
        )
        if pred.get("global_points") is not None:
            pred["global_points"] /= norm_factor[:, None, None, None, None]

        camera_poses = camera_poses.clone()
        camera_poses[..., :3, 3] /= norm_factor[:, None, None]
        pred["camera_poses"] = camera_poses
        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        point_details, scale = self.point_loss(pred, gt)
        pose_details = self.pose_loss(pred, gt, scale)
        details = {**point_details, **pose_details}

        final_loss = (
            self.point_weight * details["local_pts_loss"]
            + self.normal_weight * details["normal_loss"]
            + self.pose_weight * details["pose_loss"]
            + self.smooth_weight * details["smooth_loss"]
            + self.confidence_weight * details["confidence_loss"]
        )
        details["point_scale"] = scale.mean()
        return final_loss, details
