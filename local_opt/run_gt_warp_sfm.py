"""Temporary ScanNet GT-warp frontend for isolating the SfM backend."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from pi3.models.glob3r.geometry import build_ground_truth_warp

from .backend.proj import MIN_DEPTH
from .sfm import Glob3RSfMConfig, Glob3RSfMPipeline, _weighted_scale_ransac


# Temporary experiment settings. Edit these values directly before running.
DATA_ROOT = "data/scannet"
MODE = "train"
SCENE_INDEX = 123
FRAME_NUM = 30
FRAME_STEP = 5
HEIGHT = 336
WIDTH = 448
KEYFRAME_THRESHOLD = 0.5
DEVICE = "cuda:0"
OUTPUT_DIR = Path("outputs/gt_warp_sfm")
RANDOM_SEED = 0
ROTATION_NOISE_DEG = 3.0
TRANSLATION_NOISE_STD_M = 0.05
WARP_NOISE_STD_PX = 1.5
DEPTH_NOISE_STD_REL = 0.02


def perturb_camera_poses(T_WCs: torch.Tensor) -> torch.Tensor:
    """Perturb the SfM initialization while keeping frame 0 as the gauge."""

    S = T_WCs.shape[0]
    omegas = torch.randn(S, 3, device=T_WCs.device, dtype=T_WCs.dtype)
    omegas *= math.radians(ROTATION_NOISE_DEG)
    omegas[0] = 0

    theta = torch.linalg.vector_norm(omegas, dim=-1, keepdim=True)
    axes = omegas / theta.clamp_min(1.0e-8)
    zeros = torch.zeros(S, device=T_WCs.device, dtype=T_WCs.dtype)
    K = torch.stack(
        (
            zeros,
            -axes[:, 2],
            axes[:, 1],
            axes[:, 2],
            zeros,
            -axes[:, 0],
            -axes[:, 1],
            axes[:, 0],
            zeros,
        ),
        dim=-1,
    ).reshape(S, 3, 3)
    eye = torch.eye(3, device=T_WCs.device, dtype=T_WCs.dtype).expand(S, -1, -1)
    R_delta = (
        eye
        + theta.sin().unsqueeze(dim=-1) * K
        + (1 - theta.cos()).unsqueeze(dim=-1) * (K @ K)
    )

    translation_noise = torch.randn(
        S, 3, device=T_WCs.device, dtype=T_WCs.dtype
    ) * TRANSLATION_NOISE_STD_M
    translation_noise[0] = 0

    T_delta = torch.eye(
        4, device=T_WCs.device, dtype=T_WCs.dtype
    ).repeat(S, 1, 1)
    T_delta[:, :3, :3] = R_delta
    T_delta[:, :3, 3] = translation_noise
    return T_WCs @ T_delta


def pose_errors(
    T_WCs: torch.Tensor,
    T_WCs_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean rotation error in degrees and camera-center RMSE."""

    R_error = T_WCs[:, :3, :3].transpose(-1, -2) @ T_WCs_gt[:, :3, :3]
    cos_angle = (
        (R_error.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2
    ).clamp(-1, 1)
    rotation_error = torch.rad2deg(torch.acos(cos_angle)).mean()
    translation_error = torch.sqrt(
        ((T_WCs[:, :3, 3] - T_WCs_gt[:, :3, 3]) ** 2).sum(dim=-1).mean()
    )
    return rotation_error, translation_error


class GTWarp(nn.Module):
    """Expose controlled noisy ScanNet observations to Glob3RSfM."""

    def __init__(
        self,
        Ds_gt: torch.Tensor,
        Ds_init: torch.Tensor,
        Ks: torch.Tensor,
        T_WCs_gt: torch.Tensor,
        T_WCs_init: torch.Tensor,
    ) -> None:
        super().__init__()
        _, H, W = Ds_gt.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=Ds_gt.device, dtype=Ds_gt.dtype),
            torch.arange(W, device=Ds_gt.device, dtype=Ds_gt.dtype),
            indexing="ij",
        )
        pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
        rays = torch.einsum("sij,hwj->shwi", torch.linalg.inv(Ks), pixels)

        self.register_buffer("Ds_gt", Ds_gt)
        self.register_buffer("Ks", Ks)
        self.register_buffer("T_WCs_gt", T_WCs_gt)
        self.register_buffer("T_WCs_init", T_WCs_init)
        self.register_buffer("Xs_C_gt", rays * Ds_gt.unsqueeze(dim=-1))
        self.register_buffer("Xs_C_init", rays * Ds_init.unsqueeze(dim=-1))
        self.register_buffer(
            "Cs_logits",
            torch.where(
                Ds_gt > 0,
                Ds_gt.new_tensor(20.0),
                Ds_gt.new_tensor(-20.0),
            ),
        )

    @torch.no_grad()
    def infer_window(self, Is: torch.Tensor):
        """Return perturbed Eq. (1) geometry and unused feature placeholders."""

        geometry = {
            "local_points": self.Xs_C_init.unsqueeze(dim=0),
            "camera_poses": self.T_WCs_init.unsqueeze(dim=0),
            "conf": self.Cs_logits.unsqueeze(dim=0).unsqueeze(dim=-1),
        }
        return geometry, None, None

    @torch.no_grad()
    def match_pair(
        self,
        patch_tokens,
        encoder_features,
        Is: torch.Tensor,
        reference_index: int,
    ):
        """Return noisy GT-centered reference-to-target observations."""

        target_indices = [
            index for index in range(self.Ds_gt.shape[0])
            if index != reference_index
        ]
        target_from_reference = (
            torch.linalg.inv(self.T_WCs_gt[target_indices])
            @ self.T_WCs_gt[reference_index]
        )
        supervision = build_ground_truth_warp(
            self.Ds_gt[reference_index].unsqueeze(dim=0),
            self.Ds_gt[target_indices].unsqueeze(dim=0),
            self.Ks[reference_index].unsqueeze(dim=0),
            self.Ks[target_indices].unsqueeze(dim=0),
            target_from_reference.unsqueeze(dim=0),
        )
        # [B, T, H, W, 2] -> [B, T, 2, H, W]
        W_r2t = supervision.warp.permute(0, 1, 4, 2, 3)
        # [B, T, H, W] -> [B, T, 1, H, W]
        Q_r2t = supervision.confidence.unsqueeze(dim=2).float()
        # Keep GT correspondences as the center of each observation. The
        # unbounded pixel noise deliberately leaves outliers for the backend.
        W_r2t = W_r2t + WARP_NOISE_STD_PX * torch.randn_like(W_r2t) * Q_r2t
        return SimpleNamespace(
            target_indices=target_indices,
            coarse_warp=W_r2t,
            coarse_confidence=Q_r2t,
            warp_stages=[W_r2t],
            confidence_stages=[Q_r2t],
        )


def main() -> None:
    from datasets.glob3r_scannet_dataset import Glob3RScannetValidationDataset
    from .run_glob3r_sfm import save_ply

    torch.manual_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Glob3RScannetValidationDataset(
        data_root=DATA_ROOT,
        mode=MODE,
        frame_num=FRAME_NUM,
        frame_step=FRAME_STEP,
        resolution=[[WIDTH, HEIGHT]],
    )
    views = sorted(
        dataset[SCENE_INDEX],
        key=lambda view: int(view["instance"]),
    )
    Is = torch.stack([view["img"] for view in views]).to(DEVICE)
    Ds_gt = torch.stack(
        [torch.as_tensor(view["depthmap"]) for view in views]
    ).float().to(DEVICE)
    Ks = torch.stack(
        [torch.as_tensor(view["camera_intrinsics"]) for view in views]
    ).float().to(DEVICE)
    T_WCs_gt = torch.stack(
        [torch.as_tensor(view["camera_pose"]) for view in views]
    ).float().to(DEVICE)
    T_WCs_init = perturb_camera_poses(T_WCs_gt)
    depth_noise = DEPTH_NOISE_STD_REL * torch.randn_like(Ds_gt)
    Ds_init = torch.where(Ds_gt > 0, Ds_gt * (1 + depth_noise), Ds_gt)

    print("GT frame order:")
    for index, view in enumerate(views):
        print(f"  [{index:04d}] {view['label']}/{view['instance']}")

    model = GTWarp(Ds_gt, Ds_init, Ks, T_WCs_gt, T_WCs_init).to(DEVICE).eval()
    config = Glob3RSfMConfig(
        keyframe_projection_threshold=KEYFRAME_THRESHOLD,
    )
    pipeline = Glob3RSfMPipeline(model, config)
    result = pipeline.run(
        Is,
        Ks,
        visualization_dir=OUTPUT_DIR / "matching",
    )

    T_C0W_gt = torch.linalg.inv(T_WCs_gt[0])
    T_WCs_gt = T_C0W_gt.unsqueeze(dim=0) @ T_WCs_gt
    rotation_init, translation_init = pose_errors(
        result.raw_frames.T_WCs, T_WCs_gt
    )
    rotation_final, translation_final = pose_errors(result.T_WCs, T_WCs_gt)
    print(
        "Pose error (mean rotation / center RMSE): "
        f"{float(rotation_init):.4f} deg / {float(translation_init):.6f} m -> "
        f"{float(rotation_final):.4f} deg / {float(translation_final):.6f} m"
    )

    gt_Ps_W = []
    gt_RGBs = []
    gt_frame_ids = []
    dba_Ps_W = []
    dba_RGBs = []
    dba_frame_ids = []
    scale_track_inliers = torch.zeros_like(result.track_inliers)
    surface_distance = torch.full_like(result.tracks.ws[0], -1)
    for r_tensor in result.keyframes:
        r = int(r_tensor)
        Xs_C_gt = model.Xs_C_gt[r]
        RGB = Is[r].permute(1, 2, 0)

        Ps_W_gt = torch.einsum(
            "ij,hwj->hwi", T_WCs_gt[r, :3, :3], Xs_C_gt
        ) + T_WCs_gt[r, None, None, :3, 3]
        gt_mask = (
            (Ds_gt[r] > 0)
            & torch.isfinite(Ps_W_gt).all(dim=-1)
        )
        gt_Ps_W.append(Ps_W_gt[gt_mask])
        gt_RGBs.append(RGB[gt_mask])
        gt_frame_ids.append(
            torch.full(
                (int(gt_mask.sum()),), r, device=DEVICE, dtype=torch.long
            )
        )

        js = torch.nonzero(result.tracks.rs == r, as_tuple=False).squeeze(dim=-1)
        T_WC = result.T_WCs[r]
        T_CW = torch.linalg.inv(T_WC)
        sparse_Xs_C = torch.einsum(
            "ij,pj->pi", T_CW[:3, :3], result.Xs_W[js]
        ) + T_CW[None, :3, 3]

        anchor_us = result.tracks.us[r, js].round().long()
        anchor_Xs_C_gt = Xs_C_gt[anchor_us[:, 1], anchor_us[:, 0]]
        scale_candidates = (
            sparse_Xs_C[:, 2]
            / anchor_Xs_C_gt[:, 2].clamp_min(1.0e-8)
        )
        scale_weights = (
            result.tracks.ws[:, js] * result.tracks.mask[:, js]
        ).sum(dim=0)
        scale_valid = (
            torch.isfinite(sparse_Xs_C).all(dim=-1)
            & torch.isfinite(scale_candidates)
            & (sparse_Xs_C[:, 2] > MIN_DEPTH)
            & (anchor_Xs_C_gt[:, 2] > 0)
            & (scale_candidates > 0)
        )
        scale, inliers = _weighted_scale_ransac(
            scale_candidates[scale_valid],
            scale_weights[scale_valid],
            threshold=config.scale_ransac_threshold,
        )
        valid_js = js[scale_valid]
        scale_track_inliers[valid_js[inliers]] = True

        Xs_C_dba = Xs_C_gt * scale
        anchor_surface_error = torch.linalg.vector_norm(
            sparse_Xs_C[scale_valid] - scale * anchor_Xs_C_gt[scale_valid],
            dim=-1,
        )
        surface_distance[valid_js] = anchor_surface_error
        inlier_surface_error = anchor_surface_error[inliers]
        Ps_W_dba = torch.einsum(
            "ij,hwj->hwi", T_WC[:3, :3], Xs_C_dba
        ) + T_WC[None, None, :3, 3]
        dba_mask = (
            (Ds_gt[r] > 0)
            & torch.isfinite(Ps_W_dba).all(dim=-1)
            & (Xs_C_dba[..., 2] > 0)
        )
        print(
            f"GT dense frame {r}: sparse scale={float(scale):.6g}, "
            f"scale inliers={int(inliers.sum())}/{int(scale_valid.sum())}, "
            f"surface median={float(inlier_surface_error.median()):.6g}"
        )
        dba_Ps_W.append(Ps_W_dba[dba_mask])
        dba_RGBs.append(RGB[dba_mask])
        dba_frame_ids.append(
            torch.full(
                (int(dba_mask.sum()),), r, device=DEVICE, dtype=torch.long
            )
        )
    save_ply(
        OUTPUT_DIR / "gt_reference.ply",
        torch.cat(gt_Ps_W),
        torch.cat(gt_RGBs),
        {"frame_id": torch.cat(gt_frame_ids)},
    )
    observation_count = result.tracks.mask.sum(dim=0)
    observation_count[~scale_track_inliers] = -1
    sparse_fields = {
        "reference_id": result.tracks.rs,
        "track_id": result.tracks.ks,
        "observation_count": observation_count,
    }
    save_ply(
        OUTPUT_DIR / "sparse_perturbed.ply",
        result.Xs_W0,
        scalar_fields=sparse_fields,
    )
    save_ply(
        OUTPUT_DIR / "sparse_optimized.ply",
        result.Xs_W,
        scalar_fields={**sparse_fields, "surface_distance": surface_distance},
    )
    save_ply(
        OUTPUT_DIR / "dense_dba.ply",
        torch.cat(dba_Ps_W),
        torch.cat(dba_RGBs),
        {"frame_id": torch.cat(dba_frame_ids)},
    )
    print(f"Saved GT-warp diagnostic results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
