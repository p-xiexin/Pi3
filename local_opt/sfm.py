"""Single-window Glob3R structure-from-motion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from pi3.utils.geometry import depth_edge

from .factor_graph import (
    MIN_EDGE_COVISIBILITY,
    DroidFactorGraph,
    build_droid_factor_graph,
)
from .frame import Frames
from .matching import PairMatch, match_batch
from .visualization import save_keyframe_matching_overviews


@dataclass
class Glob3RSfMConfig:
    """Appendix C.1 defaults and local solver controls."""

    keyframe_projection_threshold: float = 0.2
    depth_confidence_threshold: float = 0.4
    warp_confidence_threshold: float = 0.8
    droid_solver: str = "moba"
    droid_iterations: int = 12
    droid_downsample: int = 8


@dataclass
class SfMResult:
    raw_frames: Frames
    optimized_frames: Frames
    keyframes: torch.Tensor
    matches: list[PairMatch]
    raw_Ps_W: torch.Tensor
    raw_RGBs: torch.Tensor
    raw_frame_ids: torch.Tensor
    dense_Ps_W: Optional[torch.Tensor]
    dense_RGBs: Optional[torch.Tensor]
    dense_frame_ids: Optional[torch.Tensor]
    disps: torch.Tensor
    initial_optimization_error: torch.Tensor
    optimization_error: torch.Tensor

    @property
    def T_WCs(self) -> torch.Tensor:
        return self.optimized_frames.T_WCs

    @property
    def T_CWs(self) -> torch.Tensor:
        return torch.linalg.inv(self.optimized_frames.T_WCs)

    @property
    def Ks(self) -> torch.Tensor:
        return self.optimized_frames.Ks

    @property
    def deltas(self) -> torch.Tensor:
        return self.optimized_frames.deltas


def select_keyframes_eq4(
    frames: Frames,
    proj_threshold: float,
    conf_threshold: float,
) -> torch.Tensor:
    """Select keyframes using the valid-projection count in paper Eq. (4).

    n_t = max_{r in K} sum_u 1[pi(T_{t->r} X_t(u)) in D,
                                z_{t->r}(u) > 0, C_t(u) > tau_c].
    Candidate t is added to K when n_t < tau_proj.
    """

    keyframes = [0]
    height, width = frames.Xs_C.shape[1:3]
    pixel_threshold = proj_threshold * height * width
    for candidate in range(1, len(frames)):
        # Eq. (4): transform candidate points X_t into every existing
        # keyframe r, then retain the largest valid projection count n_t.
        T_Cr_Ct = (
            torch.linalg.inv(frames.T_WCs[keyframes]) @ frames.T_WCs[candidate]
        )
        # [H, W, 3] -> [M, 3]
        Xs_Ct = frames.Xs_C[candidate].reshape(-1, 3)
        Xs_Cr = torch.einsum(
            "kij,mj->kmi",
            T_Cr_Ct[:, :3, :3],
            Xs_Ct,
        ) + T_Cr_Ct[:, None, :3, 3]
        ps = torch.einsum(
            "kij,kmj->kmi", frames.Ks[keyframes], Xs_Cr
        )
        uvs = ps[..., :2] / ps[..., 2:3].clamp_min(1.0e-8)
        valid = (
            (frames.Cs[candidate].reshape(1, -1) > conf_threshold)
            & (Xs_Cr[..., 2] > 0)
            & (uvs[..., 0] >= 0)
            & (uvs[..., 0] <= width - 1)
            & (uvs[..., 1] >= 0)
            & (uvs[..., 1] <= height - 1)
        )
        count = valid.sum(dim=-1).max()
        if count < pixel_threshold:
            keyframes.append(candidate)
    return torch.tensor(keyframes, device=frames.Xs_C.device, dtype=torch.long)


def _undistort_normalized(
    uvs_distorted: torch.Tensor,
    delta: torch.Tensor,
    iterations: int = 12,
) -> torch.Tensor:
    """Invert the Eq. (6) Brown-Conrady model for dense backprojection."""

    k1, k2, p1, p2 = delta.unbind()
    uvs = uvs_distorted.clone()
    for _ in range(iterations):
        x, y = uvs.unbind(dim=-1)
        radius_squared = x.square() + y.square()
        radial = 1 + k1 * radius_squared + k2 * radius_squared.square()
        tangential = torch.stack(
            (
                2 * p1 * x * y + p2 * (radius_squared + 2 * x.square()),
                p1 * (radius_squared + 2 * y.square()) + 2 * p2 * x * y,
            ),
            dim=-1,
        )
        uvs = (uvs_distorted - tangential) / radial[..., None]
    return uvs


class Glob3RSfMPipeline:
    """Run the single-window pipeline in Glob3R Secs. 3.2 and 3.3.

    The frozen backbone predicts Eq. (1) once. Eq. (4) selects keyframes, and
    Eq. (2) matches every selected keyframe to all other frames using the same
    cached features. The directed dense matches directly form DROID-style
    motion factors, followed by keyframe point-cloud fusion. Sliding-window
    scheduling and cross-window association are outside this local pipeline.
    """

    def __init__(self, model, config: Optional[Glob3RSfMConfig] = None):
        self.model = model.eval()
        self.config = config or Glob3RSfMConfig()

    @torch.no_grad()
    def run(
        self,
        Is: torch.Tensor,
        K: torch.Tensor,
        visualization_dir: Optional[str | Path] = None,
    ) -> SfMResult:
        """Reconstruct one image window with the paper's local SfM flow."""

        frame_count = Is.shape[0]
        Is = Is.float()
        Ks = K.to(device=Is.device, dtype=Is.dtype)
        if Ks.ndim == 2:
            # [3, 3] -> [1, 3, 3]
            Ks = Ks.unsqueeze(dim=0)
            # [1, 3, 3] -> [N, 3, 3]
            Ks = Ks.expand(frame_count, -1, -1).clone()

        # Sec. 3.2: evaluate Eq. (1) once, select Eq. (4) keyframes here, then
        # evaluate Eq. (2) for those references using the cached features.
        print(f"Backbone inference: frames [0, {frame_count - 1}]")
        geometry, patch_tokens, encoder_features = self.model.infer_window(
            Is.unsqueeze(dim=0)
        )

        # Convert the batch-one network output exactly once. Every downstream
        # stage reads the same batched frame state, as in MASt3R-Fusion.
        # [B, N, H, W, 3] -> [N, H, W, 3]
        Xs_C = geometry["local_points"].squeeze(dim=0)
        # [B, N, 4, 4] -> [N, 4, 4]
        T_WCs = geometry["camera_poses"].squeeze(dim=0).clone()
        # [B, N, H, W, 1] -> [N, H, W]
        Cs = geometry["conf"].squeeze(dim=0).sigmoid().squeeze(dim=-1)
        # Express all Pi3 poses in the first camera frame. Raw and optimized
        # clouds therefore share one world frame, while the first pose is the
        # fixed gauge used by DROID motion-only BA.
        T_C0W = torch.linalg.inv(T_WCs[0])
        T_WCs = T_C0W.unsqueeze(dim=0) @ T_WCs
        frames = Frames(
            Is=Is,
            Xs_C=Xs_C,
            Cs=Cs,
            Ks=Ks,
            deltas=torch.zeros(
                frame_count,
                4,
                device=Is.device,
                dtype=Is.dtype,
            ),
            T_WCs=T_WCs,
        )
        keyframes = select_keyframes_eq4(
            frames,
            proj_threshold=self.config.keyframe_projection_threshold,
            conf_threshold=self.config.depth_confidence_threshold,
        )

        matches = match_batch(
            self.model,
            patch_tokens,
            encoder_features,
            frames,
            keyframes,
        )

        # Keep the first line for all-frame BA. Swap these two lines to compare
        # keyframe-only BA with exactly the same matching and solver code.
        # factor_matches = matches
        factor_matches = [match for match in matches if match.t in keyframes]
        ba_scope = "all frames" if factor_matches is matches else "keyframes only"
        print(f"BA frame scope: {ba_scope}")
        factor_graph = build_droid_factor_graph(
            frames,
            factor_matches,
            stride=self.config.droid_downsample,
            depth_confidence_threshold=self.config.depth_confidence_threshold,
            warp_confidence_threshold=self.config.warp_confidence_threshold,
        )
        factor_counts = (factor_graph.weight[0, ..., 0] > 0).sum(dim=(1, 2))
        print(
            "DROID factor graph: retained "
            f"{factor_graph.rs.numel()}/{len(factor_matches)} edges "
            f"with covisibility >= {MIN_EDGE_COVISIBILITY:.0%}"
        )
        for r, t, count, covisibility in zip(
            factor_graph.rs.tolist(),
            factor_graph.ts.tolist(),
            factor_counts.tolist(),
            factor_graph.covisibility.tolist(),
        ):
            print(
                f"  {r} -> {t}: factors={count}, "
                f"covisibility={covisibility:.2%}"
            )
        if visualization_dir is not None:
            paths = save_keyframe_matching_overviews(
                visualization_dir,
                frames.Is,
                factor_matches,
                factor_graph,
            )
            print(f"Saved {len(paths)} matching overview(s) to {visualization_dir}")

        from .droid_ba.adapter import optimize_droid_ba

        print(f"Optimization method: DROID-SLAM {self.config.droid_solver.upper()}")
        optimization = optimize_droid_ba(
            frames.T_WCs,
            factor_graph,
            solver=self.config.droid_solver,
            iterations=self.config.droid_iterations,
        )

        raw_Ps_W, raw_RGBs, raw_frame_ids = self._reconstruct_raw(
            frames,
            keyframes,
        )
        Xs_C_opt = None
        if self.config.droid_solver == "ba":
            Xs_C_opt = self._apply_ba_depths(
                frames,
                factor_graph,
                optimization.disps,
            )
        reconstruction_frames = frames.with_optimization(
            T_WCs=optimization.T_WCs,
            Xs_C=Xs_C_opt,
        )
        dense_Ps_W, dense_RGBs, dense_frame_ids = self._reconstruct_dense(
            reconstruction_frames,
            keyframes,
        )
        return SfMResult(
            raw_frames=frames,
            optimized_frames=reconstruction_frames,
            keyframes=keyframes,
            matches=matches,
            raw_Ps_W=raw_Ps_W,
            raw_RGBs=raw_RGBs,
            raw_frame_ids=raw_frame_ids,
            dense_Ps_W=dense_Ps_W,
            dense_RGBs=dense_RGBs,
            dense_frame_ids=dense_frame_ids,
            disps=optimization.disps,
            initial_optimization_error=optimization.initial_error,
            optimization_error=optimization.final_error,
        )

    def _apply_ba_depths(
        self,
        frames: Frames,
        graph: DroidFactorGraph,
        disps: torch.Tensor,
    ) -> torch.Tensor:
        """Lift the full-BA inverse-depth correction to calibrated point maps."""

        height, width = frames.Xs_C.shape[1:3]
        low_height, low_width = graph.disps.shape[-2:]
        disps = disps.unsqueeze(dim=0)
        depth_scale_low = torch.where(
            disps > 0,
            graph.disps / disps.clamp_min(1.0e-8),
            torch.ones_like(disps),
        )
        ys_full = torch.arange(
            height, device=disps.device, dtype=disps.dtype
        ) / graph.stride
        xs_full = torch.arange(
            width, device=disps.device, dtype=disps.dtype
        ) / graph.stride
        ys_full, xs_full = torch.meshgrid(ys_full, xs_full, indexing="ij")
        full_grid = torch.stack(
            (
                2 * xs_full / (low_width - 1) - 1,
                2 * ys_full / (low_height - 1) - 1,
            ),
            dim=-1,
        )
        depth_scale_full = F.grid_sample(
            depth_scale_low,
            full_grid.unsqueeze(dim=0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(dim=0)
        Ds_opt = frames.Xs_C[..., 2] * depth_scale_full
        ys, xs = torch.meshgrid(
            torch.arange(height, device=disps.device, dtype=disps.dtype),
            torch.arange(width, device=disps.device, dtype=disps.dtype),
            indexing="ij",
        )
        uv1s = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
        rays_C = torch.einsum(
            "nij,hwj->nhwi", torch.linalg.inv(frames.Ks), uv1s
        )
        return rays_C * Ds_opt.unsqueeze(dim=-1)

    def _reconstruct_raw(
        self,
        frames: Frames,
        keyframes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse Eq. (4) keyframe point maps using predicted camera poses."""

        Ps_W_all, RGBs_all, frame_ids_all = [], [], []
        for frame_index in keyframes:
            frame = frames[frame_index]
            X_C = frame.X_C
            X_W = torch.einsum(
                "ij,hwj->hwi", frame.T_WC[:3, :3], X_C
            ) + frame.T_WC[None, None, :3, 3]
            valid = (
                (frame.C > self.config.depth_confidence_threshold)
                & ~depth_edge(X_C[..., 2], rtol=0.03)
                & torch.isfinite(X_W).all(dim=-1)
                & (X_C[..., 2] > 0)
            )
            Ps_W = X_W[valid]
            Ps_W_all.append(Ps_W)
            RGBs_all.append(frame.I.permute(1, 2, 0)[valid])
            frame_ids_all.append(
                torch.full(
                    (Ps_W.shape[0],),
                    frame.frame_id,
                    device=Ps_W.device,
                    dtype=torch.long,
                )
            )
        return (
            torch.cat(Ps_W_all),
            torch.cat(RGBs_all),
            torch.cat(frame_ids_all),
        )

    def _reconstruct_dense(
        self,
        frames: Frames,
        keyframes: torch.Tensor,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Fuse calibrated keyframe point maps using optimized poses."""

        Ps_W_all, RGBs_all, frame_ids_all = [], [], []
        for frame_index in keyframes:
            frame = frames[frame_index]
            D = frame.D
            y_grid, x_grid = torch.meshgrid(
                torch.arange(D.shape[0], device=D.device, dtype=D.dtype),
                torch.arange(D.shape[1], device=D.device, dtype=D.dtype),
                indexing="ij",
            )
            uv1s = torch.stack(
                (x_grid, y_grid, torch.ones_like(x_grid)), dim=-1
            )
            uvs_norm = torch.einsum(
                "ij,hwj->hwi", torch.linalg.inv(frame.K), uv1s
            )
            uvs_undistorted = _undistort_normalized(
                uvs_norm[..., :2] / uvs_norm[..., 2:3],
                frame.delta,
            )
            Xs_C = torch.cat(
                (uvs_undistorted * D[..., None], D[..., None]), dim=-1
            )
            valid = (
                (frame.C > self.config.depth_confidence_threshold)
                & ~depth_edge(D, rtol=0.03)
                & torch.isfinite(Xs_C).all(dim=-1)
                & (D > 0)
            )
            Ps_W = torch.einsum(
                "ij,nj->ni", frame.T_WC[:3, :3], Xs_C[valid]
            ) + frame.T_WC[:3, 3]
            Ps_W_all.append(Ps_W)
            RGBs_all.append(frame.I.permute(1, 2, 0)[valid])
            frame_ids_all.append(
                torch.full(
                    (Ps_W.shape[0],),
                    frame.frame_id,
                    device=Ps_W.device,
                    dtype=torch.long,
                )
            )
            print(f"Dense frame {frame.frame_id}: points={Ps_W.shape[0]}")
        if not Ps_W_all:
            return None, None, None
        return (
            torch.cat(Ps_W_all),
            torch.cat(RGBs_all),
            torch.cat(frame_ids_all),
        )


__all__ = [
    "Glob3RSfMPipeline",
    "Glob3RSfMConfig",
    "SfMResult",
    "select_keyframes_eq4",
]
