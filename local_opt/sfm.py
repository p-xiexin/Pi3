"""Single-window Glob3R structure-from-motion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import torch

from pi3.utils.geometry import depth_edge

from .backend import bundle_adjust, opt_pose_ray
from .frame import Frames
from .matching import Tracks, match_tracks
from .visualization import save_matching_matrix


@dataclass
class Glob3RSfMConfig:
    """Appendix C.1 defaults and local optimization controls."""

    tracking_points_per_keyframe: int = 512
    keyframe_projection_threshold: float = 0.2
    depth_confidence_threshold: float = 0.6
    warp_confidence_threshold: float = 0.8
    scale_ransac_threshold: float = 0.1
    eq5_iterations: int = 15
    eq6_iterations: int = 20


@dataclass
class SfMResult:
    raw_frames: Frames
    optimized_frames: Frames
    keyframes: torch.Tensor
    tracks: Tracks
    track_inliers: torch.Tensor
    Xs_W0: torch.Tensor
    Xs_W: torch.Tensor
    raw_Ps_W: torch.Tensor
    raw_RGBs: torch.Tensor
    raw_frame_ids: torch.Tensor
    dense_Ps_W: Optional[torch.Tensor]
    dense_RGBs: Optional[torch.Tensor]
    dense_frame_ids: Optional[torch.Tensor]
    eq5_loss: torch.Tensor
    eq6_loss: torch.Tensor

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
    """Select keyframes using the valid-projection count in paper Eq. (4)."""

    keyframes = [0]
    H, W = frames.Xs_C.shape[1:3]
    pixel_threshold = proj_threshold * H * W
    for t in range(1, len(frames)):
        T_Cr_Ct = torch.linalg.inv(frames.T_WCs[keyframes]) @ frames.T_WCs[t]
        Xs_Ct = frames.Xs_C[t].reshape(-1, 3)
        Xs_Cr = torch.einsum(
            "kij,mj->kmi", T_Cr_Ct[:, :3, :3], Xs_Ct
        ) + T_Cr_Ct[:, None, :3, 3]
        ps = torch.einsum("kij,kmj->kmi", frames.Ks[keyframes], Xs_Cr)
        us = ps[..., :2] / ps[..., 2:3].clamp_min(1.0e-8)
        mask = (
            (frames.Cs[t].reshape(1, -1) > conf_threshold)
            & (Xs_Cr[..., 2] > 0)
            & (us[..., 0] >= 0)
            & (us[..., 0] <= W - 1)
            & (us[..., 1] >= 0)
            & (us[..., 1] <= H - 1)
        )
        if mask.sum(dim=-1).max() < pixel_threshold:
            keyframes.append(t)
    return torch.tensor(keyframes, device=frames.Xs_C.device, dtype=torch.long)


def _weighted_scale_ransac(
    ss: torch.Tensor,
    ws: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every 1D scale hypothesis by its confidence-weighted consensus."""

    consensus = (
        ss.log().unsqueeze(dim=0) - ss.log().unsqueeze(dim=1)
    ).abs() < math.log1p(threshold)
    scores = consensus.to(ws.dtype) @ ws
    inliers = consensus[scores.argmax()]

    ss_inliers = ss[inliers]
    ws_inliers = ws[inliers]
    order = ss_inliers.argsort()
    cumulative = ws_inliers[order].cumsum(dim=0)
    median_index = torch.searchsorted(cumulative, 0.5 * cumulative[-1])
    return ss_inliers[order[median_index]], inliers


class Glob3RSfMPipeline:
    """Run Glob3R Eqs. (1), (2), and (4)–(6) on one image window."""

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
        S = Is.shape[0]
        Is = Is.float()
        Ks = K.to(device=Is.device, dtype=Is.dtype)
        if Ks.ndim == 2:
            Ks = Ks.unsqueeze(dim=0).expand(S, -1, -1).clone()

        print(f"Backbone inference: frames [0, {S - 1}]")
        geometry, patch_tokens, encoder_features = self.model.infer_window(
            Is.unsqueeze(dim=0)
        )
        # [B, S, H, W, 3] -> [S, H, W, 3]
        Xs_C = geometry["local_points"].squeeze(dim=0)
        # [B, S, 4, 4] -> [S, 4, 4]
        T_WCs = geometry["camera_poses"].squeeze(dim=0).clone()
        # [B, S, H, W, 1] -> [S, H, W]
        Cs = geometry["conf"].squeeze(dim=0).sigmoid().squeeze(dim=-1)

        T_C0W = torch.linalg.inv(T_WCs[0])
        T_WCs = T_C0W.unsqueeze(dim=0) @ T_WCs
        frames = Frames(
            Is=Is,
            Xs_C=Xs_C,
            Cs=Cs,
            Ks=Ks,
            deltas=torch.zeros(S, 4, device=Is.device, dtype=Is.dtype),
            T_WCs=T_WCs,
        )
        keyframes = select_keyframes_eq4(
            frames,
            proj_threshold=self.config.keyframe_projection_threshold,
            conf_threshold=self.config.depth_confidence_threshold,
        )
        if visualization_dir is not None:
            visualization_dir = Path(visualization_dir)
            visualization_dir.mkdir(parents=True, exist_ok=True)
            for path in visualization_dir.glob("reference_*.png"):
                path.unlink()

        track_parts = []
        matching_paths = []
        # [S, 3, H, W] -> [B, S, 3, H, W]
        Is_window = frames.Is.unsqueeze(dim=0)
        for r_tensor in keyframes:
            r = int(r_tensor)
            print(f"Matching inference: reference {r} -> frames [0, {S - 1}]")
            output = self.model.match_pair(
                patch_tokens,
                encoder_features,
                Is_window,
                reference_index=r,
            )
            tracks_r = match_tracks(
                output,
                frames,
                reference_index=r,
                points_per_keyframe=self.config.tracking_points_per_keyframe,
                depth_confidence_threshold=self.config.depth_confidence_threshold,
                warp_confidence_threshold=self.config.warp_confidence_threshold,
            )
            track_parts.append(tracks_r)
            if visualization_dir is not None:
                matching_paths.append(
                    save_matching_matrix(
                        visualization_dir,
                        frames,
                        r,
                        output,
                        tracks_r,
                    )
                )

        tracks = Tracks(
            rs=torch.cat([part.rs for part in track_parts]),
            Xs_Cr=torch.cat([part.Xs_Cr for part in track_parts]),
            us=torch.cat([part.us for part in track_parts], dim=1),
            mask=torch.cat([part.mask for part in track_parts], dim=1),
            ws=torch.cat([part.ws for part in track_parts], dim=1),
        )
        print(
            f"Tracks: P={tracks.us.shape[1]}, "
            f"observations={int(tracks.mask.sum())}, "
            f"mean length={float(tracks.mask.sum(dim=0).float().mean()):.2f}"
        )
        if visualization_dir is not None:
            print(
                f"Saved {len(matching_paths)} matching matrix/matrices "
                f"to {visualization_dir}"
            )

        # Each track j is initialized only from its keyframe-local Pi3 point.
        # X_j^0 = T_WC,r X_r(u_r); target-frame point maps are not substituted.
        T_WCr = frames.T_WCs[tracks.rs]
        Xs_W0 = torch.einsum(
            "pij,pj->pi", T_WCr[:, :3, :3], tracks.Xs_Cr
        ) + T_WCr[:, :3, 3]

        is_, js = torch.nonzero(tracks.mask, as_tuple=True)
        uv1s = torch.cat(
            (tracks.us[is_, js], torch.ones_like(tracks.us[is_, js, :1])),
            dim=-1,
        )
        vs = torch.einsum("oij,oj->oi", torch.linalg.inv(frames.Ks[is_]), uv1s)
        T_CWs0 = torch.linalg.inv(frames.T_WCs)
        Rs = T_CWs0[:, :3, :3]
        cs0 = frames.T_WCs[:, :3, 3]
        ws = tracks.ws[is_, js]

        eq5 = opt_pose_ray(
            Rs,
            cs0,
            Xs_W0,
            vs,
            is_,
            js,
            ws,
            iterations=self.config.eq5_iterations,
        )
        T_CWs5 = torch.eye(4, device=Is.device, dtype=Is.dtype).repeat(S, 1, 1)
        T_CWs5[:, :3, :3] = Rs
        T_CWs5[:, :3, 3] = -torch.einsum("sij,sj->si", Rs, eq5.cs)
        print(f"Eq. (5) loss: {float(eq5.loss):.6g}")

        eq6 = bundle_adjust(
            T_CWs5,
            eq5.Xs_W,
            frames.Ks,
            frames.deltas,
            tracks,
            iterations=self.config.eq6_iterations,
        )
        print(f"Eq. (6) loss: {float(eq6.loss):.6g}")
        optimized_frames = frames.with_optimization(
            T_WCs=torch.linalg.inv(eq6.T_CWs)
        )

        raw_Ps_W, raw_RGBs, raw_frame_ids = self._reconstruct_raw(
            frames, keyframes
        )
        dense_Ps_W, dense_RGBs, dense_frame_ids, track_inliers = self._reconstruct_dense(
            optimized_frames,
            tracks,
            eq6.Xs_W,
            keyframes,
        )
        return SfMResult(
            raw_frames=frames,
            optimized_frames=optimized_frames,
            keyframes=keyframes,
            tracks=tracks,
            track_inliers=track_inliers,
            Xs_W0=Xs_W0,
            Xs_W=eq6.Xs_W,
            raw_Ps_W=raw_Ps_W,
            raw_RGBs=raw_RGBs,
            raw_frame_ids=raw_frame_ids,
            dense_Ps_W=dense_Ps_W,
            dense_RGBs=dense_RGBs,
            dense_frame_ids=dense_frame_ids,
            eq5_loss=eq5.loss,
            eq6_loss=eq6.loss,
        )

    def _reconstruct_raw(
        self,
        frames: Frames,
        keyframes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Ps_W_all, RGBs_all, frame_ids_all = [], [], []
        for r_tensor in keyframes:
            r = int(r_tensor)
            frame = frames[r]
            Ps_W = torch.einsum(
                "ij,hwj->hwi", frame.T_WC[:3, :3], frame.X_C
            ) + frame.T_WC[None, None, :3, 3]
            mask = (
                (frame.C > self.config.depth_confidence_threshold)
                & ~depth_edge(frame.D, rtol=0.03)
                & torch.isfinite(Ps_W).all(dim=-1)
                & (frame.D > 0)
            )
            Ps_W_all.append(Ps_W[mask])
            RGBs_all.append(frame.I.permute(1, 2, 0)[mask])
            frame_ids_all.append(
                torch.full(
                    (int(mask.sum()),), r, device=Ps_W.device, dtype=torch.long
                )
            )
        return torch.cat(Ps_W_all), torch.cat(RGBs_all), torch.cat(frame_ids_all)

    def _reconstruct_dense(
        self,
        frames: Frames,
        tracks: Tracks,
        Xs_W: torch.Tensor,
        keyframes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Ps_W_all, RGBs_all, frame_ids_all = [], [], []
        track_inliers = torch.zeros(
            Xs_W.shape[0], device=Xs_W.device, dtype=torch.bool
        )
        T_CWs = torch.linalg.inv(frames.T_WCs)
        for r_tensor in keyframes:
            r = int(r_tensor)
            js = torch.nonzero(tracks.rs == r, as_tuple=False).squeeze(dim=-1)
            Xs_C = torch.einsum(
                "ij,pj->pi", T_CWs[r, :3, :3], Xs_W[js]
            ) + T_CWs[r, :3, 3]
            ss_all = Xs_C[:, 2] / tracks.Xs_Cr[js, 2].clamp_min(1.0e-8)
            ws = (tracks.ws[:, js] * tracks.mask[:, js]).sum(dim=0)
            valid = torch.isfinite(ss_all) & (ss_all > 0)
            ss, inliers = _weighted_scale_ransac(
                ss_all[valid],
                ws[valid],
                threshold=self.config.scale_ransac_threshold,
            )
            valid_js = js[valid]
            track_inliers[valid_js[inliers]] = True

            frame = frames[r]
            Xs_Cr = frame.X_C * ss
            Ps_W = torch.einsum(
                "ij,hwj->hwi", frame.T_WC[:3, :3], Xs_Cr
            ) + frame.T_WC[None, None, :3, 3]
            mask = (
                (frame.C > self.config.depth_confidence_threshold)
                & ~depth_edge(Xs_Cr[..., 2], rtol=0.03)
                & torch.isfinite(Ps_W).all(dim=-1)
                & (Xs_Cr[..., 2] > 0)
            )
            print(
                f"Dense frame {r}: scale={float(ss):.6g}, "
                f"scale_inliers={int(inliers.sum())}/{int(valid.sum())}, "
                f"points={int(mask.sum())}"
            )
            Ps_W_all.append(Ps_W[mask])
            RGBs_all.append(frame.I.permute(1, 2, 0)[mask])
            frame_ids_all.append(
                torch.full(
                    (int(mask.sum()),), r, device=Ps_W.device, dtype=torch.long
                )
            )
        return (
            torch.cat(Ps_W_all),
            torch.cat(RGBs_all),
            torch.cat(frame_ids_all),
            track_inliers,
        )


__all__ = [
    "Glob3RSfMConfig",
    "Glob3RSfMPipeline",
    "SfMResult",
    "select_keyframes_eq4",
]
