"""Adjacent-pose prediction and motion-visual rotation refinement.

``AdjacentPoseHead`` implements report Eqs. (2)-(5).  It turns five camera
tokens per frame into adjacent SE(3) transforms and composes them into a global
trajectory.  ``TemporalRotationRefiner`` implements Eqs. (7)-(11), preserving
the predicted translation and correcting only the relative rotation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TemporalRotationRefiner(nn.Module):
    """Motion-visual contextualized rotation refiner from Eqs. (7)-(11).

    A camera-token pair supplies motion evidence, dense frame tokens supply
    visual evidence, and a depthwise gated TCN aggregates the latest K pairwise
    features.  The output is a bounded axis-angle residual ``delta_omega_i``.
    """

    def __init__(self, desc_dim=512, frame_dim=512, hidden_dim=512,
                 kernel_size=10, max_rot_deg=2.0, num_heads=8):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.kernel_size = int(kernel_size)
        self.max_rad = math.radians(float(max_rot_deg))
        # Eq. (7): phi_m encodes the adjacent camera-token descriptor q_i.
        self.desc_proj = nn.Sequential(
            nn.LayerNorm(desc_dim * 4), nn.Linear(desc_dim * 4, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        # Eq. (8): phi_v turns pooled adjacent frame tokens into the query used
        # to retrieve pair-specific evidence from [G_(i-1); G_i].
        self.frame_query_proj = nn.Sequential(
            nn.LayerNorm(frame_dim * 4), nn.Linear(frame_dim * 4, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.frame_proj = nn.Linear(frame_dim, hidden_dim)
        self.frame_role_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.frame_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.frame_dropout = nn.Dropout(0.0)
        self.frame_norm = nn.LayerNorm(hidden_dim)
        # Eq. (9): phi_fuse combines motion and visual evidence into f_i.
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.age_embed = nn.Embedding(self.kernel_size, hidden_dim)
        # Eq. (10): T_h and T_g are depthwise temporal convolutions over W_i.
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, self.kernel_size, groups=hidden_dim)
        self.gate = nn.Conv1d(hidden_dim, hidden_dim, self.kernel_size, groups=hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)
        self.reset_parameters()

    def reset_parameters(self):
        """Match the released head initialization, including zero residual output."""

        for sequence in (self.desc_proj, self.frame_query_proj, self.fuse_proj):
            for module in sequence:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.frame_proj.weight)
        nn.init.zeros_(self.frame_proj.bias)
        nn.init.normal_(self.frame_role_embed, std=0.02)
        nn.init.xavier_uniform_(self.frame_attn.in_proj_weight)
        nn.init.zeros_(self.frame_attn.in_proj_bias)
        nn.init.xavier_uniform_(self.frame_attn.out_proj.weight)
        nn.init.zeros_(self.frame_attn.out_proj.bias)
        nn.init.normal_(self.age_embed.weight, std=0.02)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv.bias)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, prev_desc, curr_desc, prev_tokens, curr_tokens, buffer=None):
        """Predict Eq. (10) ``delta_omega_i`` for one adjacent frame pair."""

        # Eqs. (3) and (7): q_i = [z_(i-1), z_i, z_i-z_(i-1),
        # z_(i-1) odot z_i], followed by the motion MLP phi_m.
        pair = torch.cat((prev_desc, curr_desc, curr_desc - prev_desc,
                          curr_desc * prev_desc), dim=-1).float()
        desc_feature = self.desc_proj(pair)
        # Eq. (8): mean-pool dense tokens, construct the same relational
        # descriptor, and map it to a cross-attention query.
        prev_mean, curr_mean = prev_tokens.float().mean(1), curr_tokens.float().mean(1)
        frame_pair = torch.cat((prev_mean, curr_mean, curr_mean - prev_mean,
                                curr_mean * prev_mean), dim=-1)
        query = self.frame_query_proj(frame_pair)[:, None]
        memory = self.frame_proj(torch.cat((prev_tokens, curr_tokens), dim=1).float())
        split = prev_tokens.shape[1]
        memory = memory + torch.cat((
            self.frame_role_embed[0].expand(split, -1),
            self.frame_role_embed[1].expand(curr_tokens.shape[1], -1)), dim=0)[None]
        # Eq. (8): f_v = CrossAttn(phi_v(R(Gbar_(i-1),Gbar_i)),
        #                          [G_(i-1);G_i]).
        context, _ = self.frame_attn(query, memory, memory, need_weights=False)
        # Eq. (9): f_i = phi_fuse([f_m; f_v]).
        fused = self.fuse_proj(torch.cat((
            desc_feature,
            self.frame_norm(query + self.frame_dropout(context))[:, 0]), -1))
        # Eq. (10): W_i stores f_(i-K+1)..f_i.  Left zero padding makes the
        # first K-1 steps causal without inventing earlier observations.
        buffer = fused[:, None] if buffer is None else torch.cat((buffer, fused[:, None]), 1)
        buffer = buffer[:, -self.kernel_size:]
        pad = self.kernel_size - buffer.shape[1]
        window = F.pad(buffer, (0, 0, pad, 0))
        ages = torch.arange(self.kernel_size - 1, -1, -1, device=window.device)
        valid = (torch.arange(self.kernel_size, device=window.device) >= pad).to(window.dtype)
        window = window + self.age_embed(ages)[None] * valid[None, :, None]
        temporal = window.transpose(1, 2).float()
        # Eq. (10): T_h(W_i) odot sigmoid(T_g(W_i)).
        hidden = (self.conv(temporal) * torch.sigmoid(self.gate(temporal))).squeeze(-1)
        # phi_o produces an axis-angle residual.  The tanh and two-degree bound
        # come from the released implementation and are more specific than Eq. (10).
        residual = self.max_rad * torch.tanh(self.out(hidden))
        return residual.to(curr_desc.dtype), buffer


class AdjacentPoseHead(nn.Module):
    """Camera-token pose head implementing report Eqs. (2)-(5)."""

    def __init__(self, dim=512, hidden_dim=512, pair_hidden_dim=512,
                 num_pose_tokens=5, rot_correction_kernel=10,
                 rot_correction_max_deg=2.0, init_std=1e-4,
                 enable_rotation_refiner=True, use_checkpoint=False):
        super().__init__()
        self.num_pose_tokens = int(num_pose_tokens)
        self.enable_rotation_refiner = bool(enable_rotation_refiner)
        self.use_checkpoint = bool(use_checkpoint)
        # Eq. (2): the shared phi_desc MLP is applied to every camera token.
        self.frame_descriptor = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        # Eqs. (3)-(4): encode the relational descriptor q_i before separate
        # direct-translation and scalar-last quaternion regressors.
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, pair_hidden_dim), nn.ReLU(),
            nn.Linear(pair_hidden_dim, pair_hidden_dim), nn.ReLU())
        self.delta_t_head = nn.Linear(pair_hidden_dim, 3)
        self.delta_q_head = nn.Linear(pair_hidden_dim, 4)
        self.rot_correction = TemporalRotationRefiner(
            hidden_dim, dim, hidden_dim, rot_correction_kernel,
            rot_correction_max_deg, num_heads=8)
        for sequence in (self.frame_descriptor, self.pair_mlp):
            for module in sequence:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        self.rot_correction.reset_parameters()
        nn.init.normal_(self.delta_t_head.weight, std=init_std)
        nn.init.zeros_(self.delta_t_head.bias)
        nn.init.normal_(self.delta_q_head.weight, std=init_std)
        nn.init.zeros_(self.delta_q_head.bias)
        with torch.no_grad():
            self.delta_q_head.bias[-1] = 1.0

    @staticmethod
    def quat_to_mat(q, eps=1e-8):
        """Convert the Eq. (4) scalar-last quaternion output to SO(3)."""

        q = F.normalize(q, dim=-1, eps=eps)
        x, y, z, w = q.unbind(-1)
        s = 2.0 / q.square().sum(-1).clamp_min(eps)
        return torch.stack((
            1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
            s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
            s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)

    @staticmethod
    def rotvec_to_mat(v, eps=1e-8):
        """Rodrigues exponential map ``Exp([delta_omega]_x)`` from Eq. (11)."""

        theta2 = v.square().sum(-1, keepdim=True)
        theta = theta2.clamp_min(eps * eps).sqrt()
        a = torch.where(theta2 < eps*eps, 1-theta2/6, torch.sin(theta)/theta)
        b = torch.where(theta2 < eps*eps, .5-theta2/24,
                        (1-torch.cos(theta))/theta2.clamp_min(eps*eps))
        x, y, z = v.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero), -1).reshape(*v.shape[:-1],3,3)
        eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(skew)
        return eye + a[..., None] * skew + b[..., None] * (skew @ skew)

    def _delta(self, prev, curr):
        """Regress the unrefined adjacent transform in Eqs. (3)-(4)."""

        # Eq. (3): R(x,y) = [x, y, y-x, x odot y].
        pair = torch.cat((prev, curr, curr-prev, curr*prev), -1)
        hidden = self.pair_mlp(pair)
        # Eq. (4): Head_pose(q_i) predicts translation and rotation for
        # T_(i-1<-i).  Translation is expressed in frame i-1 coordinates.
        t = self.delta_t_head(hidden.float()).to(curr.dtype)
        r = self.quat_to_mat(self.delta_q_head(hidden.float())).to(curr.dtype)
        delta = torch.zeros((*curr.shape[:-1], 4, 4), device=curr.device, dtype=curr.dtype)
        delta[..., :3, :3], delta[..., :3, 3], delta[..., 3, 3] = r, t, 1
        return delta

    def forward(self, features) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the Eq. (5) composed trajectory and every adjacent increment."""

        if features.ndim != 4 or features.shape[2] <= self.num_pose_tokens:
            raise ValueError("Expected pose features [B,N,T,C] with image tokens")
        b, n, _, c = features.shape
        pose_tokens = features[:, :, :self.num_pose_tokens]
        # Eq. (2): z_i = mean_l phi_desc(c_i^l).
        flat_pose_tokens = pose_tokens.reshape(-1, self.num_pose_tokens, c)
        if self.use_checkpoint and self.training:
            desc = checkpoint(
                self.frame_descriptor, flat_pose_tokens, use_reentrant=False
            ).mean(1)
        else:
            desc = self.frame_descriptor(flat_pose_tokens).mean(1)
        desc = desc.reshape(b, n, -1)
        frame_tokens = features[:, :, self.num_pose_tokens:]
        identity = torch.eye(4, device=features.device, dtype=features.dtype).expand(b,4,4).clone()
        poses, raw, corrected, residuals = [identity], [], [], []
        buffer = None
        for index in range(1, n):
            # Eq. (4): initial T_(i-1<-i) from adjacent frame descriptors.
            delta_args = (desc[:, index-1], desc[:, index])
            if self.use_checkpoint and self.training:
                delta = checkpoint(self._delta, *delta_args, use_reentrant=False)
            else:
                delta = self._delta(*delta_args)
            if self.enable_rotation_refiner:
                refine_args = (
                    desc[:, index-1], desc[:, index], frame_tokens[:, index-1],
                    frame_tokens[:, index],
                )
                if self.use_checkpoint and self.training:
                    if buffer is None:
                        residual, buffer = checkpoint(
                            self.rot_correction, *refine_args,
                            use_reentrant=False,
                        )
                    else:
                        residual, buffer = checkpoint(
                            self.rot_correction, *refine_args, buffer,
                            use_reentrant=False,
                        )
                else:
                    residual, buffer = self.rot_correction(*refine_args, buffer)
            else:
                residual = desc.new_zeros((b, 3))
            refined = delta.clone()
            # Eq. (11): R_hat_(i-1<-i) = R_tilde_(i-1<-i)
            # Exp([delta_omega_i]_x).  The translation column is left unchanged.
            refined[:, :3, :3] = delta[:, :3, :3] @ self.rotvec_to_mat(residual.float()).to(delta.dtype)
            # Eq. (5): T_(0<-i) = product_(k=1..i) T_(k-1<-k).
            poses.append(poses[-1] @ refined)
            raw.append(delta)
            corrected.append(refined)
            residuals.append(residual)
        empty_pose = features.new_empty((b, 0, 4, 4))
        empty_rot = features.new_empty((b, 0, 3))
        state = {
            "raw_adjacent_poses": torch.stack(raw, 1) if raw else empty_pose,
            "adjacent_poses": torch.stack(corrected, 1) if corrected else empty_pose,
            "rotation_residual": torch.stack(residuals, 1) if residuals else empty_rot,
        }
        return torch.stack(poses, 1), state
