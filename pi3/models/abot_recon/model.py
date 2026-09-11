"""ABot-Recon local-context reconstruction backbone.

The module follows Secs. 3.1-3.2 of the technical report.  In particular,
``forward`` implements the encoder, decoder, point head, and confidence head in
Eq. (1), while ``_window_attention`` implements the fixed causal memory in
Eq. (6).  The adjacent-pose equations are isolated in ``pose_head.py`` so the
network and geometric parameterization remain easy to audit independently.

This file is the differentiable full-clip path used for training.  It evaluates
the same K-frame causal dependency for every frame, but it does not implement
the released frame-by-frame paged-KV runtime.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from pi3.models.pi3_training import Pi3, freeze_all_params
from pi3.models.layers.transformer_head import LinearPts3d
from pi3.utils.geometry import homogenize_points

from .pose_head import AdjacentPoseHead
from .rope3d import RoPE3D, apply_rope3d


def _checkpoint_state(path: str):
    """Extract and normalize a model state dict from supported containers."""

    if str(path).lower().endswith(".safetensors"):
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {path} does not contain a model state dict")

    state = dict(state)
    removed_prefix = True
    while removed_prefix:
        removed_prefix = False
        for prefix in ("module.", "_orig_mod."):
            if state and all(key.startswith(prefix) for key in state):
                state = {key[len(prefix):]: value for key, value in state.items()}
                removed_prefix = True
    return state


def _pi3_compatible_state(checkpoint_state, model_state):
    """Keep only shape-compatible Pi3 representation and geometry weights."""

    prefixes = (
        "encoder.",
        "decoder.",
        "point_decoder.",
        "point_head.",
        "camera_decoder.",
        "register_token",
        "image_mean",
        "image_std",
    )
    compatible = {}
    for key, value in checkpoint_state.items():
        if not key.startswith(prefixes) or key not in model_state:
            continue
        if (
            hasattr(value, "shape")
            and tuple(value.shape) == tuple(model_state[key].shape)
        ):
            compatible[key] = value
    return compatible


class ABotRecon(Pi3):
    """Trainable clip implementation of ABot-Recon Eqs. (1)-(6).

    It preserves Pi3/VGGT encoder and decoder parameter names.  Every odd decoder
    block performs Eq. (6) causal attention over at most
    ``local_window_frames`` frames.  Even blocks remain independent intra-frame
    attention as shown in Fig. 3.  PyTorch SDPA keeps the training path free of a
    FlashInfer dependency.
    """
    def __init__(
        self,
        pos_type="rope100",
        decoder_size="large",
        load_vggt=False,
        load_pi3=None,
        ckpt=None,
        freeze_encoder=True,
        enable_confidence=None,
        confidence_only_train=False,
        enable_rotation_refiner=True,
        adj_pose_head_to_ckpt=False,
        local_window_frames=12,
        num_dec_blk_not_to_checkpoint=4,
        point_z_log_max=10.0,
        global_pos_encoding="rope3d",
        rope3d_fhw_dim=(20, 22, 22),
        max_frames=4096,
        gate_layers=None,
    ):
        if load_pi3 is not None and ckpt is not None:
            raise ValueError("Pi3 initialization and ckpt are mutually exclusive")
        if load_pi3 is not None and load_vggt:
            raise ValueError("load_pi3 and load_vggt are mutually exclusive")

        pi3_state = _checkpoint_state(load_pi3) if load_pi3 is not None else None
        model_state = _checkpoint_state(ckpt) if ckpt is not None else None
        if enable_confidence is None:
            enable_confidence = bool(model_state) and any(
                key.startswith(("conf_decoder.", "conf_head."))
                for key in model_state)
        super().__init__(
            pos_type=pos_type,
            decoder_size=decoder_size,
            load_vggt=load_vggt,
            freeze_encoder=freeze_encoder,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=num_dec_blk_not_to_checkpoint,
            ckpt=None,
        )
        if int(local_window_frames) <= 0:
            raise ValueError("local_window_frames must be positive")
        self.local_window_frames = int(local_window_frames)
        self.point_z_log_max = float(point_z_log_max)
        self.global_pos_encoding = str(global_pos_encoding).lower()
        if self.global_pos_encoding not in ("rope3d", "pi3_2d"):
            raise ValueError("global_pos_encoding must be rope3d or pi3_2d")
        head_dim = self.dec_embed_dim // self.decoder[0].attn.num_heads
        self.rope3d = RoPE3D(
            head_dim=head_dim, max_seq_len=max_frames,
            fhw_dim=tuple(rope3d_fhw_dim)) if self.global_pos_encoding == "rope3d" else None
        self.enable_confidence = bool(enable_confidence)
        self.confidence_only_train = bool(confidence_only_train)
        # Sec. 3.2 replaces Pi3's absolute camera head with Eqs. (2)-(5), which
        # regress and compose adjacent-frame relative transforms.
        self.camera_head = AdjacentPoseHead(
            dim=512,
            hidden_dim=512,
            pair_hidden_dim=512,
            num_pose_tokens=5,
            rot_correction_kernel=10,
            rot_correction_max_deg=2.0,
            enable_rotation_refiner=enable_rotation_refiner,
            use_checkpoint=adj_pose_head_to_ckpt,
        )
        if not enable_rotation_refiner:
            freeze_all_params([self.camera_head.rot_correction])
        selected_gates = range(len(self.decoder)) if gate_layers is None else gate_layers
        self.gate_layers = tuple(int(index) for index in selected_gates)
        for index in self.gate_layers:
            if index < 0 or index >= len(self.decoder):
                raise ValueError(f"gate layer {index} is outside the decoder")
            self.decoder[index].attn.gate_proj = torch.nn.Linear(
                self.dec_embed_dim, self.dec_embed_dim, bias=False)
        if self.enable_confidence:
            # Eq. (1) predicts S_i from a confidence branch cloned from the
            # local-point decoder.  Sec. 4.1 trains this branch alone in Stage III.
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(
                patch_size=14, dec_embed_dim=1024, output_dim=1)
        if pi3_state is not None:
            compatible = _pi3_compatible_state(pi3_state, self.state_dict())
            if not compatible:
                raise RuntimeError(f"No compatible Pi3 weights found in {load_pi3}")
            result = self.load_state_dict(compatible, strict=False)
            print(
                f"[ABot-Recon] loaded {len(compatible)}/{len(pi3_state)} "
                f"compatible Pi3 tensors from {load_pi3}: {result}",
                flush=True,
            )
        if model_state is not None:
            required_abot_keys = (
                "camera_head.delta_t_head.weight",
                "camera_head.delta_q_head.weight",
            )
            missing_abot_keys = [
                key for key in required_abot_keys if key not in model_state
            ]
            if missing_abot_keys:
                raise RuntimeError(
                    f"ckpt {ckpt} is not an ABot-Recon checkpoint; "
                    f"missing {missing_abot_keys}. Use load_pi3 for Pi3 weights."
                )
            result = self.load_state_dict(model_state, strict=False)
            checkpoint_has_confidence = any(
                key.startswith(("conf_decoder.", "conf_head."))
                for key in model_state)
            if self.enable_confidence and not checkpoint_has_confidence:
                self.conf_decoder.load_state_dict(
                    self.point_decoder.state_dict(), strict=True)
                print(
                    "[ABot-Recon] initialized conf_decoder from the loaded point_decoder",
                    flush=True)
            print(f"[ABot-Recon] loaded model checkpoint {ckpt}: {result}", flush=True)
        if self.confidence_only_train:
            if not self.enable_confidence:
                raise ValueError("confidence_only_train requires enable_confidence")
            freeze_all_params([self])
            for module in (self.conf_decoder, self.conf_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    @staticmethod
    def _attention_output(attention, input_tokens, output):
        """Apply the released elementwise attention gate before projection.

        The gate is a checkpoint-level implementation detail and is not assigned
        a separate equation in the report.
        """

        gate = getattr(attention, "gate_proj", None)
        if gate is not None:
            output = output * torch.sigmoid(gate(input_tokens))
        return attention.proj_drop(attention.proj(output))

    def _local_attention(self, attention, x, xpos):
        """Evaluate the per-frame self-attention blocks illustrated in Fig. 3."""

        batch, tokens, channels = x.shape
        qkv = attention.qkv(x).reshape(
            batch, tokens, 3, attention.num_heads,
            channels // attention.num_heads).transpose(1, 3)
        q, k, v = (qkv[:, :, index] for index in range(3))
        q, k = attention.q_norm(q).to(v.dtype), attention.k_norm(k).to(v.dtype)
        if attention.rope is not None:
            q, k = attention.rope(q, xpos), attention.rope(k, xpos)
        dropout = attention.attn_drop.p if self.training else 0.0
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return ABotRecon._attention_output(attention, x, output)

    def _window_attention(self, attention, x, xpos, spatial_shape=None):
        """Evaluate Eq. (6) with a bounded K-frame causal key/value window.

        ``x`` is ``[B,N,T,C]``.  Query frame ``j`` sees frames
        ``max(0,j-K+1)..j`` and therefore has no dependency on future frames.
        Computing each query against only this slice gives the report's O(NK)
        temporal-attention complexity in the full-clip training path.
        """

        b, n, tokens, channels = x.shape
        flat = x.reshape(b * n, tokens, channels)
        qkv = attention.qkv(flat).reshape(
            b * n, tokens, 3, attention.num_heads,
            channels // attention.num_heads).transpose(1, 3)
        q, k, v = (qkv[:, :, index] for index in range(3))
        q = attention.q_norm(q).to(v.dtype)
        k = attention.k_norm(k).to(v.dtype)
        if self.global_pos_encoding == "rope3d":
            if spatial_shape is None:
                raise ValueError("spatial_shape is required for 3D RoPE")
            frequencies = self.rope3d.grid(
                n, spatial_shape[0], spatial_shape[1], self.patch_start_idx,
                x.device)[None, :, None]
            q = q.reshape(b, n, attention.num_heads, tokens, -1)
            k = k.reshape(b, n, attention.num_heads, tokens, -1)
            q, k = apply_rope3d(q, frequencies), apply_rope3d(k, frequencies)
        elif attention.rope is not None:
            pos = xpos.reshape(b * n, tokens, -1)
            q, k = attention.rope(q, pos), attention.rope(k, pos)
        head_dim = channels // attention.num_heads
        q = q.reshape(b, n, attention.num_heads, tokens, head_dim)
        k = k.reshape(b, n, attention.num_heads, tokens, head_dim)
        v = v.reshape(b, n, attention.num_heads, tokens, head_dim)
        dropout = attention.attn_drop.p if self.training else 0.0
        outputs = []
        for frame in range(n):
            # Eq. (6): M^(K)_(j-1) contains only the most recent K-1 cached
            # frames.  The current frame is appended for self-attention.
            start = max(0, frame + 1 - self.local_window_frames)
            keys = k[:, start:frame + 1].permute(0, 2, 1, 3, 4).reshape(
                b, attention.num_heads, -1, head_dim)
            values = v[:, start:frame + 1].permute(0, 2, 1, 3, 4).reshape(
                b, attention.num_heads, -1, head_dim)
            outputs.append(F.scaled_dot_product_attention(
                q[:, frame], keys, values, dropout_p=dropout))
        output = torch.stack(outputs, 1).transpose(2, 3).reshape(b, n, tokens, channels)
        return ABotRecon._attention_output(attention, x, output)

    def _local_block(self, block, hidden, xpos):
        residual = self._local_attention(block.attn, block.norm1(hidden), xpos)
        hidden = hidden + block.drop_path1(block.ls1(residual))
        return hidden + block.drop_path2(block.ls2(block.mlp(block.norm2(hidden))))

    def _window_global_block(self, block, hidden, xpos, spatial_shape):
        residual = self._window_attention(
            block.attn, block.norm1(hidden), xpos, spatial_shape)
        hidden = hidden + block.drop_path1(block.ls1(residual))
        mlp = block.mlp(block.norm2(hidden))
        return hidden + block.drop_path2(block.ls2(mlp))

    def decode(self, hidden, n, height, width):
        """Alternate Fig. 3 frame attention and Eq. (6) causal attention."""

        b = hidden.shape[0] // n
        hidden = hidden.reshape(b * n, hidden.shape[1], -1)
        registers = self.register_token.repeat(b, n, 1, 1).reshape(
            b * n, self.patch_start_idx, self.dec_embed_dim)
        hidden = torch.cat((registers, hidden), 1)
        tokens = hidden.shape[1]
        pos = self.position_getter(
            b * n, height // self.patch_size, width // self.patch_size,
            hidden.device) + 1
        special_pos = torch.zeros(
            b * n, self.patch_start_idx, 2, device=hidden.device,
            dtype=pos.dtype)
        pos = torch.cat((special_pos, pos), 1).reshape(b, n, tokens, 2)
        final = []
        checkpoint_from = self.num_dec_blk_not_to_checkpoint
        spatial_shape = (height // self.patch_size, width // self.patch_size)
        for index, block in enumerate(self.decoder):
            if index % 2 == 0:
                # Frame-attention blocks process every image independently.
                block_hidden = hidden.reshape(b * n, tokens, -1)
                block_pos = pos.reshape(b * n, tokens, 2)
                if self.training and index >= checkpoint_from:
                    fn = partial(self._local_block, block)
                    hidden = checkpoint(fn, block_hidden, block_pos, use_reentrant=False)
                else:
                    hidden = self._local_block(block, block_hidden, block_pos)
                hidden = hidden.reshape(b, n, tokens, -1)
            else:
                # Causal-attention blocks retain the explicit frame axis so each
                # query can select exactly its Eq. (6) temporal neighborhood.
                hidden = hidden.reshape(b, n, tokens, -1)
                if self.training and index >= checkpoint_from:
                    fn = partial(
                        self._window_global_block, block,
                        spatial_shape=spatial_shape)
                    hidden = checkpoint(fn, hidden, pos, use_reentrant=False)
                else:
                    hidden = self._window_global_block(
                        block, hidden, pos, spatial_shape)
            if index >= len(self.decoder) - 2:
                final.append(hidden.reshape(b * n, tokens, -1))
        return torch.cat(final, -1), pos.reshape(b * n, tokens, 2)

    def forward(self, imgs):
        """Predict Eq. (1) local geometry and Eqs. (4)-(5) camera trajectory.

        Args:
            imgs: RGB sequence ``[B,N,3,H,W]`` in the zero-to-one range.

        Returns:
            Local point maps ``P_i``, optional confidence logits ``S_i``, refined
            adjacent transforms ``T_(i-1<-i)``, and their composed c2w trajectory.
        """

        if imgs.ndim != 5:
            raise ValueError("ABotRecon expects images [B,N,3,H,W]")
        b, n, channels, height, width = imgs.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by 14")
        normalized = (imgs - self.image_mean) / self.image_std
        # Eq. (1): F_i = Encoder(I_i).
        encoded = self.encoder(
            normalized.reshape(b * n, channels, height, width), is_training=True)
        if isinstance(encoded, dict):
            encoded = encoded["x_norm_patchtokens"]
        # Eq. (1): (G_i, C_i, M_i) = Decoder([F_i, C], M_(i-1)).  This
        # full-clip implementation materializes all G_i and C_i while preserving
        # the identical causal dependency through Eq. (6).
        hidden, pos = self.decode(encoded, n, height, width)
        point_hidden = self.point_decoder(hidden, xpos=pos).float()
        camera_hidden = self.camera_decoder(hidden, xpos=pos).float()
        # Eq. (1): P_i = Head_point(G_i).  Following Pi3, the head emits
        # perspective xy factors and log-depth before reconstructing xyz.
        raw_points = self.point_head(
            [point_hidden[:, self.patch_start_idx:]], (height, width)
        ).reshape(b, n, height, width, 3)
        xy, log_z = raw_points.split((2, 1), -1)
        depth = torch.exp(log_z.clamp(max=self.point_z_log_max))
        local_points = torch.cat((xy * depth, depth), -1)
        camera_features = camera_hidden.reshape(b, n, camera_hidden.shape[1], -1)
        # Eqs. (2)-(5): camera tokens produce adjacent transforms, then the
        # transforms are composed from the identity pose of frame zero.
        camera_poses, pose_state = self.camera_head(camera_features)
        # Sec. 3.1 maps each current-camera point map into the accumulated global
        # frame using the composed camera-to-world pose.
        points = torch.einsum(
            "bnij,bnhwj->bnhwi", camera_poses,
            homogenize_points(local_points))[..., :3]
        conf = None
        if self.enable_confidence:
            # Eq. (1): S_i = Head_conf(G_i).  Raw logits are retained because
            # Stage III uses BCE-with-logits supervision.
            conf_hidden = self.conf_decoder(hidden, xpos=pos).float()
            conf = self.conf_head(
                [conf_hidden[:, self.patch_start_idx:]], (height, width)
            ).reshape(b, n, height, width, 1)
        return {
            "points": points,
            "local_points": local_points,
            "camera_poses": camera_poses,
            "adjacent_poses": pose_state["adjacent_poses"],
            "raw_adjacent_poses": pose_state["raw_adjacent_poses"],
            "rotation_residual": pose_state["rotation_residual"],
            "conf": conf,
            "global_points": None,
        }

    @torch.inference_mode()
    def inference(self, imgs):
        return self.forward(imgs)
