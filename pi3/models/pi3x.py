"""Pi3X inference model with compatibility layers for the training branch.

The public Pi3X model lives on the repository main branch, while the training
branch predates a few of its supporting layers.  The small private classes in
this module preserve the main-branch state-dict layout without requiring edits
to the shared layer package.
"""

from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend
from torch.utils.checkpoint import checkpoint

from .dinov2.hub.backbones import dinov2_vitl14_reg
from .dinov2.layers import Mlp, PatchEmbed
from .dinov2.layers.layer_scale import LayerScale
from .layers.attention import AttentionRope, FlashAttentionRope, FlashCrossAttentionRope
from .layers.block import BlockRope
from .layers.camera_head import CameraHead
from .layers.pos_embed import PositionGetter, RoPE2D
from .layers.transformer_head import TransformerDecoder
from ..utils.geometry import get_pixel, homogenize_points, se3_inverse


def _normalized_view_plane_uv(width, height, aspect_ratio=None, dtype=None, device=None):
    """Build the aspect-ratio-aware UV channels used by the released ConvHead."""
    aspect_ratio = width / height if aspect_ratio is None else aspect_ratio
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack((u, v), dim=-1)


class _ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, padding_mode="replicate", activation="relu", norm="group_norm"):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        activations = {
            "relu": lambda: nn.ReLU(inplace=True),
            "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True),
            "silu": lambda: nn.SiLU(inplace=True),
            "elu": lambda: nn.ELU(inplace=True),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function {activation}")
        groups = hidden_channels // 32 if norm == "group_norm" else 1
        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activations[activation](),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(groups, hidden_channels),
            activations[activation](),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, padding_mode=padding_mode),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.skip_connection(x)


class _ConvHead(nn.Module):
    """MoGe-style convolutional head kept local for main-branch compatibility."""
    def __init__(self, dim_in, dim_out, dim_proj=512, dim_upsample=(256, 128, 128), dim_times_res_block_hidden=1, num_res_blocks=1, res_block_norm="group_norm", last_res_blocks=0, last_conv_channels=32, last_conv_size=1, projects=None, using_uv=True, **_):
        super().__init__()
        self.using_uv = using_uv
        self.projects = projects
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2 if using_uv else in_ch, out_ch),
                *(_ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks)),
            )
            for in_ch, out_ch in zip([dim_proj] + list(dim_upsample[:-1]), dim_upsample)
        ])
        self.output_block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_upsample[-1] + (2 if using_uv else 0), last_conv_channels, 3, padding=1, padding_mode="replicate"),
                *(_ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation="relu", norm=res_block_norm) for _ in range(last_res_blocks)),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_conv_channels, out_dim, last_conv_size, padding=last_conv_size // 2, padding_mode="replicate"),
            )
            for out_dim in dim_out
        ])

    @staticmethod
    def _make_upsampler(in_channels, out_channels):
        module = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="replicate"),
        )
        module[0].weight.data[:] = module[0].weight.data[:, :, :1, :1]
        return module

    def _run(self, module, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, hidden_states, image=None, patch_h=None, patch_w=None):
        if image is not None:
            img_h, img_w = image.shape[-2:]
            patch_h, patch_w = img_h // 14, img_w // 14
        else:
            if patch_h is None or patch_w is None:
                raise ValueError("patch_h and patch_w are required when image is omitted")
            img_h, img_w = patch_h * 14, patch_w * 14
        x = self.projects(hidden_states).permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous() if self.projects is not None else hidden_states
        for block in self.upsample_blocks:
            if self.using_uv:
                uv = _normalized_view_plane_uv(x.shape[-1], x.shape[-2], img_w / img_h, x.dtype, x.device)
                x = torch.cat((x, uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)), dim=1)
            for layer in block:
                x = self._run(layer, x)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        if self.using_uv:
            uv = _normalized_view_plane_uv(x.shape[-1], x.shape[-2], img_w / img_h, x.dtype, x.device)
            x = torch.cat((x, uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)), dim=1)
        return [self._run(block, x) for block in self.output_block]


def _invert_se3(transforms):
    """Invert rigid 4 by 4 transforms without a generic matrix inverse."""
    rotation = transforms[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(transforms)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation, transforms[..., :3, 3])
    result[..., 3, 3] = 1
    return result


def _lift_intrinsics(intrinsics):
    result = torch.zeros(intrinsics.shape[:-2] + (4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    result[..., :3, :3] = intrinsics
    result[..., 3, 3] = 1
    return result


def _invert_intrinsics(intrinsics):
    result = torch.zeros_like(intrinsics)
    result[..., 0, 0] = 1 / intrinsics[..., 0, 0]
    result[..., 1, 1] = 1 / intrinsics[..., 1, 1]
    result[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    result[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    result[..., 2, 2] = 1
    return result


def _rope_coefficients(positions, feature_dim):
    frequencies = 100.0 ** (-torch.arange(feature_dim // 2, device=positions.device) / (feature_dim // 2))
    angles = positions[None, None, :, None] * frequencies[None, None, None]
    return angles.cos(), angles.sin()


def _apply_rope(features, coefficients, inverse=False):
    cos, sin = coefficients
    if cos.shape[2] != features.shape[2]:
        repetitions = features.shape[2] // cos.shape[2]
        cos, sin = cos.repeat(1, 1, repetitions, 1), sin.repeat(1, 1, repetitions, 1)
    x, y = features.chunk(2, dim=-1)
    if inverse:
        return torch.cat((cos * x - sin * y, sin * x + cos * y), dim=-1)
    return torch.cat((cos * x + sin * y, -sin * x + cos * y), dim=-1)


def _apply_projection(features, matrix):
    batch, heads, sequence, feature_dim = features.shape
    cameras, dimension = matrix.shape[1], matrix.shape[-1]
    shaped = features.reshape(batch, heads, cameras, -1, feature_dim // dimension, dimension)
    return torch.einsum("bcij,bncpkj->bncpki", matrix, shaped).reshape(features.shape)


def _prepare_projective_functions(head_dim, viewmats, intrinsics, patches_x, patches_y, image_width, image_height):
    """Create the query, key-value, and output transforms used by PRoPE."""
    batch, cameras = viewmats.shape[:2]
    if intrinsics is not None:
        # PRoPE expresses intrinsics on a centered, resolution-independent image plane.
        normalized = torch.zeros_like(intrinsics)
        normalized[..., 0, 0] = intrinsics[..., 0, 0] / image_width
        normalized[..., 1, 1] = intrinsics[..., 1, 1] / image_height
        normalized[..., 0, 2] = intrinsics[..., 0, 2] / image_width - 0.5
        normalized[..., 1, 2] = intrinsics[..., 1, 2] / image_height - 0.5
        normalized[..., 2, 2] = 1
        projection = _lift_intrinsics(normalized) @ viewmats
        inverse = _invert_se3(viewmats) @ _lift_intrinsics(_invert_intrinsics(normalized))
    else:
        projection, inverse = viewmats, _invert_se3(viewmats)
    xs = torch.arange(patches_x, device=viewmats.device).tile(patches_y * cameras)
    ys = torch.arange(patches_y, device=viewmats.device).repeat_interleave(patches_x).tile(cameras)
    coeff_x, coeff_y = _rope_coefficients(xs, head_dim // 4), _rope_coefficients(ys, head_dim // 4)

    def apply_blocks(features, matrix, invert_rope=False):
        # Half of each head carries projective geometry. The remaining quarters
        # carry horizontal and vertical rotary coordinates respectively.
        projective, horizontal, vertical = torch.split(features, (head_dim // 2, head_dim // 4, head_dim // 4), dim=-1)
        return torch.cat((_apply_projection(projective, matrix), _apply_rope(horizontal, coeff_x, invert_rope), _apply_rope(vertical, coeff_y, invert_rope)), dim=-1)

    return (
        partial(apply_blocks, matrix=projection.transpose(-1, -2)),
        partial(apply_blocks, matrix=inverse),
        partial(apply_blocks, matrix=projection, invert_rope=True),
    )


class _ProjectiveAttention(AttentionRope):
    """Self-attention whose Q, K, V and output share one projective frame."""
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(dim, num_heads=num_heads, **kwargs)
        self.head_dim = dim // num_heads

    def forward(self, x, extrinsics, height, width, patch_h, patch_w, intrinsics=None, attn_mask=None):
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).reshape(batch, sequence, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = (qkv[:, :, index] for index in range(3))
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        apply_q, apply_kv, apply_output = _prepare_projective_functions(self.head_dim, extrinsics, intrinsics, patch_w, patch_h, width, height)
        q, k, v = apply_q(q), apply_kv(k), apply_kv(v)
        if q.dtype == torch.bfloat16 and q.is_cuda and attn_mask is None:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = F.scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        output = apply_output(output).transpose(1, 2).reshape(batch, sequence, channels)
        return self.proj_drop(self.proj(output))


class _PoseInjectBlock(nn.Module):
    """Inject camera-relative projective attention into image patch tokens."""
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, proj_bias=True, ffn_bias=True, drop=0, attn_drop=0, init_values=None, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, ffn_layer=Mlp, qk_norm=False, **_):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _ProjectiveAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, rope=None)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, poses, height, width, patch_h, patch_w, intrinsics=None, connect=False, attn_mask=None):
        residual = self.ls1(self.attn(self.norm1(x), se3_inverse(poses), height, width, patch_h, patch_w, intrinsics, attn_mask))
        residual = residual + self.ls2(self.mlp(self.norm2(x)))
        return x + residual if connect else residual


class _CrossOnlyBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, proj_bias=True, ffn_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm, ffn_layer=Mlp, init_values=None, qk_norm=False, rope=None, **_):
        super().__init__()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_y = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2, self.norm_y, self.norm3 = norm_layer(dim), norm_layer(dim), norm_layer(dim)
        self.cross_attn = FlashCrossAttentionRope(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, rope=rope, qk_norm=qk_norm)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias)

    def forward(self, x, context, xpos=None, ypos=None):
        context = self.norm_y(context)
        x = x + self.ls_y(self.cross_attn(self.norm2(x), context, context, qpos=xpos, kpos=ypos))
        return x + self.ls2(self.mlp(self.norm3(x)))


class _ContextOnlyTransformerDecoder(nn.Module):
    """Cross-attention-only decoder used to regress one metric scale per scene."""
    def __init__(self, in_dim, out_dim, dec_embed_dim=512, depth=5, dec_num_heads=8, mlp_ratio=4, rope=None, prenorm=False, use_checkpoint=True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_dim) if prenorm else None
        self.projects_x, self.projects_y = nn.Linear(in_dim, dec_embed_dim), nn.Linear(in_dim, dec_embed_dim)
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([_CrossOnlyBlock(dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, proj_bias=True, ffn_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), qk_norm=False, rope=rope) for _ in range(depth)])
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, context, xpos=None, ypos=None):
        if self.pre_norm is not None:
            hidden, context = self.pre_norm(hidden), self.pre_norm(context)
        hidden, context = self.projects_x(hidden), self.projects_y(context)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                hidden = checkpoint(block, hidden, context, xpos=xpos, ypos=ypos, use_reentrant=False)
            else:
                hidden = block(hidden, context, xpos=xpos, ypos=ypos)
        return self.linear_out(hidden)


class Pi3X(nn.Module, PyTorchModelHubMixin):
    """Released Pi3X inference network with optional depth, ray and pose priors.

    Images have shape B by N by 3 by H by W and values in the zero-to-one
    range. Poses follow the OpenCV camera-to-world convention. Image dimensions
    must be divisible by the fourteen-pixel DINO patch size.
    """

    def __init__(self, ckpt=None, use_multimodal=True, head_chunk_size=64):
        super().__init__()
        self.use_multimodal = use_multimodal
        self.head_chunk_size = head_chunk_size
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token
        self.rope = RoPE2D(freq=100)
        self.position_getter = PositionGetter()
        self.dec_embed_dim = 1024
        self.decoder = nn.ModuleList([BlockRope(dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, ffn_layer=Mlp, init_values=0.01, qk_norm=True, attn_class=FlashAttentionRope, rope=self.rope) for _ in range(36)])
        self.patch_start_idx = 5
        self.register_token = nn.Parameter(torch.randn(1, 1, 5, 1024))
        nn.init.normal_(self.register_token, std=1e-6)
        if use_multimodal:
            from copy import deepcopy
            self.depth_encoder = deepcopy(self.encoder)
            del self.depth_encoder.patch_embed
            self.depth_encoder.patch_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            self.depth_emb = nn.Parameter(torch.zeros(1, 1, 1024))
            self.ray_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            nn.init.constant_(self.ray_embed.proj.weight, 0)
            nn.init.constant_(self.ray_embed.proj.bias, 0)
            self.pose_inject_blk = nn.ModuleList([_PoseInjectBlock(dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.01, qk_norm=True) for _ in range(5)])
        self.point_decoder = TransformerDecoder(in_dim=2048, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope)
        self.point_head = _ConvHead(num_features=4, dim_in=1024, projects=nn.Identity(), dim_out=[2, 1], dim_proj=1024, dim_upsample=[256, 128, 64], dim_times_res_block_hidden=2, num_res_blocks=2, last_res_blocks=0, last_conv_channels=32, last_conv_size=1, using_uv=True)
        self.camera_decoder = TransformerDecoder(in_dim=2048, dec_embed_dim=1024, dec_num_heads=16, out_dim=512, rope=self.rope)
        self.camera_head = CameraHead(dim=512)
        self.metric_token = nn.Parameter(torch.randn(1, 1, 2048))
        self.metric_decoder = _ContextOnlyTransformerDecoder(in_dim=2048, dec_embed_dim=512, dec_num_heads=8, out_dim=512, rope=self.rope)
        self.metric_head = nn.Linear(512, 1)
        nn.init.normal_(self.metric_token, std=1e-6)
        self.conf_decoder = TransformerDecoder(in_dim=2048, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope)
        self.conf_head = _ConvHead(num_features=4, dim_in=1024, projects=nn.Identity(), dim_out=[1], dim_proj=1024, dim_upsample=[256, 128, 64], dim_times_res_block_hidden=2, num_res_blocks=2, last_res_blocks=0, last_conv_channels=32, last_conv_size=1, using_uv=True)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if ckpt is not None:
            self.load_checkpoint(ckpt)

    def load_checkpoint(self, checkpoint_path, strict=False):
        """Load safetensors, raw PyTorch state dicts, or common wrapped states."""
        path = Path(checkpoint_path)
        state = load_file(str(path), device="cpu") if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint {path} does not contain a state dict")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        result = self.load_state_dict(state, strict=strict)
        print(f"[Pi3X] Loaded checkpoint {path} with {result}")
        return result

    def disable_multimodal(self, free_cuda_cache=True):
        """Release condition encoders when RGB-only inference is requested."""
        self.use_multimodal = False
        for name in ("depth_encoder", "depth_emb", "ray_embed", "pose_inject_blk"):
            if hasattr(self, name):
                delattr(self, name)
        if free_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def normalize_depth(depths, method="median"):
        """Normalize valid positive depths independently for every batch item."""
        if not isinstance(depths, torch.Tensor):
            depths = torch.as_tensor(depths, dtype=torch.float32)
        if method not in ("median", "mean"):
            raise ValueError(f"Invalid normalization method {method}")
        valid = torch.where(depths > 0, depths, torch.nan).reshape(depths.shape[0], -1)
        factors = torch.nanmedian(valid, dim=1).values if method == "median" else torch.nanmean(valid, dim=1)
        factors = torch.nan_to_num(factors, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-8)
        return depths / factors[:, None, None, None], factors

    @staticmethod
    def _condition_mask(mask, batch, views, probability, device):
        if mask is None:
            return torch.rand((batch, views), device=device) <= probability
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
        if mask.shape != (batch, views):
            raise ValueError(f"Condition mask must have shape {(batch, views)}, got {tuple(mask.shape)}")
        return mask.clone()

    def encode(self, imgs, with_prior=True, depths=None, rays=None, intrinsics=None, poses=None, mask_add_depth=None, mask_add_ray=None, mask_add_pose=None):
        """Encode RGB and selected priors and return their normalization state."""
        batch, views, _, height, width = imgs.shape
        device = imgs.device
        hidden = self.encoder(imgs.reshape(batch * views, 3, height, width), is_training=True)["x_norm_patchtokens"]
        if not self.use_multimodal:
            return hidden, None, None, None, None
        probability = 1.0 if with_prior is True else 0.0
        if depths is None:
            depths, depth_probability = torch.zeros((batch, views, height, width), device=device), 0.0
        else:
            depths, depth_probability = depths.to(device), probability
        has_ray_geometry = rays is not None or intrinsics is not None
        if rays is not None:
            rays = rays.to(device)
            rays = rays[..., :2] / rays[..., 2:3].clamp_min(1e-6)
            ray_probability = probability
        elif intrinsics is not None:
            pixels = torch.from_numpy(get_pixel(height, width).T.reshape(height, width, 3)).to(device=device, dtype=torch.float32)
            rays = torch.einsum("bnij,hwj->bnhwi", torch.linalg.inv(intrinsics.float()), pixels)[..., :2]
            ray_probability = probability
        else:
            rays, ray_probability = torch.zeros((batch, views, height, width, 2), device=device), 0.0
        if poses is None:
            poses, pose_probability = torch.eye(4, device=device)[None, None].repeat(batch, views, 1, 1), 0.0
        else:
            poses, pose_probability = poses.to(device), probability
            if not has_ray_geometry:
                raise ValueError("Pose conditioning requires intrinsics or rays")
        depth_mask = self._condition_mask(mask_add_depth, batch, views, depth_probability, device)
        ray_mask = self._condition_mask(mask_add_ray, batch, views, ray_probability, device)
        pose_mask = self._condition_mask(mask_add_pose, batch, views, pose_probability, device)
        pose_mask[pose_mask.sum(dim=1) == 1] = False
        normalized_depths, depth_scale = self.normalize_depth(depths, method="mean")
        relative_poses = torch.einsum("bij,bnjk->bnik", se3_inverse(poses[:, 0]), poses)
        relative_poses[..., :3, 3] /= depth_scale[:, None, None]
        batches_without_depth = depth_mask.sum(dim=1) == 0
        if batches_without_depth.any() and views > 1:
            # Without a depth prior, mean camera translation supplies the
            # scale gauge. Static sequences retain unit scale.
            translation_scale = relative_poses[:, 1:, :3, 3].norm(dim=-1)
            moving = batches_without_depth & (translation_scale.max(dim=1).values >= 2e-2)
            translation_scale = translation_scale.mean(dim=1).clamp_min(1e-8)
            relative_poses[moving, :, :3, 3] /= translation_scale[moving, None, None]
            normalized_depths[moving] /= translation_scale[moving, None, None, None]
            depth_scale[moving] *= translation_scale[moving]
        depth_input = normalized_depths.reshape(batch * views, 1, height, width)
        depth_valid = (depth_input > 0).to(depth_input.dtype)
        if depth_mask.any():
            depth_embedding = self.depth_encoder(torch.cat((depth_input, depth_valid), dim=1), is_training=True)["x_norm_patchtokens"] + self.depth_emb
        else:
            depth_embedding = torch.zeros_like(hidden)
        ray_embedding = self.ray_embed(rays.reshape(batch * views, height, width, 2).permute(0, 3, 1, 2)) if ray_mask.any() else torch.zeros_like(hidden)
        hidden = hidden + depth_embedding * depth_mask.reshape(batch * views, 1, 1)
        hidden = hidden + ray_embedding * ray_mask.reshape(batch * views, 1, 1)
        return hidden, relative_poses, depth_mask, pose_mask, depth_scale

    def decode(self, hidden, views, height, width, poses, pose_mask):
        """Alternate frame and global blocks and inject pose after five stages."""
        if hidden.ndim == 4:
            batch, views, tokens = hidden.shape[:3]
        else:
            tokens, batch = hidden.shape[1], hidden.shape[0] // views
        hidden = hidden.reshape(batch * views, tokens, -1)
        registers = self.register_token.repeat(batch, views, 1, 1).reshape(batch * views, self.patch_start_idx, -1)
        hidden = torch.cat((registers, hidden), dim=1)
        tokens = hidden.shape[1]
        pos = self.position_getter(batch * views, height // self.patch_size, width // self.patch_size, hidden.device)
        special_pos = torch.zeros(batch * views, self.patch_start_idx, 2, dtype=pos.dtype, device=pos.device)
        pos = torch.cat((special_pos, pos + 1), dim=1)
        pose_block_index = 0
        pose_enabled = self.use_multimodal and pose_mask is not None and pose_mask.any()
        pose_attention_mask = None
        if pose_enabled and not pose_mask.all():
            # A pair participates in projective attention only when both views
            # expose pose conditioning. Register tokens never enter this mask.
            view_mask = pose_mask.unsqueeze(2) & pose_mask.unsqueeze(1)
            patch_tokens = tokens - self.patch_start_idx
            pose_attention_mask = view_mask.repeat_interleave(patch_tokens, 1).repeat_interleave(patch_tokens, 2)[:, None]
        penultimate = None
        for index, block in enumerate(self.decoder):
            if index % 2 == 0:
                hidden, pos = hidden.reshape(batch * views, tokens, -1), pos.reshape(batch * views, tokens, -1)
            else:
                hidden, pos = hidden.reshape(batch, views * tokens, -1), pos.reshape(batch, views * tokens, -1)
            hidden = block(hidden, xpos=pos)
            if pose_enabled and index in (1, 9, 17, 25, 33):
                hidden = hidden.reshape(batch, views, tokens, -1)
                patches = hidden[..., self.patch_start_idx:, :].reshape(batch, views * (tokens - self.patch_start_idx), -1)
                pose_features = self.pose_inject_blk[pose_block_index](patches, poses, height, width, height // 14, width // 14, attn_mask=pose_attention_mask)
                hidden[..., self.patch_start_idx:, :] += pose_features.reshape(batch, views, -1, 1024) * pose_mask[:, :, None, None]
                hidden = hidden.reshape(batch, views * tokens, -1)
                pose_block_index += 1
            if index == len(self.decoder) - 2:
                penultimate = hidden.reshape(batch * views, tokens, -1)
        final = hidden.reshape(batch * views, tokens, -1)
        return torch.cat((penultimate, final), dim=-1), pos.reshape(batch * views, tokens, -1)

    def _chunked_conv_head(self, head, features, patch_h, patch_w):
        """Bound peak activation memory while preserving per-frame outputs."""
        if features.shape[0] <= self.head_chunk_size:
            return head(features, patch_h=patch_h, patch_w=patch_w)
        chunks = [head(features[start:start + self.head_chunk_size], patch_h=patch_h, patch_w=patch_w) for start in range(0, features.shape[0], self.head_chunk_size)]
        return [torch.cat([chunk[index] for chunk in chunks], dim=0) for index in range(len(chunks[0]))]

    def forward_head(self, hidden, pos, batch, views, height, width, patch_h, patch_w):
        """Decode geometry, confidence, camera poses and absolute scene scale."""
        tokens = patch_h * patch_w + self.patch_start_idx
        point_features = self.point_decoder(hidden, xpos=pos)[:, self.patch_start_idx:].float()
        camera_features = self.camera_decoder(hidden, xpos=pos)
        metric_features = self.metric_decoder(self.metric_token.repeat(batch, 1, 1), hidden.reshape(batch, views * tokens, -1), xpos=pos.reshape(batch, views * tokens, -1)[:, :1], ypos=pos.reshape(batch, views * tokens, -1))
        confidence_features = self.conf_decoder(hidden, xpos=pos)[:, self.patch_start_idx:].float()
        with torch.amp.autocast(device_type="cuda", enabled=False):
            xy, z = self._chunked_conv_head(self.point_head, point_features, patch_h, patch_w)
            xy = xy.permute(0, 2, 3, 1).reshape(batch, views, height, width, 2)
            z = z.permute(0, 2, 3, 1).reshape(batch, views, height, width, 1).clamp(max=15).exp()
            local_points = torch.cat((xy * z, z), dim=-1)
            rays = F.normalize(torch.cat((xy, torch.ones_like(z)), dim=-1), dim=-1)
            camera_poses = self.camera_head(camera_features[:, self.patch_start_idx:].float(), patch_h, patch_w).reshape(batch, views, 4, 4)
            metric = self.metric_head(metric_features.float()).reshape(batch).exp()
            confidence = self._chunked_conv_head(self.conf_head, confidence_features, patch_h, patch_w)[0]
            confidence = confidence.permute(0, 2, 3, 1).reshape(batch, views, height, width, 1)
            # The point and pose heads predict in the normalized scene gauge.
            # The metric head converts all translations and points together.
            points = torch.einsum("bnij,bnhwj->bnhwi", camera_poses, homogenize_points(local_points))[..., :3] * metric[:, None, None, None, None]
            camera_poses = camera_poses.clone()
            camera_poses[..., :3, 3] *= metric[:, None, None]
            local_points = local_points * metric[:, None, None, None, None]
        return {"points": points, "local_points": local_points, "rays": rays, "conf": confidence, "camera_poses": camera_poses, "metric": metric}

    def forward(self, imgs, depths=None, intrinsics=None, rays=None, poses=None, with_prior=True, mask_add_depth=None, mask_add_ray=None, mask_add_pose=None):
        """Run Pi3X inference and return metric geometry in a shared world frame."""
        if imgs.ndim != 5 or imgs.shape[2] != 3:
            raise ValueError(f"imgs must have shape B,N,3,H,W, got {tuple(imgs.shape)}")
        batch, views, _, height, width = imgs.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Image size {(height, width)} must be divisible by patch size {self.patch_size}")
        normalized = (imgs - self.image_mean) / self.image_std
        hidden, relative_poses, depth_mask, pose_mask, depth_scale = self.encode(normalized, with_prior, depths, rays, intrinsics, poses, mask_add_depth, mask_add_ray, mask_add_pose)
        hidden, pos = self.decode(hidden.reshape(batch, views, -1, self.dec_embed_dim), views, height, width, relative_poses, pose_mask)
        return self.forward_head(hidden, pos, batch, views, height, width, height // 14, width // 14)


__all__ = ["Pi3X"]
