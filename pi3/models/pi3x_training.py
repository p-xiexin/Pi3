"""Trainable Pi3X variant built from the released model and dirty trainer code."""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from .pi3x import Pi3X as InferencePi3X
from ..utils.geometry import get_pixel, se3_inverse


def _axis_angle_to_matrix(rotation):
    """Convert batched axis-angle vectors to rotation matrices with Rodrigues."""
    angle = rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rotation / angle
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*rotation.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand_as(skew)
    return identity + angle[..., None].sin() * skew + (1 - angle[..., None].cos()) * (skew @ skew)


def add_randomized_smooth_pose_noise(poses, rotation_std=0.002, translation_std=0.002):
    """Apply a small temporally smooth left-multiplicative SE3 perturbation."""
    if poses.shape[-3] == 0:
        return poses
    shape = poses.shape[:-2]
    rotation_noise = torch.randn(*shape, 3, dtype=poses.dtype, device=poses.device) * rotation_std
    translation_noise = torch.randn(*shape, 3, dtype=poses.dtype, device=poses.device) * translation_std
    rotation_noise = rotation_noise.cumsum(dim=-2)
    translation_noise = translation_noise.cumsum(dim=-2)
    transform = torch.eye(4, dtype=poses.dtype, device=poses.device).expand(*shape, 4, 4).clone()
    transform[..., :3, :3] = _axis_angle_to_matrix(rotation_noise)
    transform[..., :3, 3] = translation_noise
    return transform @ poses


class Pi3X(InferencePi3X):
    """Training variant retaining the conditioning policy from dirty training.

    It keeps geometry in the normalized training gauge so Pi3XLoss can align
    points and supervise the separate metric head without double scaling.
    """

    def __init__(
        self,
        ckpt=None,
        ckpts=None,
        use_multimodal=True,
        checkpoint_strategy="global_only",
        use_checkpoint=None,
        weight_source=None,
        depth_condition_probability=0.3,
        ray_condition_probability=0.5,
        pose_condition_probability=0.2,
        scale_augmentation=(0.8, 1.2),
        pose_noise_probability=0.5,
        teach_datasets=(),
        head_chunk_size=64,
    ):
        super().__init__(ckpt=None, use_multimodal=use_multimodal, head_chunk_size=head_chunk_size)
        if use_checkpoint is False:
            checkpoint_strategy = None
        elif use_checkpoint is True and checkpoint_strategy in (None, "none"):
            checkpoint_strategy = "all"
        if checkpoint_strategy not in (None, "none", "all", "global_only"):
            raise ValueError(f"Unknown checkpoint strategy {checkpoint_strategy}")
        self.checkpoint_strategy = checkpoint_strategy
        self.depth_condition_probability = float(depth_condition_probability)
        self.ray_condition_probability = float(ray_condition_probability)
        self.pose_condition_probability = float(pose_condition_probability)
        self.scale_augmentation = tuple(float(value) for value in scale_augmentation) if scale_augmentation is not None else None
        self.pose_noise_probability = float(pose_noise_probability)
        self.teach_datasets = frozenset(teach_datasets)
        if weight_source is not None and (ckpt is not None or ckpts is not None):
            raise ValueError("weight_source and ckpt are mutually exclusive")
        checkpoint_path = ckpt if ckpt is not None else ckpts
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif weight_source is not None:
            self.load_backbone_weights(weight_source)

    def load_backbone_weights(self, source):
        """Initialize the shared encoder and alternating decoder from a backbone."""
        source = source.lower()
        if source == "vggt":
            path = Path("ckpts/VGGT-1B/model.safetensors")
            if not path.is_file():
                raise FileNotFoundError(path)
            weights = load_file(str(path), device="cpu")
            encoder = {key.removeprefix("aggregator.patch_embed."): value for key, value in weights.items() if key.startswith("aggregator.patch_embed.")}
            decoder = {}
            for key, value in weights.items():
                if key.startswith("aggregator.global_blocks."):
                    suffix = key.removeprefix("aggregator.global_blocks.")
                    index, remainder = suffix.split(".", 1)
                    decoder[f"{int(index) * 2 + 1}.{remainder}"] = value
                elif key.startswith("aggregator.frame_blocks."):
                    suffix = key.removeprefix("aggregator.frame_blocks.")
                    index, remainder = suffix.split(".", 1)
                    decoder[f"{int(index) * 2}.{remainder}"] = value
            print(f"[Pi3X] Loaded VGGT encoder with {self.encoder.load_state_dict(encoder, strict=False)}")
            print(f"[Pi3X] Loaded VGGT decoder with {self.decoder.load_state_dict(decoder, strict=False)}")
            return
        if source == "dino":
            path = Path("ckpts/dinov2_vitl14_reg4_pretrain.pth")
            if not path.is_file():
                raise FileNotFoundError(path)
            weights = torch.load(path, map_location="cpu", weights_only=False)
            print(f"[Pi3X] Loaded DINO encoder with {self.encoder.load_state_dict(weights, strict=False)}")
            blocks = {key.removeprefix("blocks."): value for key, value in weights.items() if key.startswith("blocks.")}
            print(f"[Pi3X] Loaded DINO decoder with {self.decoder.load_state_dict(blocks, strict=False)}")
            return
        if source == "pi3":
            path = Path("ckpts/Pi3/model.safetensors")
            if not path.is_file():
                raise FileNotFoundError(path)
            weights = load_file(str(path), device="cpu")
            encoder = {key.removeprefix("encoder."): value for key, value in weights.items() if key.startswith("encoder.")}
            decoder = {key.removeprefix("decoder."): value for key, value in weights.items() if key.startswith("decoder.")}
            print(f"[Pi3X] Loaded Pi3 encoder with {self.encoder.load_state_dict(encoder, strict=False)}")
            print(f"[Pi3X] Loaded Pi3 decoder with {self.decoder.load_state_dict(decoder, strict=False)}")
            return
        raise ValueError(f"Unsupported weight source {source}")

    def _sample_training_masks(self, batch, views, device, with_prior, mask_add_depth, mask_add_ray, mask_add_pose):
        """Resolve stochastic training priors or deterministic evaluation masks."""
        if with_prior is None:
            probabilities = (self.depth_condition_probability, self.ray_condition_probability, self.pose_condition_probability)
        elif with_prior:
            probabilities = (1.0, 1.0, 1.0)
        else:
            probabilities = (0.0, 0.0, 0.0)
        return tuple(
            self._condition_mask(mask, batch, views, probability, device)
            for mask, probability in zip((mask_add_depth, mask_add_ray, mask_add_pose), probabilities)
        )

    def encode(self, imgs, with_prior=None, depths=None, rays=None, intrinsics=None, poses=None, mask_add_depth=None, mask_add_ray=None, mask_add_pose=None, dataset_names=None):
        """Fuse training priors and retain the sampled masks for loss routing."""
        batch, views, _, height, width = imgs.shape
        device = imgs.device
        hidden = self.encoder(imgs.reshape(batch * views, 3, height, width), is_training=True)["x_norm_patchtokens"]
        if not self.use_multimodal:
            return hidden, None, None, None, None
        depth_mask, ray_mask, pose_mask = self._sample_training_masks(batch, views, device, with_prior, mask_add_depth, mask_add_ray, mask_add_pose)
        if depths is None:
            depths = torch.zeros((batch, views, height, width), dtype=imgs.dtype, device=device)
            depth_mask.zero_()
        else:
            depths = depths.to(device)
        has_ray_geometry = rays is not None or intrinsics is not None
        if rays is not None:
            rays = rays.to(device)
            rays = rays[..., :2] / rays[..., 2:3].clamp_min(1e-6)
        elif intrinsics is not None:
            pixels = torch.from_numpy(get_pixel(height, width).T.reshape(height, width, 3)).to(device=device, dtype=torch.float32)
            rays = torch.einsum("bnij,hwj->bnhwi", torch.linalg.inv(intrinsics.float()), pixels)[..., :2]
        else:
            rays = torch.zeros((batch, views, height, width, 2), dtype=imgs.dtype, device=device)
            ray_mask.zero_()
        if poses is None:
            poses = torch.eye(4, dtype=imgs.dtype, device=device)[None, None].repeat(batch, views, 1, 1)
            pose_mask.zero_()
        else:
            poses = poses.to(device)
            if not has_ray_geometry:
                raise ValueError("Pose conditioning requires intrinsics or rays")
        pose_mask[pose_mask.sum(dim=1) == 1] = False
        if dataset_names is not None:
            for index, name in enumerate(dataset_names):
                if name in self.teach_datasets:
                    depth_mask[index].zero_()
        normalized_depths, depth_scale = self.normalize_depth(depths, method="mean")
        if self.training and self.scale_augmentation is not None:
            # The paired rescaling leaves metric geometry unchanged while
            # preventing the normalized branch from memorizing one gauge.
            low, high = self.scale_augmentation
            augmentation = torch.empty(batch, device=device).uniform_(low, high)
            normalized_depths /= augmentation[:, None, None, None]
            depth_scale *= augmentation
        relative_poses = torch.einsum("bij,bnjk->bnik", se3_inverse(poses[:, 0]), poses)
        relative_poses[..., :3, 3] /= depth_scale[:, None, None]
        no_depth = depth_mask.sum(dim=1) == 0
        if no_depth.any() and views > 1:
            # Camera motion becomes the scale reference when depth is hidden.
            scale = relative_poses[:, 1:, :3, 3].norm(dim=-1)
            moving = no_depth & (scale.max(dim=1).values >= 2e-2)
            scale = scale.mean(dim=1).clamp_min(1e-8)
            if self.training and self.scale_augmentation is not None:
                low, high = self.scale_augmentation
                scale *= torch.empty(batch, device=device).uniform_(low, high)
            relative_poses[moving, :, :3, 3] /= scale[moving, None, None]
            normalized_depths[moving] /= scale[moving, None, None, None]
            depth_scale[moving] *= scale[moving]
        if self.training and views > 1 and self.pose_noise_probability > 0:
            # Cumulative perturbations are temporally smoother than independent
            # frame noise and preserve the first frame as the reference.
            noisy = torch.rand(batch, device=device) < self.pose_noise_probability
            relative_poses[noisy, 1:] = add_randomized_smooth_pose_noise(relative_poses[noisy, 1:])
        depth_input = normalized_depths.reshape(batch * views, 1, height, width)
        depth_valid = (depth_input > 0).to(depth_input.dtype)
        depth_embedding = self.depth_encoder(torch.cat((depth_input, depth_valid), dim=1), is_training=True)["x_norm_patchtokens"] + self.depth_emb if depth_mask.any() else torch.zeros_like(hidden)
        ray_embedding = self.ray_embed(rays.reshape(batch * views, height, width, 2).permute(0, 3, 1, 2)) if ray_mask.any() else torch.zeros_like(hidden)
        hidden = hidden + depth_embedding * depth_mask.reshape(batch * views, 1, 1)
        hidden = hidden + ray_embedding * ray_mask.reshape(batch * views, 1, 1)
        return hidden, relative_poses, depth_mask, pose_mask, depth_scale

    def _checkpoint_decoder_block(self, block, hidden, pos, index):
        """Apply the selected activation-checkpoint policy to one decoder block."""
        use_checkpoint = self.training and (
            self.checkpoint_strategy == "all"
            or self.checkpoint_strategy == "global_only" and index % 2 == 1
        )
        return checkpoint(block, hidden, xpos=pos, use_reentrant=False) if use_checkpoint else block(hidden, xpos=pos)

    def decode(self, hidden, views, height, width, poses, pose_mask):
        """Training decoder with optional checkpointing of global and pose blocks."""
        if hidden.ndim == 4:
            batch, views, tokens = hidden.shape[:3]
        else:
            tokens, batch = hidden.shape[1], hidden.shape[0] // views
        hidden = hidden.reshape(batch * views, tokens, -1)
        hidden = torch.cat((self.register_token.repeat(batch, views, 1, 1).reshape(batch * views, self.patch_start_idx, -1), hidden), dim=1)
        tokens = hidden.shape[1]
        pos = self.position_getter(batch * views, height // 14, width // 14, hidden.device)
        pos = torch.cat((torch.zeros(batch * views, self.patch_start_idx, 2, dtype=pos.dtype, device=pos.device), pos + 1), dim=1)
        pose_enabled = self.use_multimodal and pose_mask is not None and pose_mask.any()
        pose_attention_mask = None
        if pose_enabled and not pose_mask.all():
            view_mask = pose_mask.unsqueeze(2) & pose_mask.unsqueeze(1)
            patch_tokens = tokens - self.patch_start_idx
            pose_attention_mask = view_mask.repeat_interleave(patch_tokens, 1).repeat_interleave(patch_tokens, 2)[:, None]
        pose_block_index, penultimate = 0, None
        for index, block in enumerate(self.decoder):
            if index % 2 == 0:
                hidden, pos = hidden.reshape(batch * views, tokens, -1), pos.reshape(batch * views, tokens, -1)
            else:
                hidden, pos = hidden.reshape(batch, views * tokens, -1), pos.reshape(batch, views * tokens, -1)
            hidden = self._checkpoint_decoder_block(block, hidden, pos, index)
            if pose_enabled and index in (1, 9, 17, 25, 33):
                hidden = hidden.reshape(batch, views, tokens, -1)
                patches = hidden[..., self.patch_start_idx:, :].reshape(batch, views * (tokens - self.patch_start_idx), -1)
                pose_block = self.pose_inject_blk[pose_block_index]
                if self.training and self.checkpoint_strategy in ("all", "global_only"):
                    pose_features = checkpoint(pose_block, patches, poses, height, width, height // 14, width // 14, attn_mask=pose_attention_mask, use_reentrant=False)
                else:
                    pose_features = pose_block(patches, poses, height, width, height // 14, width // 14, attn_mask=pose_attention_mask)
                hidden[..., self.patch_start_idx:, :] += pose_features.reshape(batch, views, -1, 1024) * pose_mask[:, :, None, None]
                hidden = hidden.reshape(batch, views * tokens, -1)
                pose_block_index += 1
            if index == len(self.decoder) - 2:
                penultimate = hidden.reshape(batch * views, tokens, -1)
        return torch.cat((penultimate, hidden.reshape(batch * views, tokens, -1)), dim=-1), pos.reshape(batch * views, tokens, -1)

    def forward_head(self, hidden, pos, batch, views, height, width, patch_h, patch_w):
        """Return normalized predictions and an independently supervised scale."""
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
            camera_poses = self.camera_head(camera_features[:, self.patch_start_idx:].float(), patch_h, patch_w).reshape(batch, views, 4, 4)
            metric = self.metric_head(metric_features.float()).reshape(batch).exp()
            confidence = self._chunked_conv_head(self.conf_head, confidence_features, patch_h, patch_w)[0]
            confidence = confidence.permute(0, 2, 3, 1).reshape(batch, views, height, width, 1)
        return {"local_points": local_points, "xy": xy, "rays": F.normalize(torch.cat((xy, torch.ones_like(z)), dim=-1), dim=-1), "conf": confidence, "camera_poses": camera_poses, "metric": metric}

    def forward(self, imgs, order_flow=None, depths=None, intrinsics=None, rays=None, poses=None, with_prior=None, mask_add_depth=None, mask_add_ray=None, mask_add_pose=None, dataset_names=None, dataset_name=None):
        """Run a training forward pass and attach masks consumed by Pi3XLoss."""
        if imgs.ndim != 5 or imgs.shape[2] != 3:
            raise ValueError(f"imgs must have shape B,N,3,H,W, got {tuple(imgs.shape)}")
        batch, views, _, height, width = imgs.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Image size {(height, width)} must be divisible by patch size {self.patch_size}")
        names = dataset_names if dataset_names is not None else dataset_name
        normalized = (imgs - self.image_mean) / self.image_std
        hidden, relative_poses, depth_mask, pose_mask, norm_factor = self.encode(normalized, with_prior, depths, rays, intrinsics, poses, mask_add_depth, mask_add_ray, mask_add_pose, names)
        hidden, pos = self.decode(hidden.reshape(batch, views, -1, self.dec_embed_dim), views, height, width, relative_poses, pose_mask)
        output = self.forward_head(hidden, pos, batch, views, height, width, height // 14, width // 14)
        output.update(order_flow=order_flow, use_depth_mask=depth_mask, use_pose_mask=pose_mask, norm_factor=norm_factor, ref_idxs=None)
        return output


Pi3XTraining = Pi3X

__all__ = ["Pi3X", "Pi3XTraining", "add_randomized_smooth_pose_noise"]
