"""Glob3R coarse matching on consecutive ABotRecon feature frames."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..rope3d import apply_rope3d

from pi3.models.glob3r.model import Glob3RMatchingHead
from pi3.models.glob3r.geometry import denormalize_warp


@dataclass(frozen=True)
class MatchState:
    tokens: torch.Tensor
    encoder_features: tuple[torch.Tensor, ...]
    image_size: tuple[int, int]
    frame_index: int


class StreamingMatchHead(Glob3RMatchingHead):
    """Unmodified Glob3R coarse layers, with exactly t-1 -> t pairing.

    Public warps are full target-image pixel coordinates on a stride-four
    reference grid. Normalized warps are kept separately for training.
    No fine-feature pyramid or refinement parameters are constructed.
    """

    def __init__(self, encoder_dim=1024, geometry_dim=2048, match_dim=1024, patch_size=14):
        super().__init__(encoder_dim, geometry_dim, match_dim, patch_size, enable_refinement=False)

    def _check_features(self, geometry, encoder_features, image_size):
        height, width = image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image size must be divisible by patch size")
        if geometry.ndim != 4:
            raise ValueError("geometry must be [B,N,M,C]")
        b, n, patches, channels = geometry.shape
        if patches != height // self.patch_size * (width // self.patch_size):
            raise ValueError("geometry patch count does not match image size")
        if channels != self.match_decoder.input_projection.in_features:
            raise ValueError("geometry channel count does not match the decoder")
        if len(encoder_features) != 4 or any(x.shape != (b, n, patches, self.encoder_dim) for x in encoder_features):
            raise ValueError("expected four aligned DINO patch feature tensors")

    def _pairs(self, tokens, encoder_features, image_size, references, targets):
        b, pairs = tokens.shape[:2]
        packed = tokens.flatten(0, 1)
        skips = [feature.flatten(0, 1) for feature in encoder_features]
        height, width = image_size
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        similarity, embeddings, reference_tokens, _ = self.match_embedding(
            packed, patch_h, patch_w, reference_index=0)
        normalized_warp, confidence, confidence_logits = self.dpt_match(
            reference_tokens, embeddings, skips, 0, patch_h, patch_w, height, width)
        def unpack(value):
            return value[:, 0].reshape(b, pairs, *value.shape[2:])
        return {
            "similarity": unpack(similarity),
            "coarse_warp": unpack(denormalize_warp(normalized_warp, image_size)),
            "normalized_coarse_warp": unpack(normalized_warp),
            "coarse_confidence": unpack(confidence),
            "coarse_confidence_logits": unpack(confidence_logits),
            "reference_indices": references,
            "target_indices": targets,
            "image_size": tuple(image_size),
            "patch_size": self.patch_size,
        }

    def forward(self, geometry, encoder_features, image_size, target_indices=None):
        """Train on selected adjacent pairs from a causally encoded clip."""
        self._check_features(geometry, encoder_features, image_size)
        frames = geometry.shape[1]
        targets = torch.as_tensor(
            list(range(1, frames)) if target_indices is None else target_indices,
            dtype=torch.long, device=geometry.device)
        if targets.ndim != 1 or targets.numel() == 0 or ((targets < 1) | (targets >= frames)).any():
            raise ValueError("target_indices must contain valid noninitial clip indices")
        if torch.unique(targets).numel() != targets.numel():
            raise ValueError("target_indices must not repeat a pair")
        pairs = torch.stack((targets - 1, targets), -1)
        # Decode each selected frame once, while retaining gradients on both sides.
        selected, inverse = torch.unique(pairs.flatten(), sorted=True, return_inverse=True)
        tokens = self.match_decoder(geometry[:, selected])[:, inverse].reshape(
            geometry.shape[0], targets.numel(), 2, geometry.shape[2], -1)
        skips = [feature[:, pairs] for feature in encoder_features]
        return self._pairs(tokens, skips, image_size, targets - 1, targets)

    @torch.no_grad()
    def forward_step(self, geometry, encoder_features, image_size, state=None):
        """Return one adjacent match and the next one-frame cache, or None at startup."""
        if self.training:
            raise RuntimeError("forward_step requires eval mode; use forward for training")
        geometry = geometry.unsqueeze(1)
        features = tuple(value.unsqueeze(1) for value in encoder_features)
        self._check_features(geometry, features, image_size)
        current = self.match_decoder(geometry)[:, 0]
        frame = 0 if state is None else state.frame_index + 1
        result = None
        if state is not None:
            if state.image_size != tuple(image_size) or state.tokens.shape != current.shape or state.tokens.device != current.device or state.tokens.dtype != current.dtype:
                raise ValueError("stream shape, device or dtype changed; reset the state")
            tokens = torch.stack((state.tokens, current), 1)[:, None]
            skips = [torch.stack((old, new), 1)[:, None] for old, new in zip(state.encoder_features, encoder_features)]
            targets = torch.tensor([frame], device=current.device)
            result = self._pairs(tokens, skips, image_size, targets - 1, targets)
        next_state = MatchState(current.detach(), tuple(x.detach() for x in encoder_features), tuple(image_size), frame)
        return result, next_state


@dataclass(frozen=True)
class DecoderStreamState:
    frame_index: int
    image_size: tuple[int, int]
    # Each entry has [B, heads, cached_frames, tokens, head_dim].
    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]


def stream_rope_grid(rope, frame, patch_h, patch_w, special_tokens, device):
    """Use the existing spatial convention and extend only the frame frequency.

    The one-position phase in the existing table defines the rotary frequency,
    including checkpoints instantiated with a nondefault rotary base.
    """
    if frame < 0:
        raise ValueError("frame index must be nonnegative")
    grid = rope.grid(1, patch_h, patch_w, special_tokens, device)[0].clone()
    frame_dim = rope.fhw_dim[0] // 2
    if frame < rope.frequencies.shape[0]:
        frequency = rope.frequencies[frame, :frame_dim].to(device)
    else:
        if rope.frequencies.shape[0] < 2:
            raise ValueError("streaming RoPE needs at least two precomputed positions")
        phase = torch.angle(rope.frequencies[1, :frame_dim].to(device)) * frame
        frequency = torch.polar(torch.ones_like(phase), phase)
    grid[:, :frame_dim] = frequency
    return grid


@torch.no_grad()
def decode_feature_step(backbone, encoded, height, width, state=None):
    """Decode one frame with an explicit, bounded inference state.

    This returns geometry features, not camera predictions. Passing None
    starts an independent sequence. The original backbone remains unchanged.
    """
    if backbone.training:
        raise RuntimeError("decode_step requires eval mode")
    if encoded.ndim != 3 or height % backbone.patch_size or width % backbone.patch_size:
        raise ValueError("expected patch tokens [B,M,C] and patch-aligned image size")
    b, patches, _ = encoded.shape
    if patches != (height // backbone.patch_size) * (width // backbone.patch_size):
        raise ValueError("patch count does not match image size")
    frame = 0 if state is None else state.frame_index
    if state is not None:
        if state.image_size != (height, width) or len(state.key_values) != len(backbone.decoder) // 2:
            raise ValueError("stream shape changed; reset the state")
    registers = backbone.register_token.reshape(1, backbone.patch_start_idx, backbone.dec_embed_dim)
    hidden = torch.cat((registers.expand(b, -1, -1), encoded), 1)
    tokens, channels = hidden.shape[1:]
    pos = backbone.position_getter(b, height // backbone.patch_size, width // backbone.patch_size, encoded.device) + 1
    pos = torch.cat((pos.new_zeros(b, backbone.patch_start_idx, 2), pos), 1)
    frequencies = None
    if backbone.global_pos_encoding == "rope3d":
        frequencies = stream_rope_grid(
            backbone.rope3d, frame, height // backbone.patch_size,
            width // backbone.patch_size, backbone.patch_start_idx, encoded.device)[None, None]
    new_kv, final = [], []
    for index, block in enumerate(backbone.decoder):
        if index % 2 == 0:
            hidden = backbone._local_block(block, hidden, pos)
        else:
            attention = block.attn
            value = block.norm1(hidden)
            qkv = attention.qkv(value).reshape(
                b, tokens, 3, attention.num_heads, channels // attention.num_heads).transpose(1, 3)
            q, k, v = (qkv[:, :, item] for item in range(3))
            q, k = attention.q_norm(q).to(v.dtype), attention.k_norm(k).to(v.dtype)
            if frequencies is not None:
                q, k = apply_rope3d(q, frequencies), apply_rope3d(k, frequencies)
            elif attention.rope is not None:
                q, k = attention.rope(q, pos), attention.rope(k, pos)
            keys, values = k.unsqueeze(2), v.unsqueeze(2)
            if state is not None:
                old_k, old_v = state.key_values[index // 2]
                expected = (b, attention.num_heads, min(frame, backbone.local_window_frames - 1), tokens, k.shape[-1])
                if old_k.shape != expected or old_v.shape != expected or old_k.device != k.device or old_k.dtype != k.dtype:
                    raise ValueError("stream batch, device, dtype or cache shape changed; reset the state")
                keys = torch.cat((old_k, keys), 2)
                values = torch.cat((old_v, values), 2)
            attended = F.scaled_dot_product_attention(q, keys.flatten(2, 3), values.flatten(2, 3))
            attended = attended.transpose(1, 2).reshape(b, tokens, channels)
            residual = backbone._attention_output(attention, value, attended)
            hidden = hidden + block.drop_path1(block.ls1(residual))
            hidden = hidden + block.drop_path2(block.ls2(block.mlp(block.norm2(hidden))))
            keep = backbone.local_window_frames - 1
            # clone releases the storage of the evicted frame, including K=1.
            start = max(keys.shape[2] - keep, 0)
            new_kv.append((keys[:, :, start:].clone(), values[:, :, start:].clone()))
        if index >= len(backbone.decoder) - 2:
            final.append(hidden)
    return torch.cat(final, -1), pos, DecoderStreamState(frame + 1, (height, width), tuple(new_kv))
