"""Three-axis rotary positions used by the released ABot-Recon implementation.

The report specifies windowed causal attention in Eq. (6) but does not assign a
separate equation to positional encoding.  The public checkpoint splits every
attention head across frame, patch-height, and patch-width rotary frequencies;
this module records that implementation detail explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE3D(nn.Module):
    """Build parameter-free frame, height, and width rotary frequencies."""

    def __init__(self, head_dim=64, max_seq_len=4096, theta=10000.0,
                 fhw_dim=(20, 22, 22)):
        super().__init__()
        if sum(fhw_dim) != head_dim or any(dim % 2 for dim in fhw_dim):
            raise ValueError("fhw_dim must be even and sum to the attention head dimension")
        self.fhw_dim = tuple(int(value) for value in fhw_dim)
        frequencies = []
        position = torch.arange(max_seq_len, dtype=torch.float64)
        for dim in self.fhw_dim:
            inverse = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
            frequencies.append(torch.polar(torch.ones(
                max_seq_len, dim // 2, dtype=torch.float64),
                                            torch.outer(position, inverse)))
        self.register_buffer("frequencies", torch.cat(frequencies, 1), persistent=False)

    def grid(self, frames, patch_h, patch_w, special_tokens, device):
        """Return complex frequencies aligned with ``[frame, token]`` order.

        Camera/register tokens receive distinct spatial slots and the frame
        coordinate of their owning image.  Patch tokens receive the Cartesian
        product of frame, patch-row, and patch-column coordinates.
        """

        freq_f, freq_h, freq_w = self.frequencies.to(device).split(
            [value // 2 for value in self.fhw_dim], 1)
        special = torch.cat((
            freq_f[:frames, None].expand(frames, special_tokens, -1),
            freq_h[:special_tokens][None].expand(frames, -1, -1),
            freq_w[:special_tokens][None].expand(frames, -1, -1)), -1)
        frame = freq_f[:frames, None, None].expand(frames, patch_h, patch_w, -1)
        height = freq_h[special_tokens:special_tokens + patch_h][None, :, None].expand(
            frames, patch_h, patch_w, -1)
        width = freq_w[special_tokens:special_tokens + patch_w][None, None].expand(
            frames, patch_h, patch_w, -1)
        patches = torch.cat((frame, height, width), -1).reshape(frames, patch_h * patch_w, -1)
        return torch.cat((special, patches), 1)


def apply_rope3d(value, frequencies):
    """Rotate adjacent channel pairs by the supplied complex 3D frequencies."""

    even, odd = value[..., 0::2], value[..., 1::2]
    cosine = frequencies.real.to(value.dtype)
    sine = frequencies.imag.to(value.dtype)
    rotated = torch.stack((even * cosine - odd * sine,
                           even * sine + odd * cosine), -1)
    return rotated.reshape_as(value)
