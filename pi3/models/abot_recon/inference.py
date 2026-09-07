"""Finite-clip inference preprocessing for ABot-Recon.

The resize and padding policy follows Appendix A.3 and Sec. 4.1.  Images are
width-locked to 504 pixels, then vertically center-cropped or mean-color padded
to 504 x 280 while preserving horizontal field of view.  ``infer_paths`` feeds
the resulting finite clip to the differentiable model; the public FlashInfer
paged-KV streaming state is outside this helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class FovTransform:
    """Record the crop or padding needed to map outputs to source image rows."""

    source_height: int
    source_width: int
    resized_height: int
    target_height: int
    target_width: int
    crop_top: int = 0
    crop_bottom: int = 0
    pad_top: int = 0
    pad_bottom: int = 0


def preprocess_image(image, height=280, width=504,
                     pad_rgb: Sequence[float] = (0.485, 0.456, 0.406)):
    """Apply the Appendix A.3 field-of-view-preserving spatial transform."""

    if isinstance(image, Image.Image):
        tensor = tvf.to_tensor(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    elif torch.is_tensor(image):
        tensor = image.detach().cpu().float()
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
    else:
        raise TypeError(f"Unsupported image type {type(image).__name__}")
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected an RGB image in CHW or HWC layout")
    if tensor.numel() and tensor.max() > 1.5:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0.0, 1.0).contiguous()
    source_height, source_width = tensor.shape[-2:]
    # Appendix A.3 first fixes width and preserves aspect ratio.  Only the
    # vertical axis is subsequently cropped or padded.
    resized_height = max(1, round(source_height * width / max(source_width, 1)))
    tensor = tvf.resize(tensor, [resized_height, width],
                        interpolation=InterpolationMode.BICUBIC, antialias=True)
    crop_top = crop_bottom = pad_top = pad_bottom = 0
    if resized_height > height:
        crop_top = round((resized_height - height) * 0.5)
        crop_bottom = resized_height - height - crop_top
        tensor = tvf.crop(tensor, crop_top, 0, height, width)
    elif resized_height < height:
        pad_top = (height - resized_height) // 2
        pad_bottom = height - resized_height - pad_top
        # The report specifies ImageNet mean RGB padding in normalized [0,1]
        # space so artificial rows become zero after model normalization.
        canvas = torch.tensor(pad_rgb, dtype=tensor.dtype)[:, None, None].expand(
            3, height, width).clone()
        canvas[:, pad_top:pad_top + resized_height] = tensor
        tensor = canvas
    transform = FovTransform(
        source_height, source_width, resized_height, height, width,
        crop_top, crop_bottom, pad_top, pad_bottom)
    return tensor, transform


def load_image_sequence(paths: Iterable[str | Path], height=280, width=504):
    """Preprocess an ordered clip and require a shared source-camera geometry."""

    frames, reference = [], None
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            tensor, transform = preprocess_image(image, height, width)
        if reference is None:
            reference = transform
        elif transform != reference:
            raise ValueError(
                f"Frame {index} has inconsistent geometry {transform} versus {reference}")
        frames.append(tensor)
    if not frames:
        raise ValueError("No input images were provided")
    return torch.stack(frames, 0), reference


@torch.inference_mode()
def infer_paths(model, paths, device="cuda", dtype=torch.bfloat16,
                height=280, width=504):
    """Run Eq. (1) and pose composition on a preprocessed finite image clip."""

    frames, transform = load_image_sequence(paths, height, width)
    images = frames[None].to(device=device, dtype=dtype)
    output = model.inference(images)
    result = {
        key: value.detach().float().cpu() if torch.is_tensor(value) else value
        for key, value in output.items()
    }
    result["fov_transform"] = transform
    return result
