"""Ordered image index and lazy window loading."""

from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import yaml

from pi3.utils import cropping


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sort_key(path):
    """Build a natural filename key so frame 10 follows frame 9."""
    return [(0, int(x)) if x.isdigit() else (1, x.lower()) for x in re.split(r"(\d+)", path.name)]


def _crop_resize(image, K, size, mask=None):
    """Apply Pi3 resize and center crop while updating camera intrinsics."""
    height, width = size
    image, mask, K, _, _ = cropping.rescale_image_depthmap(image, mask, K, (width, height))
    resized_K = cropping.camera_matrix_of_crop(K, image.size, (width, height))
    bbox = cropping.bbox_from_intrinsics_in_out(K, resized_K, (width, height))
    image = image.crop(bbox)
    if mask is not None:
        left, top, right, bottom = bbox
        mask = mask[top:bottom, left:right]
    return image, mask, resized_K


class ImageDataset:
    """Index every input image once and decode only the active window."""

    def __init__(self, directory, image_size, calibration=None, mask=None):
        self.paths = sorted(
            (p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
            key=_sort_key,
        )
        if len(self.paths) < 3:
            raise RuntimeError(f"{directory} must contain at least three images")
        self.image_size = tuple(map(int, image_size))
        self.mask_path = None if mask is None else Path(mask)
        self.K = self._intrinsics(calibration)

    def __len__(self):
        return len(self.paths)

    def _intrinsics(self, calibration):
        with Image.open(self.paths[0]) as source:
            if calibration is None:
                focal = float(max(source.width, source.height))
                K = np.array(
                    [[focal, 0, source.width / 2], [0, focal, source.height / 2], [0, 0, 1]],
                    dtype=np.float32,
                )
            else:
                with Path(calibration).open("r", encoding="utf-8") as stream:
                    raw = yaml.safe_load(stream)
                fx, fy, cx, cy = map(float, raw["calibration"][:4])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                K[0] *= source.width / int(raw["width"])
                K[1] *= source.height / int(raw["height"])
            shared_mask = None
            if self.mask_path is not None:
                shared_mask = np.asarray(Image.open(self.mask_path).convert("L"))
            self.native_K = K.copy()
            _, shared_mask, K = _crop_resize(
                source.convert("RGB"), self.native_K, self.image_size, shared_mask
            )
        self.valid_mask = None if shared_mask is None else torch.from_numpy(shared_mask > 0)
        return torch.from_numpy(np.asarray(K, dtype=np.float32))

    def windows(self, window_size):
        """Yield half-overlapping frame indices and include the sequence tail once."""
        size = min(int(window_size), len(self))
        stride = max(size // 2, 1)
        starts = list(range(0, max(len(self) - size + 1, 1), stride))
        tail = len(self) - size
        if starts[-1] != tail:
            starts.append(tail)
        for start in starts:
            yield list(range(start, start + size))

    def read(self, frame_ids):
        """Decode selected frames as an ``N x 3 x H x W`` float tensor."""
        images = []
        for frame_id in frame_ids:
            with Image.open(self.paths[frame_id]) as source:
                image, _, _ = _crop_resize(source.convert("RGB"), self.native_K, self.image_size)
                array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(images)


__all__ = ["ImageDataset"]
