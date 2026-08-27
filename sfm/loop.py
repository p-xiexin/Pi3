"""Glob3R retrieval-based loop association with DINOv2 SALAD."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from pi3.models.dinov2.hub.backbones import dinov2_vitb14


SALAD_MEAN = (0.485, 0.456, 0.406)
SALAD_STD = (0.229, 0.224, 0.225)
LOOP_SIMILARITY_THRESHOLD = 0.5


class _DinoV2Backbone(nn.Module):
    """DINOv2-B/14 wrapper matching the released SALAD checkpoint layout."""

    def __init__(self):
        super().__init__()
        self.model = dinov2_vitb14(pretrained=False)
        self.num_trainable_blocks = 4
        self.norm_layer = True

    def forward(self, images):
        batch, _, height, width = images.shape
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks[:-self.num_trainable_blocks]:
            tokens = block(tokens)
        for block in self.model.blocks[-self.num_trainable_blocks:]:
            tokens = block(tokens)
        if self.norm_layer:
            tokens = self.model.norm(tokens)
        global_token = tokens[:, 0]
        feature_map = tokens[:, 1:].reshape(
            batch, height // 14, width // 14, 768
        ).permute(0, 3, 1, 2)
        return feature_map, global_token


def _log_optimal_transport(log_a, log_b, scores, iterations=3):
    """Run the log-domain Sinkhorn normalization used by SALAD."""
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


class _SaladAggregator(nn.Module):
    """SALAD optimal-transport aggregation for an 8448-D descriptor."""

    def __init__(self):
        super().__init__()
        self.num_clusters = 64
        self.cluster_dim = 128
        self.token_features = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 256)
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(768, 512, 1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(768, 512, 1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        features, token = inputs
        local_features = self.cluster_features(features).flatten(2)
        scores = self.score(features).flatten(2)
        token = self.token_features(token)
        batch, clusters, patches = scores.shape
        augmented = scores.new_empty(batch, clusters + 1, patches)
        augmented[:, :clusters] = scores
        augmented[:, clusters] = self.dust_bin
        norm = -scores.new_tensor(float(patches + clusters)).log()
        log_a = norm.expand(clusters + 1).clone()
        log_b = norm.expand(patches).clone()
        log_a[-1] += scores.new_tensor(float(patches - clusters)).log()
        transport = _log_optimal_transport(
            log_a.expand(batch, -1), log_b.expand(batch, -1), augmented
        )
        assignments = torch.exp(transport - norm)[:, :-1]
        clusters = (
            local_features.unsqueeze(2) * assignments.unsqueeze(1)
        ).sum(dim=-1)
        descriptor = torch.cat(
            (
                F.normalize(token, p=2, dim=-1),
                F.normalize(clusters, p=2, dim=1).flatten(1),
            ),
            dim=-1,
        )
        return F.normalize(descriptor, p=2, dim=-1)


class DinoSalad(nn.Module):
    """Inference-only DINOv2 SALAD model loaded from the official checkpoint."""

    def __init__(self, checkpoint):
        super().__init__()
        self.backbone = _DinoV2Backbone()
        self.aggregator = _SaladAggregator()
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)

    def forward(self, images):
        return self.aggregator(self.backbone(images))


def sliding_keyframe_pairs(packets, keyframe_ids):
    """Return symmetric keyframe pairs already colocated in a sliding window."""
    keyframe_ids = set(map(int, keyframe_ids))
    pairs = set()
    for packet in packets:
        if packet.get("kind", "sliding") != "sliding":
            continue
        colocated = sorted(
            keyframe_ids.intersection(map(int, packet["frame_ids"].tolist()))
        )
        for first_index, first in enumerate(colocated):
            for second in colocated[first_index + 1:]:
                pairs.add((first, second))
                pairs.add((second, first))
    return pairs


def loop_windows_from_descriptors(
    keyframe_ids,
    descriptors,
    threshold=LOOP_SIMILARITY_THRESHOLD,
    excluded_pairs=None,
):
    """Build one retrieval window per keyframe using Glob3R's cosine rule."""
    keyframe_ids = list(map(int, keyframe_ids))
    if descriptors.ndim != 2 or descriptors.shape[0] != len(keyframe_ids):
        raise ValueError("descriptors must have shape keyframes x channels")
    if len(keyframe_ids) < 2:
        return []
    descriptors = F.normalize(descriptors.float(), p=2, dim=-1)
    similarities = descriptors @ descriptors.T
    similarities.fill_diagonal_(-torch.inf)
    excluded_pairs = set() if excluded_pairs is None else set(excluded_pairs)
    windows = []
    for query, frame_id in enumerate(keyframe_ids):
        candidates = torch.nonzero(
            similarities[query] > float(threshold), as_tuple=False
        ).squeeze(-1).tolist()
        candidates = [
            candidate for candidate in candidates
            if (frame_id, keyframe_ids[candidate]) not in excluded_pairs
        ]
        candidates.sort(
            key=lambda candidate: (
                -float(similarities[query, candidate]), keyframe_ids[candidate]
            )
        )
        if candidates:
            windows.append([frame_id, *(keyframe_ids[index] for index in candidates)])
    return windows


class LoopDetector:
    """Extract keyframe descriptors and construct Glob3R loop windows."""

    def __init__(self, checkpoint, device, image_size, batch_size=16, model=None):
        self.device = torch.device(device)
        self.image_size = tuple(map(int, image_size))
        if any(size % 14 for size in self.image_size):
            raise ValueError("SALAD image_size must be divisible by 14")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("loop_batch_size must be positive")
        self.model = DinoSalad(checkpoint) if model is None else model
        self.model = self.model.eval().to(self.device)

    def _prepare(self, images):
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"SALAD expected image_size={self.image_size}, "
                f"received={tuple(images.shape[-2:])}"
            )
        mean = images.new_tensor(SALAD_MEAN)[None, :, None, None]
        std = images.new_tensor(SALAD_STD)[None, :, None, None]
        return (images - mean) / std

    def _read_original(self, path):
        """Read the full source RGB image without applying Pi3's center crop."""
        height, width = self.image_size
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            array = np.array(image, dtype=np.float32, copy=True) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    @torch.no_grad()
    def descriptors(self, frames, keyframe_ids):
        """Describe keyframes from uncropped source RGB images."""
        output = []
        for start in range(0, len(keyframe_ids), self.batch_size):
            batch_ids = keyframe_ids[start:start + self.batch_size]
            images = torch.stack(
                [self._read_original(frames.paths[frame_id]) for frame_id in batch_ids]
            )
            images = self._prepare(images.to(self.device))
            device_type = self.device.type
            with torch.amp.autocast(device_type=device_type, enabled=device_type == "cuda"):
                descriptors = self.model(images)
            if descriptors.ndim != 2 or descriptors.shape[0] != len(batch_ids):
                raise RuntimeError("DINO SALAD returned an invalid descriptor tensor")
            output.append(descriptors.float().cpu())
        if not output:
            return torch.empty(0, 8448)
        return torch.cat(output)

    def detect(
        self, frames, threshold=LOOP_SIMILARITY_THRESHOLD, excluded_pairs=None
    ):
        """Return additional matching windows with each query in the first slot."""
        keyframe_ids = sorted(frames.keyframes)
        descriptors = self.descriptors(frames, keyframe_ids)
        return loop_windows_from_descriptors(
            keyframe_ids, descriptors, threshold, excluded_pairs
        )


__all__ = [
    "DinoSalad",
    "LOOP_SIMILARITY_THRESHOLD",
    "LoopDetector",
    "loop_windows_from_descriptors",
    "sliding_keyframe_pairs",
]
