"""Glob3R coarse dense matching and coarse-to-fine refinement.

This consolidated model module follows Appendix A of:
    Deng et al., "Glob3R: Global Structure-from-Motion with 3D Foundation
    Models", 2026.

Every equation implemented below is referenced at the corresponding operation.
The matching head is composed with the frozen Pi3 backbone by
:class:`pi3.models.glob3r.Glob3R`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn

from .geometry import sample_map_at_pixels


@dataclass
class GeometryPrediction:
    """Glob3R Eq. (1): ``f({I_i}) = {T_i, X_i, C_i, m_i}``."""

    camera_poses: torch.Tensor
    local_points: torch.Tensor
    confidence: Optional[torch.Tensor]
    metric_scale: Optional[torch.Tensor]


def pack_geometry_prediction(output: dict) -> GeometryPrediction:
    """Execute the Glob3R Eq. (1) output grouping on a Pi3X prediction dict."""

    return GeometryPrediction(
        camera_poses=output["camera_poses"],
        local_points=output["local_points"],
        confidence=output.get("conf"),
        metric_scale=output.get("metric"),
    )


class MatchAttentionBlock(nn.Module):
    """A pre-norm transformer block used by the five-layer matching decoder."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return tokens + self.mlp(self.norm2(tokens))


class MatchingDecoder(nn.Module):
    """The five-attention-layer ``Dec_match`` described in Appendix A."""

    def __init__(
        self,
        geometry_dim: int = 2048,
        match_dim: int = 1024,
        depth: int = 5,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        if depth != 5:
            raise ValueError("Glob3R Appendix A specifies exactly five matching attention layers")
        self.input_projection = nn.Linear(geometry_dim, match_dim)
        self.blocks = nn.ModuleList([MatchAttentionBlock(match_dim, num_heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(match_dim)

    def forward(self, geometry_tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, _ = geometry_tokens.shape
        tokens = self.input_projection(geometry_tokens).reshape(batch * frames, patches, -1)
        for block in self.blocks:
            tokens = block(tokens)
        # Glob3R Eq. (9): Z = Dec_match(H), Z in R^(N x H' x W' x C).
        return self.output_norm(tokens).reshape(batch, frames, patches, -1)


class MultiViewMatchEmbedding(nn.Module):
    """Patch similarity and Fourier-coordinate aggregation from Eqs. (10)-(17)."""

    def __init__(self, dim: int = 1024, temperature: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("Fourier embedding dimension must be even")
        generator = torch.Generator().manual_seed(seed)
        # Glob3R Eq. (15): fixed, non-learnable Gaussian W; omega=1.
        gaussian = torch.randn(dim // 2, 2, generator=generator)
        self.register_buffer("gaussian_matrix", gaussian, persistent=True)
        self.temperature = temperature

    def fourier_reference_coordinates(
        self, patch_height: int, patch_width: int, device, dtype
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, patch_height, device=device, dtype=dtype),
            torch.linspace(-1, 1, patch_width, device=device, dtype=dtype),
            indexing="ij",
        )
        coordinates = torch.stack((x, y), dim=-1).reshape(-1, 2)
        phase = 2 * torch.pi * F.linear(coordinates, self.gaussian_matrix.to(dtype=dtype))
        # Glob3R Eq. (15): chi_n^a = cos(2*pi*omega*W*x_n^a) (+) sin(...).
        return torch.cat((phase.cos(), phase.sin()), dim=-1)

    def forward(
        self, match_tokens: torch.Tensor, patch_height: int, patch_width: int, reference_index: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        target_indices = [i for i in range(match_tokens.shape[1]) if i != reference_index]
        # Glob3R Eq. (10): Z^a = {z_n^a}_{n=1}^M.
        reference = match_tokens[:, reference_index]
        # Glob3R Eq. (11): Z^b = {z_m^b}_{m=1}^M for every b != a.
        targets = match_tokens[:, target_indices]

        # Glob3R Eq. (13): cosim(x,y) = x^T y / (||x|| ||y||).
        reference_normalized = F.normalize(reference, dim=-1)
        targets_normalized = F.normalize(targets, dim=-1)
        cosine_similarity = torch.einsum("btmc,bnc->btmn", targets_normalized, reference_normalized)
        # Glob3R Eq. (12): S_mn^(a->b) = exp(cosim(z_m^b,z_n^a) / tau), tau=1/10.
        similarity = torch.exp(cosine_similarity / self.temperature)
        # Glob3R Eq. (14): stack S^(a->b) over B={1,...,N}\{a}.

        fourier = self.fourier_reference_coordinates(
            patch_height, patch_width, match_tokens.device, match_tokens.dtype
        )
        # Glob3R Eq. (16): chi_m^(a->b) = sum_n S_mn^(a->b) chi_n^a.
        embeddings = torch.einsum("btmn,nc->btmc", similarity, fourier)
        # Glob3R Eq. (17): stack all multi-view match embeddings as [B,N-1,H',W',C].
        return similarity, embeddings, targets, target_indices


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class DPTMatchingHead(nn.Module):
    """DPT fusion head producing a stride-four warp and confidence map."""

    def __init__(
        self,
        match_dim: int = 1024,
        encoder_dim: int = 1024,
        scratch_dim: int = 256,
        encoder_output_dims: Sequence[int] = (256, 512, 1024, 1024),
    ) -> None:
        super().__init__()
        if tuple(encoder_output_dims) != (256, 512, 1024, 1024) or scratch_dim != 256:
            raise ValueError("Glob3R Appendix A specifies scratch=256 and [256,512,1024,1024]")
        self.pair_projection = nn.Linear(2 * match_dim, match_dim)
        self.pair_to_scratch = nn.Conv2d(match_dim, scratch_dim, 1)
        self.encoder_projects = nn.ModuleList(
            [nn.Conv2d(encoder_dim, channels, 1) for channels in encoder_output_dims]
        )
        self.encoder_to_scratch = nn.ModuleList(
            [nn.Conv2d(channels, scratch_dim, 3, padding=1) for channels in encoder_output_dims]
        )
        self.fusion = nn.ModuleList([ResidualConvUnit(scratch_dim) for _ in range(4)])
        self.output = nn.Sequential(
            ResidualConvUnit(scratch_dim),
            nn.Conv2d(scratch_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 3, 1),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        match_embeddings: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        target_indices: Sequence[int],
        patch_height: int,
        patch_width: int,
        image_height: int,
        image_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, targets, patches, _ = target_tokens.shape
        # Glob3R Eq. (18): F^(a->b) = Proj(Z^b (+) chi^(a->b)).
        pair = self.pair_projection(torch.cat((target_tokens, match_embeddings), dim=-1))
        # Glob3R Eq. (19): pair features are stacked over all target views.
        pair_map = pair.reshape(batch * targets, patch_height, patch_width, -1).permute(0, 3, 1, 2)
        pair_map = self.pair_to_scratch(pair_map)

        if len(encoder_features) != 4:
            raise ValueError("DPT_match requires four multi-scale encoder features")
        pyramid = []
        target_sizes = [
            (max(image_height // 4, 1), max(image_width // 4, 1)),
            (max(image_height // 8, 1), max(image_width // 8, 1)),
            (max(image_height // 16, 1), max(image_width // 16, 1)),
            (max(image_height // 32, 1), max(image_width // 32, 1)),
        ]
        for feature, project, to_scratch, size in zip(
            encoder_features, self.encoder_projects, self.encoder_to_scratch, target_sizes
        ):
            selected = feature[:, target_indices]
            selected = selected.reshape(batch * targets, patches, -1)
            selected = selected.transpose(1, 2).reshape(batch * targets, -1, patch_height, patch_width)
            selected = to_scratch(project(selected))
            pyramid.append(F.interpolate(selected, size=size, mode="bilinear", align_corners=True))

        fused = self.fusion[-1](pyramid[-1])
        for level in range(2, -1, -1):
            fused = F.interpolate(fused, size=pyramid[level].shape[-2:], mode="bilinear", align_corners=True)
            fused = self.fusion[level](fused + pyramid[level])
        pair_map = F.interpolate(pair_map, size=target_sizes[0], mode="bilinear", align_corners=True)
        prediction = self.output(fused + pair_map)

        # Glob3R Eq. (20): (W^(a->B), p^(a->B)) = DPT_match(F^(a->B), E).
        raw_warp, confidence_logits = prediction[:, :2], prediction[:, 2:3]
        raw_warp = raw_warp.sigmoid()
        scale = raw_warp.new_tensor([image_width - 1, image_height - 1]).view(1, 2, 1, 1)
        warp = raw_warp * scale
        confidence = confidence_logits.sigmoid()
        stride4_height, stride4_width = target_sizes[0]
        # Glob3R Eq. (21): W:[B,N-1,2,H/4,W/4], p:[B,N-1,1,H/4,W/4].
        return (
            warp.reshape(batch, targets, 2, stride4_height, stride4_width),
            confidence.reshape(batch, targets, 1, stride4_height, stride4_width),
            confidence_logits.reshape(batch, targets, 1, stride4_height, stride4_width),
        )


class FineFeaturePyramid(nn.Module):
    """Image feature extractor with the exact channel/stride contract of Eq. (22)."""

    def __init__(self) -> None:
        super().__init__()
        # RoMaV2 extracts VGG19 activations immediately before the first three
        # max-pools (64/128/256 channels), then projects them to Eq. (22)'s
        # 12/48/192 channels.
        # RoMaV2's public default is VGG19-BN.  Keeping the BN variant is
        # important because Appendix B initializes this branch from RoMaV2.
        # Index 26 is the third MaxPool; activations are captured immediately
        # before each of the first three pools, at strides 1, 2, and 4.
        self.vgg_features = vgg19_bn(weights=None).features[:27]
        self.projections = nn.ModuleDict(
            {"1": nn.Conv2d(64, 12, 1), "2": nn.Conv2d(128, 48, 1), "4": nn.Conv2d(256, 192, 1)}
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch, frames, channels, height, width = images.shape
        flat = images.reshape(batch * frames, channels, height, width)
        value = (flat - self.image_mean) / self.image_std
        raw: Dict[int, torch.Tensor] = {}
        stride = 1
        for layer in self.vgg_features:
            if isinstance(layer, nn.MaxPool2d):
                raw[stride] = value
                stride *= 2
            value = layer(value)
        phi1 = self.projections["1"](raw[1])
        phi2 = self.projections["2"](raw[2])
        phi4 = self.projections["4"](raw[4])
        return {
            4: phi4.reshape(batch, frames, *phi4.shape[1:]),
            2: phi2.reshape(batch, frames, *phi2.shape[1:]),
            1: phi1.reshape(batch, frames, *phi1.shape[1:]),
        }


def local_correlation(
    reference: torch.Tensor,
    target: torch.Tensor,
    warp: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Local target-neighborhood correlation used in Glob3R Eq. (23)."""

    if radius == 0:
        return reference.new_zeros(reference.shape[0], 0, *reference.shape[-2:])
    correlations = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = warp + warp.new_tensor([dx, dy]).view(1, 1, 1, 2)
            sampled = sample_map_at_pixels(target, shifted)
            correlations.append((F.normalize(reference, dim=1) * F.normalize(sampled, dim=1)).sum(1))
    return torch.stack(correlations, dim=1)


class DepthwisePointwiseUnit(nn.Module):
    """RoMaV2 depthwise -> BatchNorm -> ReLU -> pointwise unit."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            nn.BatchNorm2d(channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class DepthwiseRefinementBlock(nn.Module):
    """Compact depthwise/normalization/non-linearity/pointwise refinement block."""

    def __init__(self, input_channels: int, hidden_blocks: int = 8) -> None:
        super().__init__()
        # RoMaV2 uses one input block followed by eight identical hidden blocks.
        self.body = nn.Sequential(
            DepthwisePointwiseUnit(input_channels),
            *[DepthwisePointwiseUnit(input_channels) for _ in range(hidden_blocks)],
        )
        self.output = nn.Conv2d(input_channels, 3, 1)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        update = self.output(self.body(feature))
        # Glob3R Eq. (24): (Delta W_s, Delta p_s) = Refine_s(F_s).
        return update[:, :2], update[:, 2:3]


class WarpRefinement(nn.Module):
    """Glob3R stride {4,2,1} residual warp/confidence refinement."""

    channels = {4: 192, 2: 48, 1: 12}
    # Appendix window sizes [7,3,0] correspond to RoMaV2 radii [3,1,None].
    radii = {4: 3, 2: 1, 1: 0}
    displacement_dims = {4: 79, 2: 23, 1: 8}

    def __init__(self) -> None:
        super().__init__()
        self.displacement = nn.ModuleDict(
            {str(s): nn.Conv2d(2, self.displacement_dims[s], 1) for s in self.channels}
        )
        self.blocks = nn.ModuleDict()
        for stride, channels in self.channels.items():
            correlation_channels = (2 * self.radii[stride] + 1) ** 2 if self.radii[stride] else 0
            input_channels = 2 * channels + self.displacement_dims[stride] + correlation_channels
            self.blocks[str(stride)] = DepthwiseRefinementBlock(input_channels)

    @staticmethod
    def _coordinate_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def _normalize_warp(warp: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
        scale = warp.new_tensor([image_width - 1, image_height - 1]).view(1, 2, 1, 1)
        return 2.0 * warp / scale.clamp_min(1) - 1.0

    @staticmethod
    def _pixel_warp(warp: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
        scale = warp.new_tensor([image_width - 1, image_height - 1]).view(1, 2, 1, 1)
        return (warp + 1.0) * 0.5 * scale

    def forward(
        self,
        coarse_warp: torch.Tensor,
        coarse_confidence_logits: torch.Tensor,
        image_features: Dict[int, torch.Tensor],
        target_indices: Sequence[int],
        reference_index: int,
        image_height: int,
        image_width: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # B: batch, T: target views, N: all views, C_s/H_s/W_s: features at stride s.
        # coarse_warp: [B,T,2,H/4,W/4] -> [B*T,2,H/4,W/4], treating each
        # reference-target pair as an independent sample for 2D refinement.
        batch, targets = coarse_warp.shape[:2]
        warp = coarse_warp.reshape(batch * targets, 2, *coarse_warp.shape[-2:])
        # RoMaV2 refiners operate in normalized [-1,1] coordinates. Keeping
        # that convention makes the Appendix-B checkpoint initialization and
        # its displacement/output scales compatible; public outputs remain px.
        warp = self._normalize_warp(warp, image_height, image_width)
        # coarse_confidence_logits: [B,T,1,H/4,W/4] -> [B*T,1,H/4,W/4].
        confidence_logits = coarse_confidence_logits.reshape(
            batch * targets, 1, *coarse_confidence_logits.shape[-2:]
        )
        warp_stages, confidence_stages = [], []

        for stride in (4, 2, 1):
            # RoMaV2 decouples stages by stopping gradients through the previous estimate.
            warp = warp.detach()
            confidence_logits = confidence_logits.detach()
            # features: [B,N,C_s,H_s,W_s]. Repeat the reference T times and
            # flatten both view selections to aligned [B*T,C_s,H_s,W_s] pairs.
            features = image_features[stride]
            reference = features[:, reference_index, None].expand(-1, targets, -1, -1, -1)
            reference = reference.reshape(batch * targets, *reference.shape[2:])
            target = features[:, target_indices].reshape(batch * targets, *reference.shape[1:])
            feature_height, feature_width = reference.shape[-2:]
            if warp.shape[-2:] != (feature_height, feature_width):
                # Normalized coordinate values are resolution-independent; only
                # their spatial prediction grid becomes [B*T,2,H_s,W_s].
                warp = F.interpolate(warp, size=(feature_height, feature_width), mode="bilinear", align_corners=True)
                confidence_logits = F.interpolate(
                    confidence_logits, size=(feature_height, feature_width), mode="bilinear", align_corners=True
                )
            # [B*T,2,H_s,W_s] -> [B*T,H_s,W_s,2] for coordinate operations;
            # feature_warp contains pixel coordinates in the stride-s feature map.
            warp_xy = warp.permute(0, 2, 3, 1)
            feature_scale = warp.new_tensor([feature_width - 1, feature_height - 1])
            feature_warp = (warp_xy + 1.0) * 0.5 * feature_scale
            # Only the previous warp estimate is detached between stages.
            # Gradients must still reach both fine-feature towers in Eq. (23).
            # target_sampled: [B*T,C_s,H_s,W_s].
            target_sampled = sample_map_at_pixels(target, feature_warp)
            # normalized_grid: [B*T,H_s,W_s,2]; displacement: [B*T,2,H_s,W_s].
            normalized_grid = self._coordinate_grid(
                batch * targets, feature_height, feature_width, warp.device, warp.dtype
            )
            displacement = (warp_xy - normalized_grid).permute(0, 3, 1, 2)
            # correlation: [B*T,K_s,H_s,W_s], K_s=(2r_s+1)^2 or 0 at stride 1.
            correlation = local_correlation(reference, target, feature_warp, self.radii[stride])
            # Glob3R Eq. (23): concat phi_a, sampled phi_b(W), g_s(W-x_a), and Corr_s.
            # refine_feature: [B*T,2C_s+D_s+K_s,H_s,W_s].
            refine_feature = torch.cat(
                (reference, target_sampled, self.displacement[str(stride)](displacement), correlation), dim=1
            )
            # Both residuals keep the flattened pair batch: [B*T,2/1,H_s,W_s].
            delta_warp, delta_confidence = self.blocks[str(stride)](refine_feature)
            # Glob3R Eq. (25): W_s <- upsample(W_2s) + Delta W_s.
            warp = warp + delta_warp
            confidence_logits = confidence_logits + delta_confidence
            pixel_warp = self._pixel_warp(warp, image_height, image_width)
            # Restore the target-view axis for public outputs at each stride.
            warp_stages.append(pixel_warp.reshape(batch, targets, 2, feature_height, feature_width))
            confidence_stages.append(
                confidence_logits.sigmoid().reshape(batch, targets, 1, feature_height, feature_width)
            )
        return warp_stages, confidence_stages


@dataclass
class MatchingOutput:
    similarity: torch.Tensor
    coarse_warp: torch.Tensor
    coarse_confidence: torch.Tensor
    warp_stages: List[torch.Tensor]
    confidence_stages: List[torch.Tensor]
    target_indices: List[int]


class Glob3RMatchingHead(nn.Module):
    """Complete matching branch from Glob3R Eqs. (2) and (7)-(25)."""

    def __init__(
        self,
        encoder_dim: int = 1024,
        geometry_dim: int = 2048,
        match_dim: int = 1024,
        patch_size: int = 14,
        enable_refinement: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim  # Appendix A, Pi3X Backbone, Eq. (7)-(8).
        self.patch_size = patch_size  # Appendix A, Pi3X Backbone, Eq. (7).
        self.match_decoder = MatchingDecoder(geometry_dim, match_dim)  # Appendix A, Eq. (9).
        self.match_embedding = MultiViewMatchEmbedding(match_dim)  # Appendix A, Eq. (10)-(17).
        self.dpt_match = DPTMatchingHead(match_dim, encoder_dim)  # Appendix A, Eq. (18)-(21).
        self.enable_refinement = enable_refinement  # Appendix A, Refinement Module.
        self.refinement_active = enable_refinement  # Appendix B, separate-stage training.
        self.fine_features = FineFeaturePyramid() if enable_refinement else None  # Appendix A, Eq. (22).
        self.refinement = WarpRefinement() if enable_refinement else None  # Appendix A, Eq. (23)-(25).

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
        images: torch.Tensor,
        reference_index: int = 0,
    ) -> MatchingOutput:
        if images.ndim != 5:
            raise ValueError(f"images must be [B,N,3,H,W], got {tuple(images.shape)}")
        batch, frames, _, height, width = images.shape
        patch_height, patch_width = height // self.patch_size, width // self.patch_size
        patches = patch_height * patch_width
        # Glob3R Eq. (7): each Pi3X encoder feature is [B,N,M,C].
        if len(encoder_features) != 4:
            raise ValueError(f"expected four Pi3X encoder features, got {len(encoder_features)}")
        # Glob3R Eq. (8): concatenated geometry tokens H are [B,N,M,2C].
        expected_geometry_shape = (batch, frames, patches, 2 * self.encoder_dim)
        if tuple(geometry_tokens.shape) != expected_geometry_shape:
            raise ValueError(
                f"geometry_tokens must be {expected_geometry_shape}, got {tuple(geometry_tokens.shape)}"
            )

        match_tokens = self.match_decoder(geometry_tokens)
        similarity, embeddings, targets, target_indices = self.match_embedding(
            match_tokens, patch_height, patch_width, reference_index
        )
        coarse_warp, coarse_confidence, coarse_confidence_logits = self.dpt_match(
            targets,
            embeddings,
            encoder_features,
            target_indices,
            patch_height,
            patch_width,
            height,
            width,
        )
        warp_stages: List[torch.Tensor] = []
        confidence_stages: List[torch.Tensor] = []
        if self.refinement_active:
            fine_features = self.fine_features(images)
            warp_stages, confidence_stages = self.refinement(
                coarse_warp,
                coarse_confidence_logits,
                fine_features,
                target_indices,
                reference_index,
                height,
                width,
            )
        # Glob3R Eq. (2): W_(a->B),p_(a->B) = DPT_match(Dec_match(H),a).
        return MatchingOutput(
            similarity,
            coarse_warp,
            coarse_confidence,
            warp_stages,
            confidence_stages,
            target_indices,
        )


def _unwrap_romav2_checkpoint(checkpoint):
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
    return checkpoint


def _find_state_suffix(state: Dict[str, torch.Tensor], suffix: str):
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _copy_compatible_tensor(
    parameter: torch.Tensor, source: Optional[torch.Tensor], name: str, loaded: List[str]
) -> None:
    if source is None:
        return
    if source.ndim == 2 and parameter.ndim == 4 and parameter.shape[-2:] == (1, 1):
        source = source[:, :, None, None]
    if parameter.shape != source.shape:
        return
    parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
    loaded.append(name)


def load_romav2_refinement(model, checkpoint_path: str) -> List[str]:
    """Shape-safely initialize the Appendix-B refinement branch from RoMaV2."""

    state = _unwrap_romav2_checkpoint(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    head = model.glob3r_matching_head
    if head.refinement is None or head.fine_features is None:
        raise ValueError("RoMaV2 initialization requires enable_refinement=true")
    loaded: List[str] = []
    with torch.no_grad():
        # Official RoMaV2 calls this VGG19-BN tower refiner_features.layers.
        for index, module in enumerate(head.fine_features.vgg_features):
            for attribute in (
                "weight", "bias", "running_mean", "running_var", "num_batches_tracked"
            ):
                if hasattr(module, attribute):
                    _copy_compatible_tensor(
                        getattr(module, attribute),
                        _find_state_suffix(
                            state, f"refiner_features.layers.{index}.{attribute}"
                        ),
                        f"fine_features.vgg_features.{index}.{attribute}",
                        loaded,
                    )

        for stride in (4, 2, 1):
            key = str(stride)
            projection = head.fine_features.projections[key]
            for attribute in ("weight", "bias"):
                _copy_compatible_tensor(
                    getattr(projection, attribute),
                    _find_state_suffix(state, f"refiners.{key}.proj.{attribute}"),
                    f"fine_features.projections.{key}.{attribute}",
                    loaded,
                )
            displacement = head.refinement.displacement[key]
            for attribute in ("weight", "bias"):
                _copy_compatible_tensor(
                    getattr(displacement, attribute),
                    _find_state_suffix(state, f"refiners.{key}.disp_emb.{attribute}"),
                    f"refinement.displacement.{key}.{attribute}",
                    loaded,
                )

            block = head.refinement.blocks[key]
            source_blocks = ["block1", *[f"hidden_blocks.{index}" for index in range(8)]]
            for target_index, source_block in enumerate(source_blocks):
                unit = block.body[target_index].layers
                for module_index, source_name in {
                    0: "conv_depthwise", 1: "norm", 3: "conv_pointwise"
                }.items():
                    module = unit[module_index]
                    for attribute in (
                        "weight", "bias", "running_mean", "running_var", "num_batches_tracked"
                    ):
                        if hasattr(module, attribute):
                            _copy_compatible_tensor(
                                getattr(module, attribute),
                                _find_state_suffix(
                                    state,
                                    f"refiners.{key}.{source_block}.{source_name}.{attribute}",
                                ),
                                f"refinement.blocks.{key}.body.{target_index}."
                                f"{module_index}.{attribute}",
                                loaded,
                            )

            warp_weight = _find_state_suffix(state, f"refiners.{key}.warp_head.weight")
            warp_bias = _find_state_suffix(state, f"refiners.{key}.warp_head.bias")
            confidence_weight = _find_state_suffix(
                state, f"refiners.{key}.confidence_head.weight"
            )
            confidence_bias = _find_state_suffix(
                state, f"refiners.{key}.confidence_head.bias"
            )
            if warp_weight is not None and confidence_weight is not None:
                _copy_compatible_tensor(
                    block.output.weight,
                    torch.cat((warp_weight[:2], confidence_weight[:1]), dim=0),
                    f"refinement.blocks.{key}.output.weight",
                    loaded,
                )
            if warp_bias is not None and confidence_bias is not None:
                _copy_compatible_tensor(
                    block.output.bias,
                    torch.cat((warp_bias[:2], confidence_bias[:1]), dim=0),
                    f"refinement.blocks.{key}.output.bias",
                    loaded,
                )
    if not loaded:
        raise ValueError("checkpoint contained no compatible official RoMaV2 refiner keys")
    return loaded
