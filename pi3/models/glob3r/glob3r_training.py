"""Use the repository training Pi3 as a frozen Glob3R geometry backbone."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .model import Glob3RMatchingHead


class Glob3R(nn.Module):
    """Compose a frozen large Pi3 backbone with the Glob3R matching head."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder_layers: Sequence[int] = (5, 11, 17, 23),
        enable_refinement: bool = True,
        matching_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder_layers = tuple(int(index) for index in encoder_layers)
        if len(self.encoder_layers) != 4:
            raise ValueError("Glob3R requires exactly four Pi3 encoder layers")

        encoder_dim = int(
            getattr(backbone.encoder, "embed_dim", backbone.dec_embed_dim)
        )
        decoder_dim = int(backbone.dec_embed_dim)
        patch_size = int(backbone.patch_size)
        if encoder_dim != 1024 or decoder_dim != 1024 or patch_size != 14:
            raise ValueError(
                "Glob3R currently requires Pi3 decoder_size='large' "
                "and patch_size=14 "
                f"(encoder_dim={encoder_dim}, decoder_dim={decoder_dim}, "
                f"patch_size={patch_size})"
            )

        self.glob3r_matching_head = Glob3RMatchingHead(
            encoder_dim=encoder_dim,
            geometry_dim=2 * decoder_dim,
            match_dim=encoder_dim,
            patch_size=patch_size,
            enable_refinement=enable_refinement,
        )
        self._captured_encoder_features: Dict[int, torch.Tensor] = {}
        self._encoder_hook_handles = []
        blocks = backbone.encoder.blocks
        if len(blocks) and isinstance(blocks[0], nn.ModuleList):
            blocks = [
                block
                for chunk in blocks
                for block in chunk
                if not isinstance(block, nn.Identity)
            ]
        else:
            blocks = list(blocks)
        for layer_index in self.encoder_layers:
            if layer_index < 0 or layer_index >= len(blocks):
                raise ValueError(
                    f"encoder layer {layer_index} does not exist (depth={len(blocks)})"
                )

            def capture_encoder(_module, _inputs, output, index=layer_index):
                normalized = self.backbone.encoder.norm(output)
                token_start = 1 + int(
                    getattr(self.backbone.encoder, "num_register_tokens", 0)
                )
                # [BN,1+R+M,C] -> [BN,M,C]: remove DINO CLS/register tokens.
                self._captured_encoder_features[index] = normalized[:, token_start:]

            self._encoder_hook_handles.append(
                blocks[layer_index].register_forward_hook(capture_encoder)
            )

        self._freeze_backbone()
        if matching_checkpoint is not None:
            self._load_matching_checkpoint(matching_checkpoint)

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.glob3r_matching_head.parameters():
            parameter.requires_grad_(True)
        self.backbone.eval()

    def configure_stage(self, stage: str) -> None:
        """Apply Appendix B's separate coarse/refinement training protocol."""

        if stage not in {"coarse", "refinement", "joint"}:
            raise ValueError(f"unknown Glob3R training stage: {stage}")
        head = self.glob3r_matching_head
        head.refinement_active = stage in {"refinement", "joint"}
        coarse_modules = (head.match_decoder, head.dpt_match)
        refinement_modules = tuple(
            module for module in (head.fine_features, head.refinement) if module is not None
        )
        for module in (*coarse_modules, *refinement_modules):
            module.requires_grad_(stage == "joint")
        if stage == "coarse":
            for module in coarse_modules:
                module.requires_grad_(True)
        elif stage == "refinement":
            if not refinement_modules:
                raise ValueError("refinement training requires enable_refinement=true")
            for module in refinement_modules:
                module.requires_grad_(True)

    def _load_matching_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            checkpoint = checkpoint["model"]
        prefix = "glob3r_matching_head."
        state = {
            key.split(prefix, 1)[1]: value
            for key, value in checkpoint.items()
            if prefix in key
        }
        self.glob3r_matching_head.load_state_dict(state or checkpoint, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        reference_index: int = 0,
    ) -> Dict[str, object]:
        if images.ndim != 5:
            raise ValueError(f"images must be [B,N,3,H,W], got {tuple(images.shape)}")
        batch, frames, _, height, width = images.shape
        if frames < 2:
            raise ValueError("Glob3R matching requires at least two frames")
        if not 0 <= reference_index < frames:
            raise ValueError(f"reference_index {reference_index} is invalid for {frames} frames")
        if height % self.backbone.patch_size or width % self.backbone.patch_size:
            raise ValueError("image height and width must be divisible by Pi3 patch_size")
        self.backbone.eval()
        self._captured_encoder_features.clear()

        normalized_images = (images - self.backbone.image_mean) / self.backbone.image_std
        with torch.no_grad():
            encoded = self.backbone.encoder(
                normalized_images.reshape(batch * frames, 3, height, width),
                is_training=True,
            )
            if isinstance(encoded, dict):
                encoded = encoded["x_norm_patchtokens"]
            geometry_tokens, _ = self.backbone.decode(encoded, frames, height, width)

        encoder_features = []
        for layer_index in self.encoder_layers:
            if layer_index not in self._captured_encoder_features:
                raise RuntimeError(f"encoder layer {layer_index} did not produce a feature")
            feature = self._captured_encoder_features[layer_index]
            encoder_features.append(
                feature.reshape(batch, frames, feature.shape[1], feature.shape[2])
            )

        patch_start = int(self.backbone.patch_start_idx)
        # [BN,5+M,2C] -> [B,N,M,2C]: remove Pi3 special tokens.
        geometry_tokens = geometry_tokens.reshape(
            batch, frames, geometry_tokens.shape[1], geometry_tokens.shape[2]
        )[:, :, patch_start:]
        matching = self.glob3r_matching_head(
            geometry_tokens,
            encoder_features,
            images,
            reference_index=reference_index,
        )
        return {
            "match_similarity": matching.similarity,
            "coarse_warp": matching.coarse_warp,
            "coarse_match_confidence": matching.coarse_confidence,
            "coarse_match_confidence_logits": matching.coarse_confidence_logits,
            "warp_stages": matching.warp_stages,
            "match_confidence_stages": matching.confidence_stages,
            "match_confidence_logits_stages": matching.confidence_logits_stages,
            "target_indices": matching.target_indices,
            "reference_index": reference_index,
            "warp": matching.warp_stages[-1] if matching.warp_stages else matching.coarse_warp,
            "warp_confidence": (
                matching.confidence_stages[-1]
                if matching.confidence_stages
                else matching.coarse_confidence
            ),
        }


__all__ = ["Glob3R"]
