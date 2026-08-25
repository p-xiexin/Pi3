"""Glob3R model specialization used only by the SfM inference pipeline."""

from __future__ import annotations

from typing import Dict

import torch

from pi3.models.glob3r.glob3r_training import Glob3R

from utils.timing import tic, toc


class Glob3RSfM(Glob3R):
    """Reuse one frozen Pi3 pass for Eq. (1) and multi-keyframe Eq. (2)."""

    def _extract_window_features(self, Is: torch.Tensor):
        """Extract frozen Pi3 features once for every reference in the window."""

        batch, frames, _, height, width = Is.shape
        backbone = self.backbone
        backbone.eval()
        self._captured_encoder_features.clear()
        normalized = (Is - backbone.image_mean) / backbone.image_std
        encoded = backbone.encoder(
            normalized.reshape(batch * frames, 3, height, width), is_training=True
        )
        if isinstance(encoded, dict):
            encoded = encoded["x_norm_patchtokens"]
        geometry_tokens, positions = backbone.decode(encoded, frames, height, width)

        encoder_features = []
        for layer_index in self.encoder_layers:
            if layer_index not in self._captured_encoder_features:
                raise RuntimeError(f"encoder layer {layer_index} did not produce a feature")
            feature = self._captured_encoder_features[layer_index]
            encoder_features.append(
                feature.reshape(batch, frames, feature.shape[1], feature.shape[2])
            )
        return geometry_tokens, positions, encoder_features

    def _predict_geometry(
        self,
        geometry_tokens: torch.Tensor,
        positions: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor | None]:
        """Decode the geometry tuple ``{T_i, X_i, C_i, m_i}`` in Eq. (1)."""

        backbone = self.backbone
        required = (
            "point_decoder",
            "point_head",
            "conf_decoder",
            "conf_head",
            "camera_decoder",
            "camera_head",
        )
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise RuntimeError(f"backbone cannot produce Eq. (1) geometry; missing {missing}")

        batch = geometry_tokens.shape[0] // frames
        patch_height, patch_width = height // backbone.patch_size, width // backbone.patch_size
        with torch.amp.autocast(device_type=geometry_tokens.device.type, enabled=False):
            point_hidden = backbone.point_decoder(geometry_tokens, xpos=positions).float()
            point_output = backbone.point_head(
                [point_hidden[:, backbone.patch_start_idx:]], (height, width)
            ).reshape(batch, frames, height, width, -1)
            xy, D = point_output.split((2, 1), dim=-1)
            D = D.exp()
            Xs_C = torch.cat((xy * D, D), dim=-1)

            camera_hidden = backbone.camera_decoder(geometry_tokens, xpos=positions).float()
            T_WCs = backbone.camera_head(
                camera_hidden[:, backbone.patch_start_idx:], patch_height, patch_width
            ).reshape(batch, frames, 4, 4)

            confidence_hidden = backbone.conf_decoder(
                geometry_tokens, xpos=positions
            ).float()
            Cs_logits = backbone.conf_head(
                [confidence_hidden[:, backbone.patch_start_idx:]], (height, width)
            ).reshape(batch, frames, height, width, -1)

        return {
            "camera_poses": T_WCs,
            "local_points": Xs_C,
            "conf": Cs_logits,
        }

    @torch.no_grad()
    def infer_window(
        self,
        Is: torch.Tensor,
    ) -> tuple[
        Dict[str, torch.Tensor | None],
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """Run Eq. (1) once and return the features reused by Eq. (2)."""

        if Is.ndim != 5:
            raise ValueError(f"Is must be [B,N,3,H,W], got {tuple(Is.shape)}")
        batch, frames, _, height, width = Is.shape
        if batch != 1:
            raise ValueError("SfM window inference currently requires batch size one")
        if height % self.backbone.patch_size or width % self.backbone.patch_size:
            raise ValueError("image height and width must be divisible by Pi3 patch_size")

        geometry_tokens, positions, encoder_features = self._extract_window_features(Is)
        geometry = self._predict_geometry(
            geometry_tokens, positions, frames, height, width
        )
        patch_start = int(self.backbone.patch_start_idx)
        patch_tokens = geometry_tokens.reshape(
            batch, frames, geometry_tokens.shape[1], geometry_tokens.shape[2]
        )[:, :, patch_start:]
        return geometry, patch_tokens, encoder_features

    @torch.no_grad()
    def match_pair(
        self,
        patch_tokens: torch.Tensor,
        encoder_features: list[torch.Tensor],
        Is: torch.Tensor,
        reference_index: int,
    ):
        """Run Eq. (2) for one reference using cached window features."""

        tic()
        output = self.glob3r_matching_head(
            patch_tokens,
            encoder_features,
            Is,
            reference_index=reference_index,
        )
        toc(f"Glob3R matching forward reference {reference_index}")
        return output

__all__ = ["Glob3RSfM"]
