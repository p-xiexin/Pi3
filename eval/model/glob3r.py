"""Compose existing Pi3-family backbones with the existing Glob3R head."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Dict

import torch

from pi3.models.glob3r.glob3r_training import Glob3R


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "pytorch_model_1.bin",
        ]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) == 1:
            return existing[0]
        if not existing:
            raise FileNotFoundError(f"no model checkpoint found in {path}")
        raise RuntimeError(f"ambiguous checkpoint directory {path}: {existing}")
    raise FileNotFoundError(path)


def _read_state_dict(path: str | Path) -> Mapping[str, torch.Tensor]:
    path = _checkpoint_file(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, Mapping) and isinstance(state.get(key), Mapping):
            state = state[key]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {path} does not contain a state dict")
    return state


def _strip_prefixes(
    state: Mapping[str, torch.Tensor], prefixes
) -> dict[str, torch.Tensor]:
    normalized = {}
    for original_key, value in state.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        normalized[key] = value
    return normalized


class EvalGlob3R(Glob3R):
    """Expose a common RGB-only evaluation path for Pi3 and Pi3X."""

    def __init__(self, *args, with_prior: bool = False, **kwargs):
        if with_prior:
            raise ValueError("EvalGlob3R requires RGB-only with_prior=false")
        super().__init__(*args, **kwargs)
        self.with_prior = False
        self._uses_pi3x_interface = all(
            callable(getattr(self.backbone, name, None))
            for name in ("encode", "forward_head")
        )

    def _extract_window_features(self, images: torch.Tensor):
        """Run one Pi3-family backbone pass and collect Glob3R features."""

        batch, frames, _, height, width = images.shape
        backbone = self.backbone
        backbone.eval()
        self._captured_encoder_features.clear()
        normalized = (images - backbone.image_mean) / backbone.image_std

        if self._uses_pi3x_interface:
            encoded, relative_poses, _depth_mask, pose_mask, _scale = (
                backbone.encode(normalized, with_prior=False)
            )
            geometry_tokens, positions = backbone.decode(
                encoded.reshape(batch, frames, -1, backbone.dec_embed_dim),
                frames,
                height,
                width,
                relative_poses,
                pose_mask,
            )
        else:
            encoded = backbone.encoder(
                normalized.reshape(batch * frames, 3, height, width),
                is_training=True,
            )
            if isinstance(encoded, dict):
                encoded = encoded["x_norm_patchtokens"]
            geometry_tokens, positions = backbone.decode(
                encoded, frames, height, width
            )

        encoder_features = []
        for layer_index in self.encoder_layers:
            if layer_index not in self._captured_encoder_features:
                raise RuntimeError(
                    f"encoder layer {layer_index} did not produce a feature"
                )
            feature = self._captured_encoder_features[layer_index]
            encoder_features.append(
                feature.reshape(
                    batch, frames, feature.shape[1], feature.shape[2]
                )
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
        """Use native Pi3X forward_head or the existing Pi3 head modules."""

        backbone = self.backbone
        batch = geometry_tokens.shape[0] // frames
        patch_height = height // backbone.patch_size
        patch_width = width // backbone.patch_size
        if self._uses_pi3x_interface:
            geometry = backbone.forward_head(
                geometry_tokens,
                positions,
                batch,
                frames,
                height,
                width,
                patch_height,
                patch_width,
            )
            missing = [
                name
                for name in ("local_points", "conf")
                if name not in geometry
            ]
            if missing:
                raise RuntimeError(
                    f"Pi3X forward_head did not produce {missing}"
                )
            return geometry

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
            raise RuntimeError(
                f"backbone cannot produce evaluation geometry; missing {missing}"
            )

        with torch.amp.autocast(
            device_type=geometry_tokens.device.type, enabled=False
        ):
            point_hidden = backbone.point_decoder(
                geometry_tokens, xpos=positions
            ).float()
            point_output = backbone.point_head(
                [point_hidden[:, backbone.patch_start_idx :]], (height, width)
            ).reshape(batch, frames, height, width, -1)
            xy, depth = point_output.split((2, 1), dim=-1)
            depth = depth.exp()
            local_points = torch.cat((xy * depth, depth), dim=-1)

            camera_hidden = backbone.camera_decoder(
                geometry_tokens, xpos=positions
            ).float()
            camera_poses = backbone.camera_head(
                camera_hidden[:, backbone.patch_start_idx :],
                patch_height,
                patch_width,
            ).reshape(batch, frames, 4, 4)

            confidence_hidden = backbone.conf_decoder(
                geometry_tokens, xpos=positions
            ).float()
            confidence = backbone.conf_head(
                [confidence_hidden[:, backbone.patch_start_idx :]],
                (height, width),
            ).reshape(batch, frames, height, width, -1)

        return {
            "camera_poses": camera_poses,
            "local_points": local_points,
            "conf": confidence,
        }

    @torch.no_grad()
    def match_pair(
        self,
        patch_tokens: torch.Tensor,
        encoder_features: list[torch.Tensor],
        images: torch.Tensor,
        reference_index: int,
    ):
        return self.glob3r_matching_head(
            patch_tokens,
            encoder_features,
            images,
            reference_index=reference_index,
        )


def load_eval_glob3r(
    backbone: torch.nn.Module,
    matching_checkpoint: str | Path,
    backbone_checkpoint: str | Path | None = None,
    with_prior: bool = False,
    device: str | torch.device = "cuda",
) -> EvalGlob3R:
    """Load checkpoints around an already instantiated backbone."""

    if backbone_checkpoint is not None:
        backbone_state = _strip_prefixes(
            _read_state_dict(backbone_checkpoint),
            ("module.", "model.", "backbone."),
        )
        result = backbone.load_state_dict(backbone_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"incomplete backbone checkpoint: {result}")

    model = EvalGlob3R(
        backbone,
        encoder_layers=(5, 11, 17, 23),
        enable_refinement=True,
        matching_checkpoint=None,
        with_prior=with_prior,
    )
    matching_state = _read_state_dict(matching_checkpoint)
    prefix = "glob3r_matching_head."
    matching_state = {
        key.split(prefix, 1)[1]: value
        for key, value in matching_state.items()
        if prefix in key
    } or dict(matching_state)
    model.glob3r_matching_head.load_state_dict(matching_state, strict=True)
    return model.to(device).eval()


__all__ = ["EvalGlob3R", "load_eval_glob3r"]
