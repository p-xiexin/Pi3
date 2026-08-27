"""Pi3 geometry plus the selected Glob3R or VGGSfM tracks model."""

from pathlib import Path

import torch

from pi3.models.glob3r.glob3r_training import Glob3R
from pi3.models.pi3 import Pi3


def _decode_geometry(backbone, tokens, positions, batch, frames, height, width):
    """Decode Pi3 tokens into per-frame point maps, poses, and confidence logits."""
    with torch.amp.autocast(device_type=tokens.device.type, enabled=False):
        hidden = backbone.point_decoder(tokens, xpos=positions).float()
        points = backbone.point_head(
            [hidden[:, backbone.patch_start_idx:]], (height, width)
        ).reshape(batch, frames, height, width, -1)
        xy, depth = points.split((2, 1), dim=-1)
        depth = depth.exp()
        points = torch.cat((xy * depth, depth), dim=-1)
        hidden = backbone.camera_decoder(tokens, xpos=positions).float()
        poses = backbone.camera_head(
            hidden[:, backbone.patch_start_idx:], height // 14, width // 14
        ).reshape(batch, frames, 4, 4)
        hidden = backbone.conf_decoder(tokens, xpos=positions).float()
        confidence = backbone.conf_head(
            [hidden[:, backbone.patch_start_idx:]], (height, width)
        ).reshape(batch, frames, height, width, -1)
    return {"local_points": points, "camera_poses": poses, "conf": confidence}


class Glob3RSfM(Glob3R):
    """Share one Pi3 encoding pass between geometry and Glob3R matching."""

    def _features(self, images):
        """Return decoder tokens plus the encoder levels consumed by Glob3R."""
        batch, frames, _, height, width = images.shape
        self.backbone.eval()
        self._captured_encoder_features.clear()
        normalized = (images - self.backbone.image_mean) / self.backbone.image_std
        encoded = self.backbone.encoder(
            normalized.reshape(batch * frames, 3, height, width), is_training=True
        )
        if isinstance(encoded, dict):
            encoded = encoded["x_norm_patchtokens"]
        tokens, positions = self.backbone.decode(encoded, frames, height, width)
        encoder = []
        for layer in self.encoder_layers:
            if layer not in self._captured_encoder_features:
                raise RuntimeError(f"encoder layer {layer} did not produce a feature")
            x = self._captured_encoder_features[layer]
            encoder.append(x.reshape(batch, frames, x.shape[1], x.shape[2]))
        return tokens, positions, encoder

    def _matching_state(self, tokens, encoder, batch, frames):
        backbone = self.backbone
        patch_tokens = tokens.reshape(
            batch, frames, tokens.shape[1], tokens.shape[2]
        )
        patch_tokens = patch_tokens[:, :, int(backbone.patch_start_idx):]
        return patch_tokens, encoder

    @torch.no_grad()
    def encode_matching_state(self, images):
        """Encode masked RGB when it differs from the geometry-stage input."""
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("images must have shape [1,N,3,H,W]")
        batch, frames = images.shape[:2]
        tokens, _, encoder = self._features(images)
        return self._matching_state(tokens, encoder, batch, frames)

    @torch.no_grad()
    def infer_window(self, images):
        """Infer geometry and reusable matching state for ``[1,N,3,H,W]`` images."""
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("images must have shape [1,N,3,H,W]")
        batch, frames, _, height, width = images.shape
        tokens, positions, encoder = self._features(images)
        backbone = self.backbone
        geometry = _decode_geometry(backbone, tokens, positions, batch, frames, height, width)
        return geometry, self._matching_state(tokens, encoder, batch, frames)

    @torch.no_grad()
    def match_pair(self, patch_tokens, encoder_features, images, reference_index):
        """Run the dense Glob3R head for one reference frame."""
        return self.glob3r_matching_head(
            patch_tokens, encoder_features, images, reference_index=reference_index
        )


class Pi3Geometry(torch.nn.Module):
    """Geometry-only Pi3 frontend used with the independent VGGSfM tracker."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def infer_window(self, images):
        """Infer geometry without retaining tracker-specific feature state."""
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("images must have shape [1,N,3,H,W]")
        batch, frames, _, height, width = images.shape
        backbone = self.backbone
        normalized = (images - backbone.image_mean) / backbone.image_std
        encoded = backbone.encoder(
            normalized.reshape(batch * frames, 3, height, width), is_training=True
        )
        if isinstance(encoded, dict):
            encoded = encoded["x_norm_patchtokens"]
        tokens, positions = backbone.decode(encoded, frames, height, width)
        return _decode_geometry(backbone, tokens, positions, batch, frames, height, width), None


def _checkpoint_file(path):
    """Require the exact checkpoint file declared in YAML."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def _state_dict(path):
    """Read a local raw state dict and reject nested training checkpoints."""
    path = _checkpoint_file(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise TypeError(f"checkpoint {path} must contain a raw tensor state dict")
    return state


def _load_backbone(checkpoint):
    """Construct Pi3 and strictly load its exact parameter namespace."""
    backbone = Pi3(pos_type="rope100", decoder_size="large")
    backbone.load_state_dict(_state_dict(checkpoint), strict=True)
    return backbone


def load_models(config):
    """Create geometry and tracks frontends; downstream code uses one shared API."""
    from .tracks import Glob3RTracks, VGGSfMTracks, load_vggsfm_tracker
    device = config["device"]
    backbone = _load_backbone(config["backbone_checkpoint"])
    if config["tracks_model"] == "glob3r":
        geometry = Glob3RSfM(
            backbone, encoder_layers=(5, 11, 17, 23), enable_refinement=True
        )
        matching = _state_dict(config["matching_checkpoint"])
        prefix = "glob3r_matching_head."
        matching = {
            key[len(prefix):]: value
            for key, value in matching.items()
            if key.startswith(prefix)
        }
        if not matching:
            raise RuntimeError(
                f"matching checkpoint must contain parameters under {prefix}"
            )
        geometry.glob3r_matching_head.load_state_dict(matching, strict=True)
        geometry = geometry.to(device).eval()
        return geometry, Glob3RTracks(geometry)
    geometry = Pi3Geometry(backbone).to(device).eval()
    tracker = load_vggsfm_tracker(
        config["vggsfm_root"], config["vggsfm_checkpoint"], device
    )
    return geometry, VGGSfMTracks(
        tracker,
        config["image_size"],
        visibility_threshold=float(
            config.get("vgg_track_visibility_threshold", 0.05)
        ),
        score_threshold=float(config.get("vgg_track_score_threshold", 0.5)),
    )


__all__ = ["Glob3RSfM", "Pi3Geometry", "load_models"]
