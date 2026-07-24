"""Training adapter for the Glob3R branch.

The entry point remains ``scripts/train_pi3.py``.  This trainer loads the Pi3
geometry weights and limits the complete Glob3R model to the requested stage.
"""

from collections.abc import Mapping

import hydra
import torch

from trainers.pi3_trainer import Pi3Trainer
from trainers.base_trainer_accelerate import BaseTrainer
from utils.basic import count_parameters
from utils.glob3r_visualization import Glob3RTensorBoardVisualizer
from pi3.models.glob3r import Glob3R, load_romav2_refinement


def _load_geometry_backbone(backbone, checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for container_key in ("model", "state_dict", "model_state_dict"):
            if isinstance(state_dict, Mapping) and isinstance(
                state_dict.get(container_key), Mapping
            ):
                state_dict = state_dict[container_key]

    if not isinstance(state_dict, Mapping):
        raise TypeError("Pi3 checkpoint must contain a state-dict mapping")

    expected = {
        key
        for key in backbone.state_dict()
        if key == "register_token" or key.startswith(("encoder.", "decoder."))
    }
    filtered = {}
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "backbone."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
                    changed = True
        if normalized_key in expected:
            filtered[normalized_key] = value

    missing = sorted(expected.difference(filtered))
    if missing:
        raise RuntimeError(
            "Pi3 checkpoint is missing geometry-backbone tensors: "
            + ", ".join(missing[:10])
        )
    backbone.load_state_dict(filtered, strict=False)


class Glob3RTrainer(Pi3Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.visualizer = Glob3RTensorBoardVisualizer(
            self.cfg.glob3r.get("visualization", {}),
            self.cfg.train.gradient_accumulation_steps,
            self.initial_global_step,
        )

    def prepare_model(self):
        options = self.cfg.glob3r
        checkpoint_path = options.backbone_checkpoint
        if bool(options.require_pretrained_backbone) and checkpoint_path is None:
            raise ValueError("set glob3r.backbone_checkpoint to a pretrained Pi3 checkpoint")
        backbone = hydra.utils.instantiate(self.cfg.model)
        if checkpoint_path is not None:
            _load_geometry_backbone(backbone, checkpoint_path)
        model = Glob3R(
            backbone,
            encoder_layers=tuple(options.encoder_layers),
            enable_refinement=bool(options.enable_refinement),
            matching_checkpoint=options.matching_checkpoint,
        )
        if str(options.stage) == "refinement" and options.romav2_refinement_checkpoint is not None:
            loaded = load_romav2_refinement(model, str(options.romav2_refinement_checkpoint))
            print(f"Loaded {len(loaded)} compatible RoMaV2 refinement tensors")
        model.configure_stage(str(options.stage))
        count_parameters(model)
        return model

    def build_optimizer(self, cfg_optimizer, model):
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("Glob3R has no trainable matching parameters")
        unexpected = [name for name, _ in trainable if not name.startswith("glob3r_matching_head.")]
        if unexpected:
            raise RuntimeError(f"frozen Pi3 parameters leaked into optimizer: {unexpected[:5]}")

        decay = [parameter for name, parameter in trainable if parameter.ndim > 1 and not name.endswith(".bias")]
        no_decay = [parameter for name, parameter in trainable if parameter.ndim <= 1 or name.endswith(".bias")]
        groups = [
            {"params": decay, "weight_decay": cfg_optimizer.weight_decay, "lr": cfg_optimizer.lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": cfg_optimizer.lr},
        ]
        return BaseTrainer.build_optimizer(
            self, cfg_optimizer, model, param_group_fn=lambda _model: groups
        )

    def forward_batch(self, batch, mode="train"):
        images = torch.stack([view["img"] for view in batch], dim=1)
        prediction = self.model(images, reference_index=0)
        return [prediction, batch]

    def calculate_loss(self, output, batch, mode="train"):
        result = super().calculate_loss(output, batch, mode)
        self.visualizer.log(self.accelerator, output, mode)
        return result

    def validate(self, epoch):
        self.visualizer.begin_validation(epoch)
        return super().validate(epoch)
