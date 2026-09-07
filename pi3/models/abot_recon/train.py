"""Stage I trainer and single-config entry point for server migration."""

import hydra
from omegaconf import DictConfig

from trainers.base_trainer_accelerate import BaseTrainer
from trainers.pi3_trainer import Pi3Trainer

from .data import prepare_abot_batch, validate_training_batch
from .viz import ABotReconTensorBoardVisualizer


class ABotReconTrainer(Pi3Trainer):
    """Connect ordered ABot-Recon batches and losses to Pi3 training."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.test_loss)
        self.visualizer = ABotReconTensorBoardVisualizer(
            self.cfg.abot_recon.get("visualization", {}),
            self.cfg.train.gradient_accumulation_steps,
            self.initial_global_step,
            self.train_loss,
            self.test_loss,
        )

    def build_optimizer(self, cfg_optimizer, model):
        """Create encoder, base-model, and rotation-refiner LR groups."""

        def groups(model_):
            encoder, refiner, other = [], [], []
            for name, parameter in model_.named_parameters():
                item = (name, parameter)
                if name.startswith("encoder."):
                    encoder.append(item)
                elif name.startswith("camera_head.rot_correction."):
                    refiner.append(item)
                else:
                    other.append(item)

            def decay_split(named_parameters, learning_rate):
                decay, no_decay = [], []
                for name, parameter in named_parameters:
                    if not parameter.requires_grad:
                        continue
                    target = (
                        no_decay
                        if parameter.ndim <= 1 or name.endswith(".bias")
                        else decay
                    )
                    target.append(parameter)
                return [
                    {
                        "params": no_decay,
                        "weight_decay": 0.0,
                        "lr": learning_rate,
                    },
                    {
                        "params": decay,
                        "weight_decay": cfg_optimizer.weight_decay,
                        "lr": learning_rate,
                    },
                ]

            result = []
            result.extend(decay_split(encoder, cfg_optimizer.encoder_lr))
            result.extend(decay_split(other, cfg_optimizer.lr))
            result.extend(
                decay_split(
                    refiner,
                    cfg_optimizer.get("refiner_lr", cfg_optimizer.lr),
                )
            )
            return [group for group in result if group["params"]]

        return BaseTrainer.build_optimizer(
            self, cfg_optimizer, model, param_group_fn=groups
        )

    def forward_batch(self, batch, mode="train"):
        sequence = prepare_abot_batch(batch)
        if mode == "train":
            validate_training_batch(sequence)
        prediction = self.model(sequence["imgs"])
        return [prediction, sequence]

    def calculate_loss(self, output, batch, mode="train"):
        result = super().calculate_loss(output, batch, mode)
        self.visualizer.log(self.accelerator, output, mode)
        return result

    def validate(self, epoch):
        self.visualizer.begin_validation(epoch)
        return super().validate(epoch)


@hydra.main(version_base="1.2", config_path=".", config_name="stage1")
def main(cfg: DictConfig) -> None:
    ABotReconTrainer(cfg).train()


if __name__ == "__main__":
    main()
