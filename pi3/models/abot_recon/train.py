"""Stage I trainer and single-config entry point for server migration."""

from itertools import islice

import hydra
from omegaconf import DictConfig

from trainers.base_trainer_accelerate import BaseTrainer
from trainers.pi3_trainer import Pi3Trainer

from .data import prepare_abot_batch, validate_training_batch
from .ema import ABotReconEMA
from .viz import ABotReconTensorBoardVisualizer


class _LimitedLoader:
    """Bound validation iterations without changing the shared trainer."""

    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = min(int(limit), len(loader))

    def __iter__(self):
        return islice(iter(self.loader), self.limit)

    def __len__(self):
        return self.limit

    def __getattr__(self, name):
        return getattr(self.loader, name)


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

    def auto_resume(self):
        """Create and register EMA before Accelerate restores checkpoint state."""

        self.ema = None
        self._ema_step_hook = None
        if bool(self.cfg.train.get("use_ema", False)):
            if self.accelerator.state.deepspeed_plugin is not None:
                raise RuntimeError("ABot-Recon EMA does not support DeepSpeed")
            model = self.accelerator.unwrap_model(self.model)
            self.ema = ABotReconEMA(
                model,
                decay=float(self.cfg.train.get("ema_decay", 0.999)),
            )
            self.accelerator.register_for_checkpointing(self.ema)
            optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            self._ema_step_hook = optimizer.register_step_post_hook(
                self._update_ema_after_step
            )
            self.log_info(
                f"ABot-Recon EMA enabled with decay={self.ema.decay}"
            )
        return super().auto_resume()

    def _update_ema_after_step(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        self.ema.update(self.accelerator.unwrap_model(self.model))

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
        training_model = self.model
        test_loader = self.test_loader
        if 0 < self.iters_per_test < len(test_loader):
            self.test_loader = _LimitedLoader(test_loader, self.iters_per_test)
        if self.ema is not None:
            self.model = self.ema.module
        try:
            return super().validate(epoch)
        finally:
            self.model = training_model
            self.test_loader = test_loader


@hydra.main(version_base="1.2", config_path=".", config_name="stage1")
def main(cfg: DictConfig) -> None:
    ABotReconTrainer(cfg).train()


if __name__ == "__main__":
    main()
