"""Frozen ABotRecon features and coarse-only streaming matching training.

Run with python -m pi3.models.abot_recon.match_head.train and coarse.yaml
overrides. glob3r.backbone_checkpoint must name a complete ABotRecon state dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..model import ABotRecon, _checkpoint_state
from .head import StreamingMatchHead, MatchState, DecoderStreamState, decode_feature_step


@dataclass(frozen=True)
class MatchingStreamState:
    backbone: DecoderStreamState
    matching: MatchState


def sample_adjacent_pairs(frames, max_pairs=4, generator=None):
    if frames < 2 or max_pairs < 1:
        raise ValueError("matching needs at least two frames and one pair")
    count = min(frames - 1, max_pairs)
    others = torch.randperm(frames - 2, generator=generator)[:count - 1] + 1
    return torch.cat((others, torch.tensor([frames - 1]))).sort().values


class ABotReconMatch(nn.Module):
    """Frozen causal backbone plus trainable adjacent Glob3R coarse layers."""

    def __init__(self, backbone, encoder_layers=(5, 11, 17, 23), match_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.encoder_layers = tuple(encoder_layers)
        if len(self.encoder_layers) != 4 or sorted(set(self.encoder_layers)) != list(self.encoder_layers):
            raise ValueError("four increasing DINO layer indices are required")
        encoder = backbone.encoder
        depth = len(encoder.blocks[-1]) if encoder.chunked_blocks else len(encoder.blocks)
        if self.encoder_layers[0] < 0 or self.encoder_layers[-1] != depth - 1:
            raise ValueError("the final skip must be the encoder's final block")
        self.match_head = StreamingMatchHead(
            encoder_dim=encoder.embed_dim, geometry_dim=2 * backbone.dec_embed_dim,
            match_dim=match_dim, patch_size=backbone.patch_size)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_pretrained(cls, checkpoint, matching_checkpoint=None, backbone_options=None, **kwargs):
        state = _checkpoint_state(checkpoint)
        if not all(key in state for key in ("camera_head.delta_t_head.weight", "camera_head.delta_q_head.weight")):
            raise ValueError("a released ABotRecon checkpoint is required, not Pi3 initialization")
        options = dict(backbone_options or {})
        if any(key in options for key in ("ckpt", "load_pi3", "load_vggt", "enable_confidence")):
            raise ValueError("backbone initialization is controlled by checkpoint")
        options["enable_confidence"] = any(key.startswith("conf_head.") for key in state)
        backbone = ABotRecon(**options)
        # Freezing a partially initialized backbone would silently train on random features.
        backbone.load_state_dict(state, strict=True)
        del state
        model = cls(backbone, **kwargs)
        if matching_checkpoint is not None:
            model.load_matching_checkpoint(matching_checkpoint)
        return model

    def load_matching_checkpoint(self, path):
        state = _checkpoint_state(path)
        for prefix in ("match_head.", "glob3r_matching_head."):
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if selected:
                state = selected
                break
        # A complete Glob3R checkpoint may contain the disabled fine branch.
        state = {key: value for key, value in state.items()
                 if not key.startswith(("fine_features.", "refinement."))}
        self.match_head.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _feature_step(self, image, state=None):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be [B,3,H,W]")
        height, width = image.shape[-2:]
        if height % self.backbone.patch_size or width % self.backbone.patch_size:
            raise ValueError("image dimensions must be divisible by patch size")
        self.backbone.eval()
        normalized = (image - self.backbone.image_mean) / self.backbone.image_std
        features = self.backbone.encoder.get_intermediate_layers(normalized, n=self.encoder_layers, norm=True)
        geometry, _, next_state = decode_feature_step(self.backbone, features[-1], height, width, state)
        return geometry[:, self.backbone.patch_start_idx:], features, next_state

    def forward(self, images, target_indices=None):
        if images.ndim != 5 or images.shape[1] < 2:
            raise ValueError("images must be [B,N,3,H,W] with N >= 2")
        frames = images.shape[1]
        targets = torch.as_tensor(list(range(1, frames)) if target_indices is None else target_indices, dtype=torch.long)
        if targets.ndim != 1:
            raise ValueError("target_indices must be a vector")
        targets_list = targets.tolist()
        if not targets_list or len(set(targets_list)) != len(targets_list) or min(targets_list) < 1 or max(targets_list) >= frames:
            raise ValueError("target_indices must select unique noninitial clip frames")
        selected = sorted(set(targets_list + [index - 1 for index in targets_list]))
        selected_set = set(selected)
        geometries, skips, state = [], [[] for _ in range(4)], None
        # Process the complete prefix, retaining only selected frames for head training.
        for frame in range(max(targets_list) + 1):
            geometry, features, state = self._feature_step(images[:, frame], state)
            if frame in selected_set:
                geometries.append(geometry)
                for values, feature in zip(skips, features):
                    values.append(feature)
        local_targets = [selected.index(frame) for frame in targets_list]
        result = self.match_head(torch.stack(geometries, 1), [torch.stack(values, 1) for values in skips], images.shape[-2:], local_targets)
        result["target_indices"] = targets.to(images.device)
        result["reference_indices"] = result["target_indices"] - 1
        return result

    @torch.no_grad()
    def forward_step(self, image, state=None):
        if self.training:
            raise RuntimeError("forward_step requires eval mode")
        geometry, features, backbone_state = self._feature_step(image, None if state is None else state.backbone)
        prediction, match_state = self.match_head.forward_step(
            geometry, features, image.shape[-2:], None if state is None else state.matching)
        return prediction, MatchingStreamState(backbone_state, match_state)


def build_optimizer(cfg_optimizer, model):
    from utils.optimizer import build_optimizer as create_optimizer

    groups = {True: [], False: []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith("match_head."):
            raise RuntimeError(f"unexpected trainable backbone parameter {name}")
        no_decay = cfg_optimizer.get("filter_bias_and_bn", True) and (parameter.ndim <= 1 or name.endswith(".bias"))
        groups[no_decay].append(parameter)
    if not any(groups.values()):
        raise ValueError("no trainable matching parameters")
    param_groups = [
        {"params": groups[False], "weight_decay": cfg_optimizer.weight_decay},
        {"params": groups[True], "weight_decay": 0.0},
    ]
    return create_optimizer(cfg_optimizer, model, param_group_fn=lambda _: param_groups,
                            betas=tuple(cfg_optimizer.betas))


def train_step(model, batch, criterion, accelerator, optimizer, lr_scheduler, max_pairs=4, clip_grad=1.0):
    """One microbatch; scheduler advances only after a successful optimizer update."""
    images = torch.stack([view["img"] for view in batch], 1)
    targets = sample_adjacent_pairs(images.shape[1], max_pairs)
    with accelerator.accumulate(model):
        with accelerator.autocast():
            predictions = model(images, target_indices=targets)
            loss, details = criterion(predictions, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite coarse matching loss")
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), clip_grad)
        optimizer.step()
        updated = accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
        if updated:
            # The shared Glob3R scheduler uses micro-step indices.
            for _ in range(accelerator.gradient_accumulation_steps):
                lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": loss.detach(), **details}, updated


@dataclass
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    iters_per_epoch: int = 0
    gradient_accumulation_steps: int = 1
    best_val_metric: float = float("inf")

    def state_dict(self):
        return dict(global_step=self.global_step, epoch=self.epoch,
                    iters_per_epoch=self.iters_per_epoch,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    best_val_metric=self.best_val_metric)

    def load_state_dict(self, state):
        if "global_step" not in state:
            raise ValueError("This checkpoint predates epoch-based training. Load its match_head.pt "
                             "with glob3r.matching_checkpoint to start a new schedule.")
        for key in ("iters_per_epoch", "gradient_accumulation_steps"):
            if int(state[key]) != getattr(self, key):
                raise ValueError(f"train.{key} must match the resumed checkpoint")
        self.global_step, self.epoch = int(state["global_step"]), int(state["epoch"])
        self.best_val_metric = float(state["best_val_metric"])


def register_matching_checkpoints(accelerator):
    """Save matching weights and optimizer state; reload the backbone from its release."""
    from pathlib import Path
    def save(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                torch.save(accelerator.unwrap_model(model).match_head.state_dict(), Path(output_dir) / "match_head.pt")
        weights.clear()
    def load(models, input_dir):
        for model in models:
            accelerator.unwrap_model(model).load_matching_checkpoint(Path(input_dir) / "match_head.pt")
        models.clear()
    accelerator.register_save_state_pre_hook(save)
    accelerator.register_load_state_pre_hook(load)


def run_training(cfg, model=None, train_dataset=None, test_dataset=None):
    """Train the adjacent matching head with the Glob3R epoch/iteration config."""
    import datetime
    from pathlib import Path
    import time
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import GradientAccumulationPlugin, set_seed, send_to_device
    from torch.utils.data import DataLoader
    from datasets.base.batched_sampler import DynamicBatchSampler, DynamicDistributedSampler
    from datasets.base.base_dataset import unified_collate_fn
    from utils.dist import MetricLogger, SmoothedValue
    from utils.misc import get_logger
    from utils.pylogger import RankedLogger
    from utils.scheduler import build_scheduler
    from .loss import AdjacentMatchingLoss
    from .viz import MatchTensorBoard

    accumulation = int(cfg.train.gradient_accumulation_steps)
    if accumulation < 1 or cfg.glob3r.max_pairs < 1 or cfg.train.clip_grad <= 0:
        raise ValueError("gradient_accumulation_steps, max_pairs and clip_grad must be positive")
    if cfg.train.num_epoch < 1 or cfg.train.iters_per_epoch == 0 or cfg.train.iters_per_epoch < -1:
        raise ValueError("num_epoch must be positive; iters_per_epoch must be positive or -1")
    if cfg.test.iters_per_test == 0 or cfg.test.iters_per_test < -1:
        raise ValueError("iters_per_test must be positive or -1")
    if min(cfg.log.ckpt_interval, cfg.log.summary_interval, cfg.train.print_freq, cfg.test.print_freq) < 1:
        raise ValueError("checkpoint, summary and print intervals must be positive")
    if cfg.train.lr_scheduler.type != "CosineWithWarmupLR" or cfg.train.lr_scheduler.accumulation_steps != accumulation:
        raise ValueError("use CosineWithWarmupLR with the training gradient_accumulation_steps")
    if cfg.test.image_num_range[0] != cfg.test.image_num_range[1]:
        raise ValueError("test.image_num_range must specify one fixed clip length")
    lengths, probabilities = cfg.train.clip_lengths, cfg.train.clip_probabilities
    if not lengths or len(lengths) != len(probabilities) or min(probabilities) < 0 or sum(probabilities) <= 0:
        raise ValueError("clip_lengths and clip_probabilities must specify a valid sampling distribution")
    if min(lengths) < max(2, cfg.train.image_num_range[0]) or max(lengths) > cfg.train.image_num_range[1]:
        raise ValueError("train.clip_lengths must lie within train.image_num_range and contain at least two frames")
    accelerator = Accelerator(
        cpu=bool(cfg.cpu), mixed_precision=cfg.train.model_dtype,
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=accumulation, sync_with_dataloader=False),
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
    output_dir, ckpt_dir = Path(cfg.log.output_dir), Path(cfg.log.ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.job_logging_cfg = HydraConfig.get().job_logging
    get_logger(cfg, Path(__file__).name)
    logger = RankedLogger(Path(__file__).name, rank_zero_only=True)
    set_seed(int(cfg.train.seed))
    logger.info(str(accelerator.state))
    logger.info("Preparing model...")
    if model is None:
        model = ABotReconMatch.from_pretrained(cfg.glob3r.backbone_checkpoint, cfg.glob3r.matching_checkpoint)
    logger.info(f"Trainable matching parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("Making train dataloader...")
    if train_dataset is None:
        train_dataset = hydra.utils.instantiate(cfg.train_dataset.ScanNet, frame_num=max(cfg.train.clip_lengths))
    logger.info("Making test dataloader...")
    if test_dataset is None:
        test_dataset = hydra.utils.instantiate(cfg.test_dataset.ScanNet)
    if not len(train_dataset) or not len(test_dataset):
        raise ValueError("training and validation datasets must not be empty")
    train_sampler = DynamicDistributedSampler(train_dataset, num_replicas=accelerator.num_processes,
                                              rank=accelerator.process_index, shuffle=True, seed=cfg.train.base_seed)
    train_batch_sampler = DynamicBatchSampler(
        train_sampler, len(cfg.train.resolution), [min(cfg.train.clip_lengths), max(cfg.train.clip_lengths)],
        seed=cfg.train.base_seed, rank=accelerator.process_index, max_img_per_gpu=min(cfg.train.clip_lengths))
    train_batch_sampler.possible_nums = list(cfg.train.clip_lengths)
    train_batch_sampler.normalized_weights = [weight / sum(cfg.train.clip_probabilities) for weight in cfg.train.clip_probabilities]
    train_generator = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=cfg.train.num_workers,
                              collate_fn=unified_collate_fn, generator=train_generator,
                              pin_memory=accelerator.device.type == "cuda")
    test_sampler = DynamicDistributedSampler(test_dataset, num_replicas=accelerator.num_processes,
                                             rank=accelerator.process_index, shuffle=False, seed=cfg.train.base_seed)
    test_batch_sampler = DynamicBatchSampler(
        test_sampler, len(cfg.train.resolution), list(cfg.test.image_num_range), seed=cfg.train.base_seed,
        rank=accelerator.process_index, max_img_per_gpu=cfg.test.image_num_range[1])
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=cfg.test.num_workers,
                             collate_fn=unified_collate_fn, generator=torch.Generator())
    # There is one clip per batch; DynamicBatchSampler.__len__ is only an estimate.
    batches_per_pass = len(train_sampler)
    iters_per_epoch = cfg.train.iters_per_epoch if cfg.train.iters_per_epoch > 0 else batches_per_pass
    iters_per_test = min(cfg.test.iters_per_test, len(test_sampler)) if cfg.test.iters_per_test > 0 else len(test_sampler)
    if iters_per_epoch % accumulation:
        raise ValueError("train.iters_per_epoch must be divisible by train.gradient_accumulation_steps for epoch checkpoints")
    passes_per_epoch = (iters_per_epoch + batches_per_pass - 1) // batches_per_pass
    with open_dict(cfg):
        cfg.train.lr_scheduler.total_steps = cfg.train.num_epoch * iters_per_epoch
    optimizer = build_optimizer(cfg.train.optimizer, model)
    lr_scheduler = build_scheduler(cfg.train.lr_scheduler, optimizer)
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(lr_scheduler)
    progress = TrainingProgress(iters_per_epoch=iters_per_epoch, gradient_accumulation_steps=accumulation)
    accelerator.register_for_checkpointing(progress)
    register_matching_checkpoints(accelerator)
    criterion = AdjacentMatchingLoss()
    if cfg.train.resume:
        logger.info(f"Resuming from checkpoint {cfg.train.resume}")
        accelerator.load_state(cfg.train.resume)
    if progress.epoch > cfg.train.num_epoch:
        raise ValueError("checkpoint has more completed epochs than train.num_epoch")
    if lr_scheduler.last_epoch != progress.global_step * accumulation:
        raise ValueError("checkpoint scheduler and global_step disagree")
    optimizer.zero_grad(set_to_none=True)
    logger.info(f"Number of epochs = {cfg.train.num_epoch}")
    logger.info(f"Number of training steps per epoch = {iters_per_epoch}")
    logger.info(f"Number of total training steps = {cfg.train.lr_scheduler.total_steps}")
    logger.info(f"Gradient Accumulation steps = {accumulation}")
    logger.info(f"Number of optimizer steps = {cfg.train.lr_scheduler.total_steps // accumulation}")

    def train_one_epoch(epoch):
        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        iter_time, data_time = SmoothedValue(fmt="{avg:.4f}"), SmoothedValue(fmt="{avg:.4f}")
        logger.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch. Training dtype {cfg.train.model_dtype}")
        end = time.perf_counter()
        loss_details_dict = {}
        for it in range(iters_per_epoch):
            if it % batches_per_pass == 0:
                data_epoch = epoch * passes_per_epoch + it // batches_per_pass
                train_batch_sampler.set_epoch(data_epoch, base_seed=cfg.train.base_seed)
                train_dataset.set_epoch(data_epoch, base_seed=cfg.train.base_seed + accelerator.process_index)
                train_generator.manual_seed(int(cfg.train.base_seed) + data_epoch * accelerator.num_processes + accelerator.process_index)
                iterator = iter(train_loader)
            batch = next(iterator)
            data_time.update(time.perf_counter() - end)
            batch = send_to_device(batch, accelerator.device)
            batch_output, updated = train_step(model, batch, criterion, accelerator, optimizer, lr_scheduler,
                                               cfg.glob3r.max_pairs, cfg.train.clip_grad)
            batch_output = {key: accelerator.reduce(value, reduction="mean").item() for key, value in batch_output.items()}
            metric_logger.update(**batch_output)
            for key, value in batch_output.items():
                loss_details_dict[key] = loss_details_dict.get(key, 0.0) + value / accumulation
            learning_rates = lr_scheduler.get_last_lr()
            metric_logger.update(lr=max(learning_rates), min_lr=min(learning_rates))
            if accelerator.sync_gradients:
                if updated:
                    progress.global_step += 1
                    if progress.global_step % cfg.log.summary_interval == 0:
                        visualizer.log_scalars(dict(lr=max(learning_rates), min_lr=min(learning_rates), **loss_details_dict),
                                               progress.global_step, "train")
                    visualizer.log_images(accelerator, model, batch, criterion, progress.global_step, "train")
                loss_details_dict = {}
            iter_time.update(time.perf_counter() - end)
            if it % cfg.train.print_freq == 0 or it == iters_per_epoch - 1:
                eta = datetime.timedelta(seconds=int(iter_time.global_avg * (iters_per_epoch - it - 1)))
                memory = (f"  max mem: {torch.cuda.max_memory_allocated(accelerator.device) / 1024 ** 2:.0f}"
                          if accelerator.device.type == "cuda" else "")
                metric_logger.logger.info(f"Epoch: [{epoch}]  [{it}/{iters_per_epoch}]  eta: {eta}  "
                                          f"{metric_logger}  time: {iter_time}  data: {data_time}{memory}")
            end = time.perf_counter()
        logger.info(f"Averaged stats: {metric_logger}")
        return {key: meter.global_avg for key, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def validate(epoch):
        logger.info(f"Start validation for epoch {epoch}")
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        test_dataset.set_epoch(0, base_seed=cfg.train.base_seed)
        test_batch_sampler.set_epoch(0, base_seed=cfg.train.base_seed)
        test_loader.generator.manual_seed(cfg.train.base_seed)
        iterator = iter(test_loader)
        for it in range(iters_per_test):
            batch = send_to_device(next(iterator), accelerator.device)
            with accelerator.autocast():
                images = torch.stack([view["img"] for view in batch], 1)
                targets = sample_adjacent_pairs(images.shape[1], cfg.glob3r.max_pairs,
                                                generator=torch.Generator().manual_seed(cfg.train.seed + it))
                prediction = model(images, target_indices=targets)
                loss, details = criterion(prediction, batch)
            batch_output = {key: accelerator.reduce(value, reduction="mean").item()
                            for key, value in {"loss": loss, **details}.items()}
            metric_logger.update(**batch_output)
            if it == 0:
                visualizer.log_images(accelerator, model, batch, criterion, progress.global_step, "test")
            if it % cfg.test.print_freq == 0 or it == iters_per_test - 1:
                metric_logger.logger.info(f"Validation Epoch: [{epoch}]  [{it}/{iters_per_test}]  {metric_logger}")
        logger.info(f"Validation results: {metric_logger}")
        result = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
        visualizer.log_scalars(result, progress.global_step, "val")
        return result

    start_time = time.perf_counter()
    with MatchTensorBoard(output_dir, cfg.glob3r.visualization, enabled=cfg.log.use_tensorboard,
                          is_main_process=accelerator.is_main_process,
                          resume_step=progress.global_step if cfg.train.resume else None) as visualizer:
        for epoch in range(progress.epoch, cfg.train.num_epoch):
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(accelerator.device)
            train_one_epoch(epoch)
            val_stats = validate(epoch)
            progress.epoch = epoch + 1
            if val_stats["loss"] < progress.best_val_metric:
                progress.best_val_metric = val_stats["loss"]
                accelerator.save_state(str(ckpt_dir / "best_model"))
                logger.info(f"Saved best model at epoch {epoch} with val_metric: {progress.best_val_metric:.4f}")
            if (epoch + 1) % cfg.log.ckpt_interval == 0 or epoch + 1 == cfg.train.num_epoch:
                accelerator.save_state(str(ckpt_dir / f"checkpoint_{epoch}"))
                logger.info(f"Saved state for global step {progress.global_step}")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).match_head.state_dict(), output_dir / "match_head.pt")
        logger.info(f"Training time {datetime.timedelta(seconds=int(time.perf_counter() - start_time))}")
    accelerator.end_training()
    return progress


def main():
    import hydra
    @hydra.main(version_base="1.2", config_path=".", config_name="coarse")
    def launch(cfg):
        run_training(cfg)
    launch()


if __name__ == "__main__":
    main()
