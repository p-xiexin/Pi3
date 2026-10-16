import json
import logging.config
from contextlib import chdir, redirect_stdout, ExitStack, nullcontext
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from pi3.models.abot_recon.model import ABotRecon
from pi3.models.abot_recon.rope3d import RoPE3D
from pi3.models.abot_recon.match_head.head import StreamingMatchHead, decode_feature_step, stream_rope_grid
from pi3.models.abot_recon.match_head.train import (
    ABotReconMatch, sample_adjacent_pairs, build_optimizer, run_training, train_step,
)
from pi3.models.abot_recon.match_head.loss import AdjacentMatchingLoss, mean_over_pairs
from pi3.models.dinov2.models.vision_transformer import DinoVisionTransformer
from pi3.models.layers.block import BlockRope
from pi3.models.layers.attention import FlashAttentionRope
from pi3.models.layers.pos_embed import PositionGetter, RoPE2D
from pi3.models.glob3r.model import Glob3RMatchingHead


def tiny_backbone(window=3, pos_encoding="rope3d"):
    # Exercise the actual DINO, gated decoder blocks, and KV implementation at small width.
    model = ABotRecon.__new__(ABotRecon)
    nn.Module.__init__(model)
    model.encoder = DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=32,
                                          depth=4, num_heads=4, block_chunks=0)
    model.patch_size, model.patch_start_idx, model.dec_embed_dim = 14, 5, 32
    model.register_token = nn.Parameter(torch.randn(1, 1, 5, 32) * 0.01)
    model.decoder = nn.ModuleList([BlockRope(32, 4, qkv_bias=True, qk_norm=True,
                                            attn_class=FlashAttentionRope, rope=RoPE2D(freq=100)) for _ in range(4)])
    for block in model.decoder:
        block.attn.gate_proj = nn.Linear(32, 32, bias=False)
    model.position_getter = PositionGetter()
    model.local_window_frames = window
    model.global_pos_encoding = pos_encoding
    model.rope3d = RoPE3D(head_dim=8, max_seq_len=8, fhw_dim=(4, 2, 2))
    model.num_dec_blk_not_to_checkpoint = 4
    model.register_buffer("image_mean", torch.zeros(1, 3, 1, 1))
    model.register_buffer("image_std", torch.ones(1, 3, 1, 1))
    # Checkpoint markers have the same names as the release.
    model.camera_head = nn.Module()
    model.camera_head.delta_t_head = nn.Linear(32, 3)
    model.camera_head.delta_q_head = nn.Linear(32, 4)
    return model.eval()


def tiny_model():
    return ABotReconMatch(tiny_backbone(), encoder_layers=(0, 1, 2, 3), match_dim=32)


def views(images, translate=0.0):
    b, n, _, h, w = images.shape
    result = []
    for frame in range(n):
        pose = torch.eye(4).repeat(b, 1, 1)
        pose[:, 0, 3] = frame * translate
        result.append({"img": images[:, frame], "depthmap": torch.ones(b, h, w),
                       "camera_intrinsics": torch.eye(3).repeat(b, 1, 1), "camera_pose": pose})
    return result


class SyntheticClips(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def set_epoch(self, epoch, base_seed=None):
        self.epoch = epoch

    def __getitem__(self, index):
        sample, _, frames = index
        images = torch.rand(1, frames, 3, 28, 28, generator=torch.Generator().manual_seed(100 + sample))
        return [{key: value[0] for key, value in view.items()} for view in views(images)]


class StreamingMatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)

    def test_decoder_step_matches_clip_beyond_window_and_rope_table(self):
        for mode in ("rope3d", "pi3_2d"):
            for window in (1, 3):
                model = tiny_backbone(window, mode)
                encoded = torch.randn(2, 11, 4, 32)
                with torch.no_grad():
                    model.rope3d = RoPE3D(head_dim=8, max_seq_len=64, fhw_dim=(4, 2, 2))
                    full, positions = model.decode(encoded.flatten(0, 1), 11, 28, 28)
                    model.rope3d = RoPE3D(head_dim=8, max_seq_len=8, fhw_dim=(4, 2, 2))
                    state, outputs = None, []
                    for frame in range(11):
                        output, pos, state = decode_feature_step(model, encoded[:, frame], 28, 28, state)
                        outputs.append(output)
                        self.assertEqual(state.frame_index, frame + 1)
                        for k, v in state.key_values:
                            self.assertEqual(k.shape[2], min(frame + 1, window - 1))
                            self.assertEqual(k.untyped_storage().nbytes(), k.numel() * k.element_size())
                            self.assertFalse(k.requires_grad)
                            self.assertEqual(k.shape, v.shape)
                    torch.testing.assert_close(torch.stack(outputs, 1).flatten(0, 1), full, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(pos, positions.reshape(2, 11, 9, 2)[:, -1])

    def test_head_reuses_exact_glob3r_coarse_and_both_frame_gradients(self):
        head = StreamingMatchHead(32, 64, 32)
        original = Glob3RMatchingHead(32, 64, 32, enable_refinement=False)
        original.load_state_dict(head.state_dict(), strict=True)
        geometry = torch.randn(1, 2, 4, 64, requires_grad=True)
        skips = [torch.randn(1, 2, 4, 32) for _ in range(4)]
        pred = head(geometry, skips, (28, 28))
        baseline = original(geometry, skips, torch.rand(1, 2, 3, 28, 28))
        torch.testing.assert_close(pred["coarse_warp"], baseline.coarse_warp)
        torch.testing.assert_close(pred["similarity"], baseline.similarity)
        pred["normalized_coarse_warp"].square().mean().backward()
        self.assertGreater(geometry.grad[:, 0].abs().sum(), 0)
        self.assertGreater(geometry.grad[:, 1].abs().sum(), 0)
        self.assertIsNone(head.refinement)
        self.assertIsNone(head.fine_features)

    def test_bfloat16_forward_backward_and_streaming(self):
        model = tiny_model().train()
        images = torch.rand(1, 3, 3, 28, 28)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = model(images)
            loss, _ = AdjacentMatchingLoss()(prediction, views(images))
        self.assertEqual(prediction["normalized_coarse_warp"].dtype, torch.float32)
        self.assertEqual(prediction["coarse_confidence_logits"].dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.match_head.parameters()))
        model.eval()
        state = None
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for frame in range(3):
                output, state = model.forward_step(images[:, frame], state)
        self.assertTrue(torch.isfinite(output["coarse_warp"]).all())

    def test_rope_positions_beyond_4096_and_state_shape_rejection(self):
        short = RoPE3D(head_dim=8, max_seq_len=4096, theta=5000, fhw_dim=(4, 2, 2))
        long = RoPE3D(head_dim=8, max_seq_len=5001, theta=5000, fhw_dim=(4, 2, 2))
        grid = stream_rope_grid(short, 5000, 2, 2, 5, "cpu")
        expected = long.grid(5001, 2, 2, 5, "cpu")[5000]
        torch.testing.assert_close(grid, expected, atol=1e-11, rtol=1e-11)
        model = tiny_model().eval()
        _, state = model.forward_step(torch.rand(1, 3, 28, 28))
        with self.assertRaisesRegex(ValueError, "reset"):
            model.forward_step(torch.rand(2, 3, 28, 28), state)

    def test_stream_matches_training_pairs_and_future_cannot_change_history(self):
        model = tiny_model().eval()
        images = torch.rand(1, 6, 3, 28, 28)
        with torch.no_grad():
            full = model(images, target_indices=[1, 3, 5])
            state, streamed = None, []
            for frame in range(6):
                pred, state = model.forward_step(images[:, frame], state)
                if frame == 0:
                    self.assertIsNone(pred)
                elif frame in (1, 3, 5):
                    streamed.append(pred["coarse_warp"])
            torch.testing.assert_close(torch.cat(streamed, 1), full["coarse_warp"], atol=2e-5, rtol=2e-5)
            self.assertEqual(full["reference_indices"].tolist(), [0, 2, 4])
            changed = images.clone()
            changed[:, 4:] = torch.rand_like(changed[:, 4:])
            prior = model(changed, target_indices=[1, 3])
            torch.testing.assert_close(prior["coarse_warp"], full["coarse_warp"][:, :2], atol=2e-5, rtol=2e-5)
            reset, state = model.forward_step(images[:, 0])
            self.assertIsNone(reset)
            reset, state = model.forward_step(images[:, 1], state)
            torch.testing.assert_close(reset["coarse_warp"], full["coarse_warp"][:, :1], atol=2e-5, rtol=2e-5)

    def test_gt_direction_occlusion_missing_depth_and_invalid_cameras(self):
        images = torch.rand(1, 3, 3, 28, 28)
        batch = views(images, translate=1.0)
        prediction = {"reference_indices": torch.tensor([1]), "target_indices": torch.tensor([2]), "image_size": (28, 28)}
        criterion = AdjacentMatchingLoss()
        gt = criterion.supervision(prediction, batch)
        self.assertEqual(gt.warp[0, 0, 4, 7, 0], 6)
        self.assertTrue(gt.mask[0, 0, 4, 0])  # finite projection outside image
        self.assertFalse(gt.confidence[0, 0, 4, 0])
        self.assertTrue(gt.confidence[0, 0, 4, 7])
        batch[2]["depthmap"][0, 4, 6] = 0
        gt = criterion.supervision(prediction, batch)
        self.assertFalse(gt.mask[0, 0, 4, 7])
        batch[2]["depthmap"][0, 4, 6] = 0.5
        gt = criterion.supervision(prediction, batch)
        self.assertTrue(gt.mask[0, 0, 4, 7])
        self.assertFalse(gt.confidence[0, 0, 4, 7])
        batch[2]["camera_pose"][:] = float("nan")
        gt = criterion.supervision(prediction, batch)
        self.assertFalse(gt.mask.any())
        self.assertTrue(torch.isfinite(gt.warp).all())

    def test_static_padding_and_fractional_depth_holes(self):
        images = torch.rand(1, 2, 3, 28, 28)
        batch = views(images, translate=0.5)
        pred = {"reference_indices": torch.tensor([0]), "target_indices": torch.tensor([1]), "image_size": (28, 28)}
        criterion = AdjacentMatchingLoss()
        batch[1]["depthmap"][0, 5, 5] = 0
        gt = criterion.supervision(pred, batch)
        self.assertFalse(gt.mask[0, 0, 5, 6])  # do not interpolate half-known GT
        batch[1]["depthmap"].fill_(1)
        batch[0]["static_mask"] = torch.ones(1, 28, 28, dtype=torch.bool)
        batch[0]["static_mask"][0, 5, 6] = False
        gt = criterion.supervision(pred, batch)
        self.assertFalse(gt.mask[0, 0, 5, 6])
        batch[0]["static_mask"].fill_(True)
        batch[1]["image_valid_mask"] = torch.zeros(1, 28, 28, dtype=torch.bool)
        gt = criterion.supervision(pred, batch)
        self.assertFalse(gt.mask[0, 0, 5, 6])

    def test_loss_backward_updates_only_matching_and_empty_labels_are_safe(self):
        model = tiny_model().train()
        images = torch.rand(1, 3, 3, 28, 28)
        before = {key: value.clone() for key, value in model.backbone.state_dict().items()}
        before_head = model.match_head.dpt_match.output[-1].weight.detach().clone()
        optimizer = build_optimizer(OmegaConf.load("pi3/models/abot_recon/match_head/coarse.yaml").train.optimizer, model)
        criterion = AdjacentMatchingLoss()
        prediction = model(images)
        loss, details = criterion(prediction, views(images))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.backbone.parameters()))
        for name, parameter in model.match_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before_head, model.match_head.dpt_match.output[-1].weight))
        for key, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        invalid = views(images)
        for view in invalid:
            view["depthmap"][:] = float("nan")
        optimizer.zero_grad()
        loss, _ = criterion(model(images), invalid)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.match_head.parameters()))

    def test_checkpoint_strictness_and_matching_roundtrip(self):
        backbone = tiny_backbone()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "backbone.pt"
            torch.save(backbone.state_dict(), checkpoint)
            with patch("pi3.models.abot_recon.match_head.train.ABotRecon", side_effect=lambda **kw: tiny_backbone()):
                loaded = ABotReconMatch.from_pretrained(checkpoint, encoder_layers=(0, 1, 2, 3), match_dim=32)
                for key, value in backbone.state_dict().items():
                    torch.testing.assert_close(value, loaded.backbone.state_dict()[key])
                missing = dict(backbone.state_dict())
                del missing["decoder.1.attn.gate_proj.weight"]
                torch.save(missing, checkpoint)
                with self.assertRaises(RuntimeError):
                    ABotReconMatch.from_pretrained(checkpoint, encoder_layers=(0, 1, 2, 3), match_dim=32)
            match = Path(directory) / "head.pt"
            torch.save(loaded.match_head.state_dict(), match)
            restored = tiny_model()
            restored.load_matching_checkpoint(match)
            for key, value in loaded.match_head.state_dict().items():
                torch.testing.assert_close(value, restored.match_head.state_dict()[key])

    def test_weighting_and_pair_sampling(self):
        value = torch.tensor([[[1., 1., 1.], [3., 3., 3.]], [[10., 10., 10.], [0., 0., 0.]]])
        mask = torch.tensor([[[1, 0, 0], [1, 1, 1]], [[1, 1, 1], [0, 0, 0]]], dtype=torch.bool)
        self.assertEqual(mean_over_pairs(value, mask), 6)
        for frames in (2, 3, 32):
            selected = sample_adjacent_pairs(frames)
            self.assertIn(frames - 1, selected)
            self.assertEqual(selected.numel(), min(frames - 1, 4))
            self.assertTrue((selected >= 1).all())

    def test_shared_scheduler_counts_successful_accumulation_groups(self):
        from utils.scheduler import build_scheduler

        cfg = OmegaConf.load("pi3/models/abot_recon/match_head/coarse.yaml")
        source = OmegaConf.load("configs/glob3r_coarse.yaml")
        for key in ("num_epoch", "iters_per_epoch", "gradient_accumulation_steps", "print_freq"):
            self.assertEqual(cfg.train[key], source.train[key])
        cfg.train.lr_scheduler.total_steps = 8
        cfg.train.lr_scheduler.warmup_steps = 1
        model = tiny_model().train()
        optimizer = build_optimizer(cfg.train.optimizer, model)
        lr_scheduler = build_scheduler(cfg.train.lr_scheduler, optimizer)
        accelerator = SimpleNamespace(accumulate=lambda _: nullcontext(), autocast=nullcontext,
                                      backward=lambda loss: loss.backward(), sync_gradients=True,
                                      clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
                                      optimizer_step_was_skipped=True, gradient_accumulation_steps=2)
        batch = views(torch.rand(1, 3, 3, 28, 28))
        with patch.object(optimizer, "step"):
            _, updated = train_step(model, batch, AdjacentMatchingLoss(), accelerator, optimizer, lr_scheduler)
        self.assertFalse(updated)
        self.assertEqual(lr_scheduler.last_epoch, 0)
        accelerator.optimizer_step_was_skipped = False
        _, updated = train_step(model, batch, AdjacentMatchingLoss(), accelerator, optimizer, lr_scheduler)
        self.assertTrue(updated)
        self.assertEqual(lr_scheduler.last_epoch, 2)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.95))

    def test_training_entrypoint_accumulation_validation_checkpoint_and_resume(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from utils.misc import get_logger
        with TemporaryDirectory() as directory, ExitStack() as stack:
            with initialize_config_dir(config_dir=str(Path("pi3/models/abot_recon/match_head").resolve()), version_base="1.2"):
                cfg = compose(config_name="coarse", return_hydra_config=True)
            stack.callback(setattr, HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)
            stack.callback(logging.config.dictConfig, {"version": 1, "disable_existing_loggers": False,
                                                       "root": {"level": "WARNING", "handlers": []}})
            HydraConfig.instance().set_config(cfg)
            shared_logger = stack.enter_context(patch("utils.misc.get_logger", wraps=get_logger))
            cfg.cpu = True
            cfg.log.merge_with(dict(output_dir=directory, ckpt_interval=1, summary_interval=1))
            cfg.train.merge_with(dict(model_dtype="no", num_workers=0, num_epoch=2, iters_per_epoch=2,
                                      gradient_accumulation_steps=2, image_num_range=[3, 3], clip_lengths=[3],
                                      clip_probabilities=[1.0], print_freq=1))
            cfg.train.lr_scheduler.warmup_steps = 0
            cfg.test.merge_with(dict(image_num_range=[3, 3], num_workers=0, iters_per_test=1))
            cfg.glob3r.visualization.interval_steps = 2
            cfg.glob3r.visualization.cell_width = 32
            model = tiny_model()
            console = StringIO()
            with redirect_stdout(console):
                progress = run_training(cfg, model, SyntheticClips(), SyntheticClips())
            self.assertEqual(progress.global_step, 2)
            self.assertEqual(progress.epoch, 2)
            self.assertEqual(cfg.train.lr_scheduler.total_steps, 4)
            checkpoint = Path(directory) / "ckpts" / "checkpoint_1"
            self.assertTrue((checkpoint / "match_head.pt").is_file())
            self.assertFalse((checkpoint / "pytorch_model.bin").exists())
            self.assertTrue((Path(directory) / "ckpts" / "best_model" / "match_head.pt").is_file())
            scheduler_state = torch.load(checkpoint / "custom_checkpoint_0.pkl", weights_only=False)
            self.assertEqual(scheduler_state["last_epoch"], 4)
            logfile = Path(directory) / "train.log"
            text = logfile.read_text(encoding="utf-8")
            training_lines = [line for line in text.splitlines() if "[rank: 0] Epoch:" in line]
            self.assertEqual(len(training_lines), 4)
            self.assertIn("Epoch: [0]  [0/2]", training_lines[0])
            self.assertIn("Epoch: [0]  [1/2]", training_lines[1])
            self.assertIn("Epoch: [1]  [1/2]", training_lines[3])
            for line in training_lines:
                for field in ("[utils.dist][INFO]", "eta:", "lr:", "loss:", "loss_nll:",
                              "loss_match_confidence:", "time:", "data:"):
                    self.assertIn(field, line)
                self.assertEqual(console.getvalue().count(line), 1)
            self.assertEqual(text.count("[rank: 0] Validation Epoch:"), 2)
            self.assertIn("Saved state for global step 2", text)
            self.assertNotIn("Update:", text)
            self.assertFalse((Path(directory) / "metrics.jsonl").exists())
            events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            self.assertEqual([event.step for event in events.Scalars("train/loss")], [1, 2])
            self.assertEqual([event.step for event in events.Scalars("val/epe")], [1, 2])
            self.assertEqual([event.step for event in events.Images("train/matching_overview")], [2])
            self.assertEqual([event.step for event in events.Images("val/matching_overview")], [1, 2])
            self.assertEqual(set(events.Tags()["images"]), {"train/matching_overview", "val/matching_overview"})
            cfg.train.resume = str(Path(directory) / "ckpts" / "checkpoint_0")
            torch.manual_seed(9)
            restored = tiny_model()
            resumed = run_training(cfg, restored, SyntheticClips(), SyntheticClips())
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(resumed.epoch, 2)
            resumed_text = logfile.read_text(encoding="utf-8")
            self.assertTrue(resumed_text.startswith(text))
            self.assertEqual(resumed_text[len(text):].count("[rank: 0] Epoch:"), 2)
            self.assertIn("Resuming from checkpoint", resumed_text[len(text):])
            self.assertEqual(shared_logger.call_count, 2)
            for key, value in model.match_head.state_dict().items():
                torch.testing.assert_close(value, restored.match_head.state_dict()[key], atol=1e-7, rtol=1e-6)
            resumed_events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            self.assertEqual([event.step for event in resumed_events.Scalars("train/loss")], [1, 2])
            self.assertEqual([event.step for event in resumed_events.Images("val/matching_overview")], [1, 2])

    def test_yaml_reuses_abot_scannet_and_trains_with_disk_images(self):
        import hydra
        from datasets.scannet_dataset import ScannetDataset
        from datasets.base.batched_sampler import DynamicBatchSampler

        with initialize_config_dir(config_dir=str(Path("pi3/models/abot_recon/match_head").resolve()), version_base="1.2"):
            cfg = compose(config_name="coarse", return_hydra_config=True)
        source = OmegaConf.load("pi3/models/abot_recon/dataset.yaml")
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.callback(setattr, HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)
            stack.callback(logging.config.dictConfig, {"version": 1, "disable_existing_loggers": False,
                                                       "root": {"level": "WARNING", "handlers": []}})
            HydraConfig.instance().set_config(cfg)
            root = Path(directory)
            scenes = ("scene0000_00", "scene0661_00")
            for scene in scenes:
                base = root / "scannet" / scene
                for folder in ("color", "depth", "pose", "intrinsic"):
                    (base / folder).mkdir(parents=True)
                intrinsic = np.eye(4, dtype=np.float32)
                intrinsic[0, 0] = intrinsic[1, 1] = 500
                intrinsic[0, 2], intrinsic[1, 2] = 320, 240
                np.savetxt(base / "intrinsic" / "intrinsic_depth.txt", intrinsic)
                for frame in range(8):
                    Image.fromarray(np.full((480, 640, 3), 80 + frame * 10, dtype=np.uint8)).save(base / "color" / f"{frame}.jpg")
                    Image.fromarray(np.full((480, 640), 2000, dtype=np.uint16)).save(base / "depth" / f"{frame}.png")
                    pose = np.eye(4, dtype=np.float32)
                    pose[0, 3] = frame * 0.01
                    if frame == 3:
                        pose[:] = np.nan
                    np.savetxt(base / "pose" / f"{frame}.txt", pose)
            (root / "data" / "dataset_cache").mkdir(parents=True)
            (root / "data" / "scannet_invalid_list.json").write_text(json.dumps({scene: [3] for scene in scenes}))
            cfg.cpu = True
            cfg.log.merge_with(dict(output_dir=str(root / "run"), ckpt_interval=1, summary_interval=1))
            cfg.train.merge_with(dict(model_dtype="no", num_workers=1, num_epoch=1, iters_per_epoch=1,
                                      gradient_accumulation_steps=1, image_num_range=[3, 3], clip_lengths=[3],
                                      clip_probabilities=[1.0], print_freq=1))
            cfg.train.lr_scheduler.warmup_steps = 0
            cfg.test.merge_with(dict(image_num_range=[3, 3], num_workers=1, iters_per_test=1))
            cfg.train_dataset.ScanNet.dataset.data_root = str(root / "scannet")
            cfg.test_dataset.ScanNet.dataset.data_root = str(root / "scannet")
            cfg.train_dataset.ScanNet.frame_num = 3
            cfg.train.resolution = [[28, 28]]
            with chdir(root):
                dataset = hydra.utils.instantiate(cfg.train_dataset.ScanNet)
                validation = hydra.utils.instantiate(cfg.test_dataset.ScanNet)
                self.assertIs(type(dataset.dataset), ScannetDataset)
                self.assertIs(type(validation.dataset), ScannetDataset)
                self.assertEqual(dataset.dataset.sequences, [scenes[0]])
                self.assertEqual(validation.dataset.sequences, [scenes[1]])
                self.assertEqual(dataset.dataset_label, source.train_dataset.ScanNet.dataset_label)
                self.assertEqual(dataset.dataset.max_distance, source.train_dataset.ScanNet.dataset.max_distance)
                self.assertEqual(validation.dataset.mode, "test")
                self.assertEqual(dataset.min_source_ratio, 1.0)
                self.assertEqual(dataset.repeat_probability, 0.0)
                batch = dataset[(0, 0, 3)]
                frame_ids = [int(view["source_frame_id"]) for view in batch]
                self.assertEqual(frame_ids, sorted(frame_ids))
                self.assertNotIn(3, frame_ids)
                self.assertTrue(all(view["img"].shape == (3, 28, 28) for view in batch))
                self.assertTrue(all(0 <= view["img"].min() <= view["img"].max() <= 1 for view in batch))
                self.assertTrue(all(np.allclose(view["depthmap"][view["valid_mask"]], 2.0) for view in batch))
                self.assertTrue(all(np.isfinite(view["camera_intrinsics"]).all() for view in batch))
                with patch("datasets.base.batched_sampler.DynamicBatchSampler", wraps=DynamicBatchSampler) as sampler:
                    progress = run_training(cfg, tiny_model())
                    self.assertEqual(sampler.call_count, 2)
                self.assertEqual(progress.global_step, 1)
                self.assertEqual(progress.epoch, 1)


if __name__ == "__main__":
    unittest.main()
