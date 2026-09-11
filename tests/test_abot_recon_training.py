import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn

from pi3.models.abot_recon.data import prepare_abot_batch
from pi3.models.abot_recon.ema import ABotReconEMA
from pi3.models.abot_recon.loss import ABotReconLoss
from pi3.models.abot_recon.model import (
    ABotRecon,
    _checkpoint_state,
    _pi3_compatible_state,
)
from pi3.models.abot_recon.pose_head import AdjacentPoseHead
from pi3.models.abot_recon.rope3d import RoPE3D
from pi3.models.layers.attention import FlashAttentionRope


class ABotReconTrainingTest(unittest.TestCase):
    def test_ema_updates_and_round_trips(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema = ABotReconEMA(model, decay=0.5)

        with torch.no_grad():
            model.weight.fill_(4.0)
        ema.update(model)
        self.assertTrue(torch.equal(ema.module.weight, torch.full((1, 2), 3.0)))
        self.assertEqual(int(ema.averaged.n_averaged.item()), 2)

        restored = ABotReconEMA(model, decay=0.9)
        restored.load_state_dict(ema.state_dict())
        self.assertEqual(restored.decay, 0.5)
        self.assertEqual(int(restored.averaged.n_averaged.item()), 2)
        self.assertTrue(torch.equal(restored.module.weight, ema.module.weight))

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "custom_checkpoint_0.pkl"
            torch.save(ema.state_dict(), checkpoint)
            loaded = _checkpoint_state(checkpoint)
        self.assertTrue(torch.equal(loaded["weight"], ema.module.weight))

    def test_checkpoint_state_removes_distributed_prefix(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            torch.save(
                {"model": {"module.encoder.weight": torch.ones(1)}},
                checkpoint,
            )
            loaded = _checkpoint_state(checkpoint)
        self.assertEqual(list(loaded), ["encoder.weight"])

    def test_pi3_preload_keeps_only_compatible_base_weights(self):
        checkpoint = {
            "encoder.weight": torch.ones(2, 2),
            "point_head.weight": torch.ones(3, 3),
            "camera_head.weight": torch.ones(2, 2),
        }
        model_state = {
            "encoder.weight": torch.zeros(2, 2),
            "point_head.weight": torch.zeros(4, 4),
            "camera_head.weight": torch.zeros(2, 2),
        }
        compatible = _pi3_compatible_state(checkpoint, model_state)
        self.assertEqual(list(compatible), ["encoder.weight"])

    def test_batch_adapter_orders_every_sample_and_field(self):
        ids = ([2, 0], [0, 2], [1, 1])
        views = []
        for frame_ids in ids:
            frame = torch.tensor(frame_ids)
            value = frame.float().view(2, 1, 1, 1)
            views.append({
                "frame_id": frame,
                "img": value.expand(2, 3, 2, 2).clone(),
                "pts3d": value.expand(2, 1, 1, 3).clone(),
                "valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
                "camera_pose": torch.eye(4).expand(2, 4, 4).clone(),
            })
        batch = prepare_abot_batch(views)
        expected = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.float32)
        self.assertTrue(torch.equal(batch["imgs"][:, :, 0, 0, 0], expected))
        self.assertTrue(torch.equal(batch["world_points"][:, :, 0, 0, 0], expected))

    def test_perfect_prediction_has_zero_geometric_loss(self):
        b, n, h, w = 1, 3, 3, 3
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        plane = torch.stack((xx, yy, torch.ones_like(xx)), -1).float()
        batch = {
            "imgs": torch.zeros(b, n, 3, h, w),
            "world_points": plane[None, None].expand(b, n, h, w, 3).clone(),
            "valid_masks": torch.ones(b, n, h, w, dtype=torch.bool),
            "camera_poses": torch.eye(4).expand(b, n, 4, 4).clone(),
        }
        criterion = ABotReconLoss(confidence_weight=0.0)
        target = criterion.prepare_targets(batch)
        prediction = {
            "local_points": target["local_points"].clone(),
            "camera_poses": target["camera_poses"].clone(),
            "rotation_residual": torch.zeros(b, n - 1, 3),
            "conf": None,
        }
        loss, details = criterion(prediction, batch)
        self.assertLess(loss.item(), 1e-6)
        self.assertLess(details["pose_loss"].item(), 1e-6)

    def test_adjacent_pose_head_contract(self):
        head = AdjacentPoseHead(
            dim=8, hidden_dim=8, pair_hidden_dim=8, num_pose_tokens=2,
            rot_correction_kernel=3, enable_rotation_refiner=False)
        poses, state = head(torch.randn(2, 4, 5, 8))
        self.assertEqual(tuple(poses.shape), (2, 4, 4, 4))
        self.assertEqual(tuple(state["adjacent_poses"].shape), (2, 3, 4, 4))
        self.assertEqual(tuple(state["rotation_residual"].shape), (2, 3, 3))
        self.assertTrue(torch.allclose(poses[:, 0], torch.eye(4).expand(2, 4, 4)))

    def test_adjacent_pose_head_checkpoint_backward(self):
        head = AdjacentPoseHead(
            dim=8, hidden_dim=8, pair_hidden_dim=8, num_pose_tokens=2,
            rot_correction_kernel=3, enable_rotation_refiner=True,
            use_checkpoint=True)
        features = torch.randn(1, 4, 5, 8, requires_grad=True)
        poses, state = head(features)
        loss = poses.square().mean() + state["rotation_residual"].square().mean()
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_window_attention_is_causal(self):
        torch.manual_seed(7)
        attention = FlashAttentionRope(dim=12, num_heads=2, qkv_bias=True, rope=None)
        attention.gate_proj = torch.nn.Linear(12, 12, bias=False)
        holder = SimpleNamespace(
            local_window_frames=2, training=False,
            global_pos_encoding="rope3d", patch_start_idx=1,
            rope3d=RoPE3D(head_dim=6, max_seq_len=16, fhw_dim=(2, 2, 2)))
        x = torch.randn(1, 4, 3, 12)
        pos = torch.zeros(1, 4, 3, 2, dtype=torch.long)
        first = ABotRecon._window_attention(
            holder, attention, x, pos, spatial_shape=(1, 2))
        changed = x.clone()
        changed[:, 3] += 100.0
        second = ABotRecon._window_attention(
            holder, attention, changed, pos, spatial_shape=(1, 2))
        self.assertTrue(torch.allclose(first[:, :3], second[:, :3], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
