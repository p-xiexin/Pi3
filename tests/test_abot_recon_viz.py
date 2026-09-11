import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from pi3.models.abot_recon.loss import ABotReconLoss
from pi3.models.abot_recon.viz import (
    ABotReconTensorBoardVisualizer,
    _poses_in_plot_frame,
    prepare_abot_diagnostics,
    render_reconstruction_overview,
)


def _example(frames=4):
    batch, height, width = 1, 4, 6
    yy, xx = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij")
    points = torch.stack((xx, yy, torch.ones_like(xx) * 4), -1).float()
    relative_poses = torch.eye(4).expand(batch, frames, 4, 4).clone()
    relative_poses[0, :, 0, 3] = torch.linspace(0.0, 0.3, frames)
    anchor = torch.eye(4)
    anchor[:3, :3] = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    anchor[:3, 3] = torch.tensor([10.0, -3.0, 2.0])
    poses = anchor[None, None] @ relative_poses
    world_points = torch.einsum("ij,hwj->hwi", anchor[:3, :3], points)
    world_points = world_points + anchor[:3, 3]
    sequence = {
        "imgs": torch.linspace(-1.0, 1.0, batch * frames * 3 * height * width).reshape(
            batch, frames, 3, height, width),
        "world_points": world_points[None, None].expand(
            batch, frames, height, width, 3).clone(),
        "valid_masks": torch.ones(batch, frames, height, width, dtype=torch.bool),
        "camera_poses": poses,
    }
    criterion = ABotReconLoss(confidence_weight=0.0, local_align_res=32)
    target = criterion.prepare_targets(sequence)
    prediction = {
        "local_points": target["local_points"].clone(),
        "camera_poses": target["camera_poses"].clone(),
        "rotation_residual": torch.zeros(batch, frames - 1, 3),
        "conf": torch.zeros(batch, frames, height, width, 1),
    }
    return prediction, sequence, criterion


class _Tracker:
    def __init__(self):
        self.calls = []

    def log_images(self, images, step):
        self.calls.append((images, step))


class ABotReconVisualizationTest(unittest.TestCase):
    def test_kitti_xz_plot_frame_is_right_handed_and_horizontal(self):
        poses = torch.eye(4).expand(2, 4, 4).clone()
        poses[1, 0, 3] = 2.0
        poses[1, 1, 3] = 3.0
        poses[1, 2, 3] = 5.0
        plotted = _poses_in_plot_frame(poses, "xz")
        self.assertTrue(torch.equal(
            plotted[1, :3, 3], torch.tensor([2.0, 5.0, -3.0])))
        self.assertAlmostEqual(torch.det(plotted[0, :3, :3]).item(), 1.0)
        camera_forward = plotted[0, :3, :3] @ torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(torch.equal(camera_forward, torch.tensor([0.0, 1.0, 0.0])))

    def test_long_sequence_render_uses_eight_aligned_frames(self):
        prediction, sequence, criterion = _example(frames=128)
        reconstruction = render_reconstruction_overview(
            prediction, sequence, criterion, num_frames=8, cell_width=32,
            pose_plot_mode="xz")
        self.assertEqual(reconstruction.width, 92 + 8 * 32)

    def test_renderers_return_rgb_images(self):
        prediction, sequence, criterion = _example()
        reconstruction = render_reconstruction_overview(
            prediction, sequence, criterion, num_frames=3, cell_width=40)
        self.assertIsInstance(reconstruction, Image.Image)
        self.assertEqual(reconstruction.mode, "RGB")
        self.assertEqual(reconstruction.width, 92 + 3 * 40)
        self.assertEqual(reconstruction.height, 24 + 7 * 27)
        diagnostics = prepare_abot_diagnostics(prediction, sequence, criterion)
        self.assertTrue(torch.allclose(
            diagnostics["world_predicted_poses"], sequence["camera_poses"],
            atol=1e-5, rtol=1e-5))
        prediction["conf"] = None
        stage_one = render_reconstruction_overview(
            prediction, sequence, criterion, num_frames=2, cell_width=32)
        self.assertEqual(stage_one.mode, "RGB")

    def test_optimizer_interval_and_validation_are_logged_once(self):
        prediction, sequence, criterion = _example()
        tracker = _Tracker()
        accelerator = SimpleNamespace(
            is_main_process=True, sync_gradients=True, trackers=[tracker])
        visualizer = ABotReconTensorBoardVisualizer(
            {
                "enabled": True,
                "interval_steps": 2,
                "num_samples": 1,
                "num_frames": 2,
                "cell_width": 32,
                "pose_camera_scale": 0.14,
            },
            gradient_accumulation_steps=1,
            initial_global_step=0,
            train_criterion=criterion,
            test_criterion=criterion,
        )
        output = [prediction, sequence]
        visualizer.log(accelerator, output, "train")
        self.assertEqual(len(tracker.calls), 0)
        visualizer.log(accelerator, output, "train")
        self.assertEqual(len(tracker.calls), 1)
        images, step = tracker.calls[0]
        self.assertEqual(step, 2)
        self.assertEqual(set(images), {"train/abot_reconstruction"})
        self.assertEqual(images["train/abot_reconstruction"].ndim, 4)

        visualizer.begin_validation(3)
        visualizer.log(accelerator, output, "test")
        visualizer.log(accelerator, output, "test")
        self.assertEqual(len(tracker.calls), 2)
        self.assertEqual(tracker.calls[1][1], 4)
        self.assertEqual(set(tracker.calls[1][0]), {"val/abot_reconstruction"})


if __name__ == "__main__":
    unittest.main()
