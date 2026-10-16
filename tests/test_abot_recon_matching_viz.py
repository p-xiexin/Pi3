from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pi3.models.abot_recon.match_head.loss import AdjacentMatchingLoss
from pi3.models.abot_recon.match_head.viz import MatchTensorBoard, render_matching_overview
from pi3.models.glob3r.geometry import WarpSupervision
from tests.test_abot_recon_matching import tiny_model, views


class MatchingVisualizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_columns_have_distinct_references_and_target_warps(self):
        height, width = 8, 12
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        warp = torch.stack((xx, yy), 0).float()[None, None].expand(1, 2, -1, -1, -1).clone()
        images = torch.zeros(1, 3, 3, height, width)
        for frame in range(3):
            images[0, frame, frame] = 1
        prediction = dict(coarse_warp=warp, coarse_confidence=torch.ones(1, 2, 1, height, width),
                          image_size=(height, width), reference_indices=torch.tensor([0, 1]),
                          target_indices=torch.tensor([1, 2]))
        positive = torch.ones(1, 2, height, width, dtype=torch.bool)
        mask = positive.clone()
        positive[:, 1] = False
        mask[:, 1] = False
        supervision = WarpSupervision(warp.permute(0, 1, 3, 4, 2), positive, mask,
                                      torch.ones(1, 2, height, width))
        grid = np.array(render_matching_overview(images, prediction, supervision, cell_width=width))
        def pixel(row, pair):
            return grid[24 + row * height + height // 2, 92 + pair * width + width // 2]
        np.testing.assert_array_equal(pixel(0, 0), [255, 0, 0])
        np.testing.assert_array_equal(pixel(0, 1), [0, 255, 0])
        np.testing.assert_array_equal(pixel(2, 0), [0, 255, 0])
        np.testing.assert_array_equal(pixel(2, 1), [0, 0, 255])
        np.testing.assert_array_equal(pixel(3, 1), [0, 0, 0])
        np.testing.assert_array_equal(pixel(5, 1), [0, 0, 0])
        np.testing.assert_array_equal(pixel(6, 1), [0, 0, 0])
        np.testing.assert_array_equal(pixel(7, 1), [0, 0, 0])

    def test_matching_visualization_skips_geometry_and_restores_training_mode(self):
        model = tiny_model().train()
        batch = views(torch.rand(1, 5, 3, 28, 28), translate=0.1)
        accelerator = SimpleNamespace(unwrap_model=lambda value: value, autocast=nullcontext)
        criterion = AdjacentMatchingLoss()
        with TemporaryDirectory() as directory:
            config = dict(interval_steps=2, num_frames=3, cell_width=32)
            with MatchTensorBoard(directory, config) as visualizer:
                with patch.object(model, "forward", wraps=model.forward) as matching, \
                     patch.object(model.backbone, "forward", side_effect=AssertionError("geometry forward is unnecessary")) as geometry:
                    visualizer.log_images(accelerator, model, batch, criterion, 1, "train")
                    matching.assert_not_called()
                    geometry.assert_not_called()
                    rng = torch.get_rng_state().clone()
                    with patch.object(criterion, "supervision", wraps=criterion.supervision) as supervision:
                        visualizer.log_images(accelerator, model, batch, criterion, 2, "train")
                        self.assertEqual(supervision.call_count, 1)
                    self.assertEqual(matching.call_count, 1)
                    geometry.assert_not_called()
                    self.assertEqual(matching.call_args.args[0].shape[1], 3)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertTrue(model.training)
                    self.assertFalse(model.backbone.training)
                    self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                    with patch.object(criterion, "supervision", side_effect=RuntimeError("render failure")):
                        with self.assertRaisesRegex(RuntimeError, "render failure"):
                            visualizer.log_images(accelerator, model, batch, criterion, 4, "train")
                    self.assertTrue(model.training)
                    self.assertFalse(model.backbone.training)
            events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            self.assertEqual(set(events.Tags()["images"]), {"train/matching_overview"})
            self.assertEqual(events.Images("train/matching_overview")[0].step, 2)
            self.assertEqual(events.Images("train/matching_overview")[0].width, 92 + 2 * 32)

    def test_disabled_or_nonmain_logging_performs_no_inference_or_writes(self):
        with TemporaryDirectory() as directory:
            for options in (dict(enabled=False), dict(is_main_process=False)):
                with MatchTensorBoard(directory, {}, **options) as visualizer:
                    visualizer.log_images(None, None, None, None, 2, "train")
                    visualizer.log_scalars(dict(loss=1.0), global_step=2)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with MatchTensorBoard(directory, dict(enabled=False)) as visualizer:
                visualizer.log_images(None, None, None, None, 2, "train")
                visualizer.log_scalars(dict(loss=1.0), global_step=2)
            events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            self.assertEqual(events.Tags()["images"], [])
            self.assertEqual(events.Scalars("train/loss")[0].value, 1.0)


if __name__ == "__main__":
    unittest.main()
