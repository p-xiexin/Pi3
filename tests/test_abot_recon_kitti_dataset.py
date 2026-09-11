import unittest
from unittest.mock import patch

import torch
from PIL import Image

from datasets.kitti_dataset import KITTIPi3XDataset
from datasets.blendmvs_dataset import BlendedMVSPi3XDataset
from datasets.scannet_dataset import ScannetDataset
from datasets.tartanair_dataset import TarTanAirDataset
from datasets.waymo_processed_dataset import WaymoPi3XDataset
from pi3.models.abot_recon.data_viz import render_abot_recon_dataset
from pi3.models.abot_recon.dataset import (
    BlendedMVSABotReconDataset,
    KITTIABotReconDataset,
    ScanNetABotReconDataset,
    TartanAirABotReconDataset,
    WaymoABotReconDataset,
)


class KITTIABotReconDatasetTest(unittest.TestCase):
    def test_dataset_filters_sequences_by_required_source_span(self):
        def initialize_parent(dataset, *args, **kwargs):
            del args, kwargs
            dataset.frame_step = 2
            dataset.records = [
                {"sequence_id": "short", "frames": list(range(4))},
                {"sequence_id": "valid", "frames": list(range(5))},
            ]
            dataset.sequences = ["short", "valid"]
            dataset.num_imgs = {"short": 4, "valid": 5}

        with patch.object(KITTIPi3XDataset, "__init__", initialize_parent):
            dataset = KITTIABotReconDataset(min_sequence_frames=3)

        self.assertEqual(dataset.sequences, ["valid"])
        self.assertEqual(dataset.num_imgs, {"valid": 5})
        self.assertEqual(len(dataset.records), 1)

    def test_dataset_rejects_empty_filtered_collection(self):
        def initialize_parent(dataset, *args, **kwargs):
            del args, kwargs
            dataset.frame_step = 1
            dataset.records = [
                {"sequence_id": "short", "frames": list(range(3))},
            ]
            dataset.sequences = ["short"]
            dataset.num_imgs = {"short": 3}

        with patch.object(KITTIPi3XDataset, "__init__", initialize_parent):
            with self.assertRaisesRegex(ValueError, "no sequence"):
                KITTIABotReconDataset(min_sequence_frames=4)

    def test_dataset_marks_parent_views_as_ordered(self):
        parent_views = [
            {"instance": "0000000012", "dataset": "KITTIPi3X"},
            {"instance": "0000000013", "dataset": "KITTIPi3X"},
        ]
        dataset = object.__new__(KITTIABotReconDataset)
        dataset.dataset_label = "KITTIABotRecon"
        with patch.object(KITTIPi3XDataset, "_get_views", return_value=parent_views):
            views = dataset._get_views(0, [8, 4], None, is_test=True)
        self.assertEqual([view["frame_id"] for view in views], [12, 13])
        self.assertEqual([view["temporal_index"] for view in views], [0, 1])
        self.assertTrue(all(view["abot_recon_ordered"] for view in views))
        self.assertTrue(all(view["dataset"] == "KITTIABotRecon" for view in views))

    def test_dataset_rejects_view_shuffle(self):
        with self.assertRaisesRegex(ValueError, "shuffle=false"):
            KITTIABotReconDataset(
                raw_root="unused",
                depth_root="unused",
                resolution=[[8, 4]],
                shuffle=True,
            )

    def test_dataset_rejects_global_random_sampling(self):
        with self.assertRaisesRegex(ValueError, "random_sample_thres=0.0"):
            KITTIABotReconDataset(
                raw_root="unused",
                depth_root="unused",
                resolution=[[8, 4]],
                random_sample_thres=0.1,
            )

    def test_tartanair_parent_views_are_sorted_by_source_frame(self):
        parent_views = [
            {"instance": "12"},
            {"instance": "3"},
            {"instance": "8"},
        ]
        dataset = object.__new__(TartanAirABotReconDataset)
        with patch.object(TarTanAirDataset, "_get_views", return_value=parent_views):
            views = dataset._get_views(0, [8, 4], None)
        self.assertEqual([view["frame_id"] for view in views], [3, 8, 12])
        self.assertEqual([view["temporal_index"] for view in views], [0, 1, 2])

    def test_scannet_parent_views_are_sorted_by_source_frame(self):
        parent_views = [{"instance": "40"}, {"instance": "10"}]
        dataset = object.__new__(ScanNetABotReconDataset)
        with patch.object(ScannetDataset, "_get_views", return_value=parent_views):
            views = dataset._get_views(0, [8, 4], None)
        self.assertEqual([view["frame_id"] for view in views], [10, 40])

    def test_waymo_preserves_parent_sequence_order(self):
        parent_views = [{"frame_id": 101}, {"frame_id": 109}]
        dataset = object.__new__(WaymoABotReconDataset)
        with patch.object(WaymoPi3XDataset, "_get_views", return_value=parent_views):
            views = dataset._get_views(0, [8, 4], None, is_test=True)
        self.assertEqual([view["frame_id"] for view in views], [101, 109])
        self.assertEqual([view["temporal_index"] for view in views], [0, 1])

    def test_blendedmvs_preserves_graph_path_as_pseudo_sequence(self):
        parent_views = [
            {"instance": "42"},
            {"instance": "7"},
            {"instance": "99"},
        ]
        dataset = object.__new__(BlendedMVSABotReconDataset)
        with patch.object(
            BlendedMVSPi3XDataset,
            "_get_views",
            return_value=parent_views,
        ):
            views = dataset._get_views(0, [8, 4], None, is_test=True)
        self.assertEqual([view["source_frame_id"] for view in views], [42, 7, 99])
        self.assertEqual([view["frame_id"] for view in views], [0, 1, 2])
        self.assertTrue(all(view["abot_recon_ordered"] for view in views))

    def test_abot_renderer_exposes_dense_contract(self):
        frames, height, width = 3, 4, 6
        yy, xx = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij")
        depth = 2.0 + xx.float() * 0.1 + yy.float() * 0.05
        local = torch.stack((xx.float() * depth, yy.float() * depth, depth), -1)
        views = []
        for frame in range(frames):
            pose = torch.eye(4)
            pose[0, 3] = frame * 0.2
            world = local + pose[:3, 3]
            views.append({
                "img": torch.full((3, height, width), frame / frames),
                "depthmap": depth.clone(),
                "pts3d": world,
                "valid_mask": torch.ones(height, width, dtype=torch.bool),
                "camera_pose": pose,
                "instance": f"{frame:010d}",
                "frame_id": frame,
            })
        overview, statistics = render_abot_recon_dataset(
            views, cell_width=30, max_frames=3)
        self.assertIsInstance(overview, Image.Image)
        self.assertEqual(overview.mode, "RGB")
        self.assertEqual(overview.width, 64 + frames * 30)
        self.assertEqual(overview.height, 3 * 20)
        self.assertIn("ordered frames: [0, 1, 2]", statistics)
        self.assertTrue(any("local-z-max-error" in line for line in statistics))


if __name__ == "__main__":
    unittest.main()
