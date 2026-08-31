import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch
import torch.nn as nn

import sfm.loop as loop_module
from sfm.loop import (
    LoopDetector,
    loop_windows_from_descriptors,
    sliding_keyframe_pairs,
)
from sfm.loop_visualization import save_loop_match_images
from sfm.match_visualization import save_match_images
from sfm.factor_graph import FactorGraph
from sfm.tracker import FrameStore, WindowTracker
from sfm.tracks import VGGSfMTracks


class _DescriptorModel(nn.Module):
    def __init__(self, descriptors):
        super().__init__()
        self.register_buffer("descriptors", descriptors)
        self.offset = 0
        self.shapes = []
        self.inputs = []

    def forward(self, images):
        self.shapes.append(tuple(images.shape))
        self.inputs.append(images.detach().cpu())
        end = self.offset + images.shape[0]
        output = self.descriptors[self.offset:end]
        self.offset = end
        return output


class _Geometry(nn.Module):
    def infer_window(self, images):
        frames, height, width = images.shape[1], images.shape[-2], images.shape[-1]
        yy, xx = torch.meshgrid(
            torch.arange(height, device=images.device),
            torch.arange(width, device=images.device),
            indexing="ij",
        )
        points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()
        poses = torch.eye(4, device=images.device).repeat(frames, 1, 1)
        poses[:, 0, 3] = torch.arange(frames, device=images.device) * 0.1
        return {
            "local_points": points[None, None].repeat(1, frames, 1, 1, 1),
            "camera_poses": poses[None],
            "conf": torch.full(
                (1, frames, height, width, 1), 10.0, device=images.device
            ),
        }, None


class _Features:
    def extract(self, image, valid_mask=None):
        return image.new_tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


class _Tracks:
    def prepare_window(self, images, state):
        self.frames = images.shape[1]

    def track(self, reference, queries):
        return {
            "tracks": queries[None].repeat(self.frames, 1, 1),
            "confidence": queries.new_ones(self.frames, queries.shape[0]),
        }


class LoopDetectionTest(unittest.TestCase):
    def setUp(self):
        def relative_pose(source_points, target_points, source_K, target_K):
            return (
                torch.eye(3, device=source_points.device, dtype=source_points.dtype),
                source_points.new_tensor([1.0, 0.0, 0.0]),
                torch.ones(
                    source_points.shape[0],
                    device=source_points.device,
                    dtype=torch.bool,
                ),
            )

        self.geometry_patches = (
            patch("sfm.geometric_verification.MIN_PAIR_MATCHES", 2),
            patch("sfm.geometric_verification.MIN_PAIR_INLIERS", 2),
            patch(
                "sfm.geometric_verification._estimate_relative_pose",
                side_effect=relative_pose,
            ),
        )
        for active_patch in self.geometry_patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.geometry_patches):
            active_patch.stop()

    def test_match_and_loop_visualizations_are_separate_modules(self):
        self.assertEqual(save_match_images.__module__, "sfm.match_visualization")
        self.assertEqual(
            save_loop_match_images.__module__, "sfm.loop_visualization"
        )
        self.assertFalse(hasattr(loop_module, "save_match_images"))
        self.assertFalse(hasattr(loop_module, "save_loop_match_images"))

    def test_filtered_sliding_query_keeps_its_id_for_later_loop_match(self):
        class SelectiveTracks(_Tracks):
            def __init__(self):
                self.allow_last = False

            def track(self, reference, queries):
                output = super().track(reference, queries)
                if not self.allow_last:
                    output["confidence"][:, -1] = 0
                    output["confidence"][reference, -1] = 1
                return output

        dataset = SimpleNamespace(
            paths=[str(index) for index in range(4)],
            K=torch.tensor(
                [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]
            ),
            valid_mask=None,
            read=lambda frame_ids: torch.full(
                (len(frame_ids), 3, 4, 4), 0.5
            ),
        )
        frames = FrameStore(dataset)
        tracks = SelectiveTracks()
        tracker = WindowTracker(
            _Geometry(),
            tracks,
            _Features(),
            frames,
            dataset,
            {"device": "cpu", "keyframe_projection_threshold": 0.0},
        )
        sliding = tracker.track([0, 1, 2], minimum_edge_views=2)
        sliding["kind"] = "sliding"
        self.assertEqual(frames.track_ids[0].tolist(), [0, 1, 2])
        self.assertEqual(sliding["parts"][0]["keys"].tolist(), [0, 1])

        tracks.allow_last = True
        loop = tracker.track([0, 3], references=[0], minimum_edge_views=2)
        loop["kind"] = "loop"
        self.assertEqual(loop["parts"][0]["keys"].tolist(), [0, 1, 2])
        self.assertEqual(loop["parts"][0]["track_ids"].tolist(), [0, 1, 2])

        graph = FactorGraph(frames)
        graph.add_factors(sliding)
        self.assertEqual(graph.point_count, 3)
        graph.add_factors(loop)
        loop_points = graph.observations[-1][2]
        self.assertIn(2, loop_points.tolist())
        self.assertEqual(graph.point_count, 3)

    def test_failed_pair_can_be_retried_in_a_later_window(self):
        class TargetSelectiveTracks(_Tracks):
            def __init__(self):
                self.block_second_frame = True

            def track(self, reference, queries):
                output = super().track(reference, queries)
                if self.block_second_frame and self.frames > 1:
                    output["confidence"][1] = 0
                return output

        dataset = SimpleNamespace(
            paths=[str(index) for index in range(3)],
            K=torch.tensor(
                [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]
            ),
            valid_mask=None,
            read=lambda frame_ids: torch.full(
                (len(frame_ids), 3, 4, 4), 0.5
            ),
        )
        frames = FrameStore(dataset)
        tracks = TargetSelectiveTracks()
        tracker = WindowTracker(
            _Geometry(),
            tracks,
            _Features(),
            frames,
            dataset,
            {"device": "cpu", "keyframe_projection_threshold": 0.0},
        )
        queries = _Features().extract(dataset.read([0])[0])
        keys = torch.arange(queries.shape[0])
        anchors = torch.cat((queries, torch.ones_like(queries[:, :1])), dim=-1)
        frames.add_keyframe(
            0,
            dataset.read([0])[0],
            torch.ones(4, 4, 3),
            torch.ones(4, 4),
            keys,
            queries,
            anchors,
            torch.ones(queries.shape[0]),
        )

        initial = tracker.track(
            [0, 1, 2], references=[0], minimum_edge_views=2
        )
        self.assertTrue(initial["parts"])
        self.assertNotIn((0, 1), tracker.processed_pairs)
        self.assertIn((0, 2), tracker.processed_pairs)

        tracks.block_second_frame = False
        loop = tracker.track([0, 1], references=[0], minimum_edge_views=2)
        self.assertTrue(loop["parts"])
        self.assertIn((0, 1), tracker.processed_pairs)

    def test_loop_reuses_sliding_track_ids_and_rejects_unknown_queries(self):
        dataset = SimpleNamespace(
            paths=[], K=torch.eye(3), valid_mask=None
        )
        frames = FrameStore(dataset)
        keys = torch.tensor([4, 7])
        queries = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        anchors = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        weights = torch.ones(2)
        track_ids = frames.add_keyframe(
            3,
            torch.zeros(3, 2, 2),
            torch.ones(2, 2, 3),
            torch.ones(2, 2),
            keys,
            queries,
            anchors,
            weights,
        )
        repeated_ids = frames.add_keyframe(
            3,
            torch.zeros(3, 2, 2),
            torch.ones(2, 2, 3),
            torch.ones(2, 2),
            keys,
            queries,
            anchors,
            weights,
        )
        self.assertEqual(track_ids.tolist(), [0, 1])
        self.assertTrue(torch.equal(repeated_ids, track_ids))

        def packet(kind, feature_id, track_id, target):
            anchor_index = (
                keys.tolist().index(feature_id)
                if feature_id in keys.tolist()
                else 0
            )
            return {
                "kind": kind,
                "frame_ids": torch.tensor([3, target]),
                "keyframes": torch.tensor([3]),
                "poses": torch.eye(4).repeat(2, 1, 1),
                "K": torch.eye(3).repeat(2, 1, 1),
                "parts": [{
                    "reference": 3,
                    "keys": torch.tensor([feature_id]),
                    "track_ids": torch.tensor([track_id]),
                    "anchors": anchors[[anchor_index]],
                    "obs_frames": torch.tensor([3, target]),
                    "obs_points": torch.tensor([0, 0]),
                    "obs_uv": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
                    "obs_weights": torch.ones(2),
                }],
                "edges": [],
            }

        graph = FactorGraph(frames)
        graph.add_factors(packet("sliding", 4, 0, 4))
        self.assertEqual(graph.point_count, 2)
        self.assertEqual(graph.point_lookup[(3, 7)], 1)
        graph.add_factors(packet("loop", 7, 1, 9))
        self.assertEqual(graph.observations[-1][2].tolist(), [1, 1])
        self.assertEqual(set(graph.point_lookup), {(3, 4), (3, 7)})
        with self.assertRaisesRegex(RuntimeError, "unknown stable track"):
            graph.add_factors(packet("loop", 8, 99, 10))
        self.assertNotIn(10, graph.poses)
        invalid_observation = packet("loop", 4, 0, 11)
        invalid_observation["parts"][0]["obs_points"][1] = -1
        with self.assertRaisesRegex(RuntimeError, "invalid stable track index"):
            graph.add_factors(invalid_observation)
        self.assertNotIn(11, graph.poses)

    def test_cosine_threshold_builds_one_query_first_window_per_keyframe(self):
        descriptors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.0, 0.8]]
        )
        windows = loop_windows_from_descriptors([2, 5, 9], descriptors, 0.5)
        self.assertEqual(windows, [[2, 5, 9], [5, 2], [9, 2]])

    def test_similarity_must_be_strictly_greater_than_threshold(self):
        descriptors = torch.tensor([[1.0, 0.0], [0.5, 3.0 ** 0.5 / 2.0]])
        self.assertEqual(
            loop_windows_from_descriptors([0, 1], descriptors, 0.5), []
        )

    def test_sliding_packets_build_symmetric_keyframe_cooccurrence(self):
        packets = [
            {"kind": "sliding", "frame_ids": torch.tensor([0, 1, 2, 3])},
            {"kind": "sliding", "frame_ids": torch.tensor([2, 3, 4, 5])},
            {"kind": "loop", "frame_ids": torch.tensor([0, 6])},
        ]
        pairs = sliding_keyframe_pairs(packets, {0, 2, 4, 6})
        self.assertEqual(pairs, {(0, 2), (2, 0), (2, 4), (4, 2)})

    def test_local_cooccurrence_is_removed_before_loop_window_construction(self):
        descriptors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.0, 0.6]]
        )
        windows = loop_windows_from_descriptors(
            [0, 2, 9], descriptors, 0.5, {(0, 2), (2, 0)}
        )
        self.assertEqual(windows, [[0, 9], [2, 9], [9, 0, 2]])

    def test_detector_reads_uncropped_source_keyframes_in_batches(self):
        descriptors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.0, 0.8]]
        )
        model = _DescriptorModel(descriptors)
        with tempfile.TemporaryDirectory() as root:
            paths = []
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for index, color in enumerate(colors):
                path = Path(root) / f"{index}.png"
                Image.new("RGB", (32, 8), color).save(path)
                paths.append(path)
            frames = SimpleNamespace(
                keyframes={2, 0, 1},
                paths=paths,
                dense={
                    frame_id: (torch.full((3, 14, 28), 0.25), None, None)
                    for frame_id in (0, 1, 2)
                },
            )
            detector = LoopDetector(
                None, "cpu", (14, 28), batch_size=2, model=model
            )
            windows = detector.detect(frames, 0.5)
        self.assertEqual(windows, [[0, 1, 2], [1, 0], [2, 0]])
        self.assertEqual(model.shapes, [(2, 3, 14, 28), (1, 3, 14, 28)])
        normalized = torch.cat(model.inputs)
        mean = normalized.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = normalized.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        source_rgb = normalized * std + mean
        self.assertTrue(
            torch.allclose(
                source_rgb[:, :, 0, 0],
                torch.eye(3),
                atol=1.0 / 255.0,
            )
        )

    def test_loop_window_forces_first_reference_and_builds_two_view_edge(self):
        dataset = SimpleNamespace(
            paths=["0", "1", "2"],
            K=torch.tensor([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]),
            valid_mask=None,
            read=lambda frame_ids: torch.full((len(frame_ids), 3, 4, 4), 0.5),
        )
        frame_store = FrameStore(dataset)
        queries = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        frame_store.add_keyframe(
            2,
            torch.zeros(3, 4, 4),
            torch.ones(4, 4, 3),
            torch.ones(4, 4),
            torch.arange(3),
            queries,
            torch.ones(3, 3),
            torch.ones(3),
        )
        tracker = WindowTracker(
            _Geometry(), _Tracks(), _Features(), frame_store, dataset, {"device": "cpu"}
        )
        packet = tracker.track([2, 0], references=[2], minimum_edge_views=2)
        self.assertEqual(packet["keyframes"].tolist(), [2])
        self.assertEqual(len(packet["edges"]), 1)
        self.assertEqual(packet["edges"][0][:2], (2, 0))
        self.assertEqual(packet["edges"][0][3], 3)

    def test_loop_rejects_an_unregistered_reference_before_inference(self):
        dataset = SimpleNamespace(
            paths=["0", "1", "2"],
            K=torch.eye(3),
            valid_mask=None,
            read=lambda frame_ids: self.fail("loop inference should not start"),
        )
        frames = FrameStore(dataset)
        tracker = WindowTracker(
            _Geometry(), _Tracks(), _Features(), frames, dataset, {"device": "cpu"}
        )
        with self.assertRaisesRegex(RuntimeError, "no stable track ID registry"):
            tracker.track([2, 0], references=[2], minimum_edge_views=2)
        self.assertEqual(frames.next_track_id, 0)
        self.assertEqual(frames.track_ids, {})

    def test_cached_loop_packet_renders_loop_named_match_image(self):
        packet = {
            "kind": "loop",
            "frame_ids": torch.tensor([0, 1, 2]),
            "parts": [{
                "reference": 0,
                "track_ids": torch.tensor([10, 11]),
                "obs_frames": torch.tensor([0, 0, 2, 2]),
                "obs_points": torch.tensor([0, 1, 0, 1]),
                "obs_uv": torch.tensor(
                    [[0.0, 0.0], [3.0, 3.0], [1.0, 0.0], [2.0, 3.0]]
                ),
                "obs_weights": torch.tensor([1.0, 1.0, 0.9, 0.8]),
            }],
        }
        frames = SimpleNamespace(
            anchors={
                0: (
                    torch.tensor([0, 1, 2]),
                    torch.tensor([[0.0, 0.0], [3.0, 3.0], [1.0, 2.0]]),
                    torch.ones(3, 3),
                    torch.ones(3),
                )
            },
            dense={
                0: (torch.zeros(3, 4, 4), None, None),
                2: (torch.ones(3, 4, 4), None, None),
            }
        )
        with tempfile.TemporaryDirectory() as root:
            stale_loop = Path(root) / "loop_999999_999999.png"
            Image.new("RGB", (1, 1)).save(stale_loop)
            paths = save_loop_match_images(Path(root), [packet], frames)
            self.assertEqual([path.name for path in paths], ["loop_000000_000002.png"])
            self.assertFalse(stale_loop.exists())
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, (8, 32))

            sliding = dict(packet, kind="sliding")
            stale = Path(root) / "match_000000_000002.png"
            Image.new("RGB", (1, 1)).save(stale)
            paths = save_match_images(
                Path(root),
                [sliding],
                frames,
                SimpleNamespace(
                    read=lambda frame_ids: torch.full(
                        (len(frame_ids), 3, 4, 4), 0.5
                    )
                ),
                cell_width=4,
            )
            self.assertEqual(
                [path.name for path in paths], ["reference_0000.png"]
            )
            self.assertFalse(stale.exists())
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, (20, 12))

    def test_vggsfm_uses_official_378x672_square_padding_and_restores_coordinates(self):
        class Tracker(nn.Module):
            def process_images_to_fmaps(self, images):
                self.prepared_shape = tuple(images.shape)
                self.prepared_images = images.clone()
                return images[:, :, :1]

            def forward(self, images, queries, fmaps, fine_tracking):
                self.forward_shape = tuple(images.shape)
                self.queries = queries.clone()
                offsets = torch.arange(images.shape[1])[None, :, None, None]
                tracks = queries[:, None] + offsets
                confidence = torch.ones(*tracks.shape[:-1])
                return tracks, tracks, confidence, confidence

        tracker = Tracker()
        frontend = VGGSfMTracks(
            tracker, image_size=(378, 672), tracker_size=1024,
            mixed_precision="none",
        )
        images = torch.ones(1, 2, 3, 378, 672)
        queries = torch.tensor([[0.0, 0.0], [671.0, 377.0]])
        frontend.prepare_window(images, None)
        result = frontend.track(1, queries)

        self.assertEqual(tracker.prepared_shape, (1, 2, 3, 1024, 1024))
        self.assertEqual(tracker.forward_shape, (1, 2, 3, 1024, 1024))
        scale = 1024 / 672
        shift = torch.tensor([0.0, 147 * scale])
        self.assertTrue(torch.allclose(tracker.queries, queries[None] * scale + shift))
        self.assertTrue(torch.allclose(result["tracks"][1], queries))
        self.assertTrue(
            torch.allclose(result["tracks"][0], queries + 1.0 / scale)
        )
        self.assertEqual(float(tracker.prepared_images[0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(tracker.prepared_images[0, 0, 0, 512, 512]), 1.0)

        with self.assertRaisesRegex(ValueError, "expected image_size"):
            frontend.prepare_window(torch.zeros(1, 2, 3, 377, 672), None)

    def test_vggsfm_filters_visibility_and_score_independently(self):
        class Tracker(nn.Module):
            def process_images_to_fmaps(self, images):
                return images[:, :, :1]

            def forward(self, images, queries, fmaps, fine_tracking):
                tracks = queries[:, None].expand(-1, images.shape[1], -1, -1).clone()
                visibility = torch.tensor([[[1.0] * 4, [0.05, 0.0501, 0.9, 0.9]]])
                score = torch.tensor([[[1.0] * 4, [0.9, 0.9, 0.5, 0.5001]]])
                return tracks, tracks, visibility, score

        frontend = VGGSfMTracks(
            Tracker(), image_size=(4, 5), tracker_size=8, mixed_precision="none"
        )
        frontend.prepare_window(torch.zeros(1, 2, 3, 4, 5), None)
        result = frontend.track(
            0,
            torch.tensor(
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
            ),
        )

        self.assertEqual(frontend.visibility_threshold, 0.05)
        self.assertEqual(frontend.score_threshold, 0.5)
        self.assertTrue(torch.equal(result["confidence"][0], torch.ones(4)))
        self.assertTrue(
            torch.allclose(
                result["confidence"][1],
                torch.tensor([0.0, 0.0501 * 0.9, 0.0, 0.9 * 0.5001]),
            )
        )


if __name__ == "__main__":
    unittest.main()
