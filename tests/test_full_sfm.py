import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sfm.factor_graph import FactorGraph, select_local_fixed_ids
from sfm.features import _top_score_indices
from sfm.geometric_verification import triangulate_tracks, verify_tracks
from sfm.geometry import camera_centers
from sfm.keyframes import select_keyframes_eq4_window
from sfm.optimizer import _project, optimize_view
from sfm.reconstruction import reconstruct
from sfm.tracker import FrameStore, WindowTracker
from sfm.tracks import Glob3RTracks, VGGSfMTracks


def _load_export():
    fake_ply = ModuleType("plyfile")
    fake_ply.PlyData = object
    fake_ply.PlyElement = object
    with patch.dict(sys.modules, {"plyfile": fake_ply}):
        return importlib.import_module("sfm.export")


class _Dataset:
    def __init__(self, count=6, height=6, width=6):
        self.paths = [str(i) for i in range(count)]
        self.K = torch.tensor([[10.0, 0, width / 2], [0, 10.0, height / 2], [0, 0, 1.0]])
        self.valid_mask = None
        self.height, self.width = height, width

    def read(self, frame_ids):
        return torch.full((len(frame_ids), 3, self.height, self.width), 0.5)


class _Geometry:
    def __init__(self, height=6, width=6):
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        self.points = torch.stack(
            ((xx - width / 2) / 10, (yy - height / 2) / 10, torch.ones_like(xx)), -1
        ).float()

    def eval(self):
        return self

    def infer_window(self, images):
        frames = images.shape[1]
        poses = torch.eye(4).repeat(frames, 1, 1)
        poses[:, 0, 3] = torch.arange(frames) * 0.01
        geometry = {
            "local_points": self.points[None, None].repeat(1, frames, 1, 1, 1),
            "camera_poses": poses[None],
            "conf": torch.full((1, frames, *self.points.shape[:2], 1), 10.0),
        }
        return geometry, None


class _Features:
    def __init__(self, max_points):
        self.max_points = max_points

    def extract(self, image, valid_mask=None):
        yy, xx = torch.meshgrid(torch.arange(image.shape[-2]), torch.arange(image.shape[-1]), indexing="ij")
        points = torch.stack((xx, yy), -1).reshape(-1, 2).to(image)
        return points[:self.max_points]


class _Tracks:
    def prepare_window(self, images, state):
        self.frame_count = images.shape[1]

    def track(self, reference, query_points):
        offsets = -0.1 * (
            torch.arange(self.frame_count, device=query_points.device) - reference
        )
        tracks = query_points[None].repeat(self.frame_count, 1, 1)
        tracks[..., 0] += offsets[:, None]
        return {
            "tracks": tracks,
            "confidence": torch.ones(self.frame_count, query_points.shape[0]),
        }


def _synthetic_relative_pose(source_points, target_points, source_K, target_K):
    translation = source_points.new_zeros(3)
    translation[0] = 10.0 * (
        target_points[:, 0] - source_points[:, 0]
    ).median()
    return (
        torch.eye(3, device=source_points.device, dtype=source_points.dtype),
        translation,
        torch.ones(source_points.shape[0], device=source_points.device, dtype=torch.bool),
    )


class FullSfMTest(unittest.TestCase):
    def setUp(self):
        self.geometry_patches = (
            patch("sfm.geometric_verification.MIN_PAIR_MATCHES", 3),
            patch("sfm.geometric_verification.MIN_PAIR_INLIERS", 3),
            patch(
                "sfm.geometric_verification._estimate_relative_pose",
                side_effect=_synthetic_relative_pose,
            ),
        )
        for active_patch in self.geometry_patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.geometry_patches):
            active_patch.stop()

    def test_overlapping_window_fixes_only_one_existing_camera(self):
        self.assertEqual(select_local_fixed_ids(0, [0, 1, 2], []), [0])
        self.assertEqual(select_local_fixed_ids(1, [2, 3, 4], [2, 3]), [2])
        with self.assertRaisesRegex(RuntimeError, "no camera shared"):
            select_local_fixed_ids(1, [3, 4, 5], [])

    def test_pose_edges_keep_first_inference_and_append_new_pairs(self):
        frames = SimpleNamespace(track_ids={}, anchors={}, next_track_id=0)
        graph = FactorGraph(frames)
        stronger = torch.eye(4)
        stronger[0, 3] = 2.0
        base = {
            "frame_ids": torch.tensor([0, 1, 2]),
            "keyframes": torch.empty(0, dtype=torch.long),
            "poses": torch.eye(4).repeat(3, 1, 1),
            "K": torch.eye(3).repeat(3, 1, 1),
            "parts": [],
        }
        graph.add_factors(
            dict(base, edges=[(0, 1, torch.eye(4), 2.0)])
        )
        graph.add_factors(
            dict(
                base,
                edges=[
                    (1, 0, torch.linalg.inv(stronger), 5.0),
                    (1, 2, stronger, 3.0),
                ],
            )
        )

        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(graph.edges[0][:2], (0, 1))
        self.assertEqual(graph.edges[0][3], 2.0)
        self.assertTrue(torch.allclose(graph.edges[0][2], torch.eye(4)))
        self.assertEqual(graph.edges[1][:2], (1, 2))
        self.assertEqual(graph.edges[1][3], 3.0)
        self.assertTrue(torch.allclose(graph.edges[1][2], stronger))

    def test_metric_scaled_pi3_poses_initialize_the_sliding_pose_graph(self):
        frames = SimpleNamespace(track_ids={}, anchors={}, next_track_id=0)
        graph = FactorGraph(frames)
        pi3_T_WCs = torch.eye(4).repeat(3, 1, 1)
        pi3_T_WCs[:, 0, 3] = torch.tensor([0.0, 0.5, 1.0])
        angles = torch.tensor([0.0, 0.2, 0.4])
        pi3_T_WCs[:, 0, 0] = torch.cos(angles)
        pi3_T_WCs[:, 0, 1] = -torch.sin(angles)
        pi3_T_WCs[:, 1, 0] = torch.sin(angles)
        pi3_T_WCs[:, 1, 1] = torch.cos(angles)
        sliding = {
            "kind": "sliding",
            "frame_ids": torch.tensor([0, 1, 2]),
            "keyframes": torch.empty(0, dtype=torch.long),
            "poses": torch.eye(4).repeat(3, 1, 1),
            "pi3_T_WCs": pi3_T_WCs,
            "metric_scale": torch.tensor(2.0),
            "K": torch.eye(3).repeat(3, 1, 1),
            "parts": [],
            "edges": [
                (0, 1, torch.eye(4), 2.0),
                (1, 2, torch.eye(4), 2.0),
            ],
        }
        graph.add_factors(sliding)
        scaled_pi3_T_WCs = pi3_T_WCs.clone()
        scaled_pi3_T_WCs[:, :3, 3] *= 2.0
        loop_relative = (
            torch.linalg.inv(scaled_pi3_T_WCs[2]) @ scaled_pi3_T_WCs[0]
        )
        loop_relative[0, 3] = 100.0
        graph.add_factors(
            {
                **sliding,
                "kind": "loop",
                "pi3_T_WCs": torch.eye(4).repeat(3, 1, 1) * 20.0,
                "metric_scale": torch.tensor(50.0),
                "edges": [(0, 2, loop_relative, 100.0)],
            }
        )
        self.assertEqual(graph.first_sliding_root_frame_id, 0)
        self.assertEqual(graph.first_sliding_midpoint_frame_id, 1)

        with patch.object(graph, "_triangulate_global") as triangulate, patch(
            "sfm.factor_graph.average_rotations",
            side_effect=lambda poses, source, target, relative, weight: poses[:, :3, :3],
        ) as average:
            graph.initialize_global()

        poses = torch.stack([graph.poses[index] for index in range(3)])
        expected_centers = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        torch.testing.assert_close(camera_centers(poses), expected_centers)
        torch.testing.assert_close(
            poses[:, :3, :3],
            torch.linalg.inv(scaled_pi3_T_WCs)[:, :3, :3],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        self.assertEqual(graph.pi3_edge_keys, {(0, 1), (1, 2)})
        torch.testing.assert_close(graph.edges[-1][2], loop_relative)
        self.assertEqual(average.call_args.args[1].numel(), 3)
        triangulate.assert_called_once()
        torch.testing.assert_close(triangulate.call_args.args[1], poses)

    def test_keyframe_coverage_uses_target_and_projected_validity(self):
        dataset = _Dataset(count=2)
        frames = FrameStore(dataset)
        points = _Geometry().points[None].repeat(2, 1, 1, 1)
        poses = torch.eye(4).repeat(2, 1, 1)
        K = dataset.K.expand(2, -1, -1).clone()
        target_valid = torch.zeros(2, dataset.height, dataset.width, dtype=torch.bool)
        target_valid[0] = True
        target_valid[1, 0] = True
        self.assertEqual(
            select_keyframes_eq4_window(
                [0, 1], points, torch.ones_like(target_valid), poses, K, target_valid
            ).tolist(), [0]
        )

        projected_invalid = target_valid.clone()
        projected_invalid[0, 0] = False
        self.assertEqual(
            select_keyframes_eq4_window(
                [0, 1], points, torch.ones_like(target_valid), poses, K, projected_invalid
            ).tolist(), [0, 1]
        )

    def test_tracker_keyframes_follow_eq4_instead_of_match_coverage(self):
        class CoverageTracks(_Tracks):
            def __init__(self):
                self.references = []

            def track(self, reference, query_points):
                output = super().track(reference, query_points)
                self.references.append(reference)
                return output

        dataset = _Dataset(count=4)
        frames = FrameStore(dataset)
        tracks = CoverageTracks()
        tracker = WindowTracker(
            _Geometry(), tracks, _Features(8), frames, dataset, {"device": "cpu"}
        )
        frame_ids = [0, 1, 2, 3]
        packet = tracker.track(frame_ids)

        self.assertEqual(packet["keyframes"].tolist(), [0])
        self.assertEqual(tracks.references, [0])

    def test_shared_pi3_mask_removes_low_confidence_track_observations(self):
        class Geometry(_Geometry):
            def infer_window(self, images):
                geometry, state = super().infer_window(images)
                geometry["conf"][0, 1, 0, 1, 0] = -10
                return geometry, state

        dataset = _Dataset(count=4)
        frames = FrameStore(dataset)
        tracker = WindowTracker(
            Geometry(), _Tracks(), _Features(12), frames, dataset, {"device": "cpu"}
        )
        packet = tracker.track([0, 1, 2, 3])
        self.assertEqual(packet["keyframes"].tolist(), [0])
        part = next(part for part in packet["parts"] if part["reference"] == 0)
        target = part["obs_frames"] == 1
        low_pi3_confidence = torch.isclose(
            part["obs_uv"][:, 0], part["obs_uv"].new_tensor(0.9)
        ) & torch.isclose(part["obs_uv"][:, 1], part["obs_uv"].new_tensor(0.0))
        self.assertFalse(bool((target & low_pi3_confidence).any()))

    def test_overlapping_windows_build_one_graph(self):
        dataset = _Dataset()
        frames = FrameStore(dataset)
        tracker = WindowTracker(_Geometry(), _Tracks(), _Features(12), frames, dataset, {"device": "cpu"})
        graph = FactorGraph(frames)
        first = graph.add_factors(tracker.track([0, 1, 2, 3]))
        second = graph.add_factors(tracker.track([2, 3, 4, 5]))
        graph.initialize_global()
        view = graph.full_view()
        self.assertEqual(first, [])
        self.assertEqual(second, [2, 3])
        self.assertEqual(graph.frame_ids, set(range(6)))
        self.assertEqual(sorted(frames.keyframes), [0, 2])
        self.assertGreater(view["points"].shape[0], 0)
        self.assertTrue({0, 2}.issubset(set(view["frame_ids"].tolist())))

    def test_overlap_depth_scale_is_stored_only_on_new_dense_frames(self):
        class ScaledGeometry(_Geometry):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def infer_window(self, images):
                geometry, state = super().infer_window(images)
                raw_scale = 1.0 if self.calls == 0 else 0.5
                geometry["local_points"] *= raw_scale
                geometry["camera_poses"][..., :3, 3] *= raw_scale
                self.calls += 1
                return geometry, state

        dataset = _Dataset()
        frames = FrameStore(dataset)
        tracker = WindowTracker(
            ScaledGeometry(),
            _Tracks(),
            _Features(12),
            frames,
            dataset,
            {"device": "cpu"},
        )
        tracker.track([0, 1, 2, 3])
        packet = tracker.track([2, 3, 4, 5])

        self.assertNotIn("pi3_depth_scale", packet)
        self.assertEqual(frames.dense[2][3], 1.0)
        self.assertEqual(frames.dense[3][3], 1.0)
        self.assertEqual(frames.dense[4][3], 2.0)
        self.assertEqual(frames.dense[5][3], 2.0)
        self.assertTrue(packet["edges"])
        for source, target, relative, _ in packet["edges"]:
            expected = float(abs(target - source))
            self.assertAlmostEqual(float(relative[:3, 3].norm()), expected, places=5)

    def test_local_and_global_views_use_the_same_optimizer(self):
        dataset = _Dataset()
        frames = FrameStore(dataset)
        tracker = WindowTracker(_Geometry(), _Tracks(), _Features(8), frames, dataset, {"device": "cpu"})
        graph = FactorGraph(frames)
        graph.add_factors(tracker.track([0, 1, 2, 3]))
        graph.add_factors(tracker.track([2, 3, 4, 5]))
        graph.initialize_global()
        local = graph.local_view([0, 1, 2, 3])
        full = graph.full_view()
        self.assertEqual(full["scale_gauge_root_frame_id"], 0)
        self.assertEqual(full["scale_gauge_frame_id"], 2)

        def identity(view, fixed, **kwargs):
            observation_inliers = torch.ones_like(view["ii"], dtype=torch.bool)
            point_inliers = torch.ones_like(view["point_ids"], dtype=torch.bool)
            count = observation_inliers.sum()
            return {
                "poses": view["poses"],
                "points": view["points"],
                "loss": view["points"].new_tensor(0.0),
                "loss_per_pixel": view["points"].new_tensor(0.0),
                "valid_observations": count,
                "input_observations": count,
                "first_inlier_observations": count,
                "observation_inliers": observation_inliers,
                "point_inliers": point_inliers,
            }

        with patch("sfm.factor_graph.optimize_view_two_rounds", side_effect=identity) as shared:
            self.assertEqual(graph.optimize(local, [0], 2)["scope"], "local")
            self.assertEqual(graph.optimize(full, [0], 5)["scope"], "global")
        self.assertEqual(shared.call_count, 2)
        self.assertIsNone(shared.call_args_list[0].kwargs["scale_gauge_camera"])
        self.assertEqual(shared.call_args_list[1].kwargs["scale_gauge_camera"], 2)

    def test_essential_verification_masks_pair_outliers_before_factors(self):
        frames, points = 4, 6
        tracks = torch.zeros(frames, points, 2)
        tracks[..., 0] = torch.arange(points)[None] - 0.1 * torch.arange(frames)[:, None]
        tracks[..., 1] = 2.0
        mask = torch.ones(frames, points, dtype=torch.bool)
        weights = torch.ones(frames, points)
        K = torch.tensor(
            [[10.0, 0.0, 3.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]
        ).repeat(frames, 1, 1)

        def reject_last(source_points, target_points, source_K, target_K):
            rotation, translation, inliers = _synthetic_relative_pose(
                source_points, target_points, source_K, target_K
            )
            inliers[-1] = False
            return rotation, translation, inliers

        with patch(
            "sfm.geometric_verification._estimate_relative_pose",
            side_effect=reject_last,
        ):
            verified = verify_tracks(tracks, mask, weights, K)

        self.assertEqual(verified.source.numel(), 6)
        self.assertEqual(verified.keep.tolist(), [True, True, True, True, True, False])
        self.assertEqual(verified.mask.shape, (4, 5))

    def test_two_view_tracks_do_not_contribute_to_pose_estimation(self):
        frames, points = 4, 7
        tracks = torch.zeros(frames, points, 2)
        tracks[..., 0] = (
            torch.arange(points)[None]
            - 0.1 * torch.arange(frames)[:, None]
        )
        tracks[..., 1] = 2.0
        mask = torch.ones(frames, points, dtype=torch.bool)
        mask[2:, -1] = False
        weights = torch.ones(frames, points)
        K = torch.tensor(
            [[10.0, 0.0, 3.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]
        ).repeat(frames, 1, 1)
        observed_counts = []

        def record_count(source_points, target_points, source_K, target_K):
            observed_counts.append(source_points.shape[0])
            return _synthetic_relative_pose(
                source_points, target_points, source_K, target_K
            )

        with patch(
            "sfm.geometric_verification._estimate_relative_pose",
            side_effect=record_count,
        ):
            verified = verify_tracks(tracks, mask, weights, K)

        self.assertEqual(observed_counts, [6] * 6)
        self.assertEqual(verified.keep.tolist(), [True] * 6 + [False])

    def test_multiview_triangulation_recovers_points_and_parallax(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, -0.5, -1.0], dtype=dtype)
        points = torch.tensor(
            [[0.2, -0.1, 4.0], [-0.3, 0.2, 6.0]], dtype=dtype
        )
        ii = torch.arange(3).repeat_interleave(points.shape[0])
        jj = torch.arange(points.shape[0]).repeat(3)
        K = torch.tensor(
            [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).repeat(3, 1, 1)
        uv, _ = _project(poses, points, K, torch.zeros(3, 4, dtype=dtype), ii, jj)

        triangulated, observation_mask, point_mask, angles = triangulate_tracks(
            poses, K, ii, jj, uv, points.shape[0]
        )

        self.assertTrue(bool(point_mask.all()))
        self.assertTrue(bool(observation_mask.all()))
        self.assertTrue(torch.allclose(triangulated, points, atol=1.0e-8))
        self.assertTrue(bool((angles > 1.5).all()))

    def test_matrix_free_ba_reduces_reprojection_error(self):
        torch.manual_seed(2)
        dtype = torch.float64
        cameras, point_count = 3, 20
        truth = torch.eye(4, dtype=dtype).repeat(cameras, 1, 1)
        truth[:, 0, 3] = torch.tensor([0.0, -0.4, -0.8], dtype=dtype)
        points = torch.randn(point_count, 3, dtype=dtype) * torch.tensor([0.4, 0.3, 0.2], dtype=dtype)
        points += torch.tensor([0.0, 0.0, 4.0], dtype=dtype)
        ii = torch.arange(cameras).repeat_interleave(point_count)
        jj = torch.arange(point_count).repeat(cameras)
        K = torch.tensor([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=dtype).repeat(cameras, 1, 1)
        distortion = torch.zeros(cameras, 4, dtype=dtype)
        target, _ = _project(truth, points, K, distortion, ii, jj)
        initial_poses = truth.clone()
        initial_poses[1:, 0, 3] += torch.tensor([0.08, -0.07], dtype=dtype)
        initial_points = points + 0.05 * torch.randn_like(points)
        before = (_project(initial_poses, initial_points, K, distortion, ii, jj)[0] - target).norm(dim=-1).mean()
        result = optimize_view(
            {"poses": initial_poses, "points": initial_points, "ii": ii, "jj": jj,
             "uv": target, "weight": torch.ones(ii.numel(), dtype=dtype), "K": K,
             "distortion": distortion},
            [0], 4,
        )
        after = (_project(result["poses"], result["points"], K, distortion, ii, jj)[0] - target).norm(dim=-1).mean()
        self.assertLess(float(after), float(before) * 1.0e-3)
        self.assertTrue(torch.isfinite(result["poses"]).all())

    def test_loss_per_pixel_uses_only_valid_projected_observations(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
        points = torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 0.1]], dtype=dtype)
        K = torch.tensor(
            [[[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]],
            dtype=dtype,
        ).repeat(3, 1, 1)
        ii = torch.arange(3).repeat_interleave(2)
        jj = torch.arange(2).repeat(3)
        view = {
            "poses": poses,
            "points": points,
            "ii": ii,
            "jj": jj,
            "uv": torch.tensor(
                [[161.0, 120.0], [1000.0, 1000.0]] * 3, dtype=dtype
            ),
            "weight": torch.ones(6, dtype=dtype),
            "K": K,
            "distortion": torch.zeros(3, 4, dtype=dtype),
        }

        result = optimize_view(view, [0], 0)

        self.assertEqual(int(result["valid_observations"]), 3)
        self.assertAlmostEqual(float(result["loss"]), 1.5)
        self.assertAlmostEqual(float(result["loss_per_pixel"]), 0.5)

    def test_global_initialization_and_dense_reconstruction(self):
        dataset = _Dataset()
        frames = FrameStore(dataset)
        tracker = WindowTracker(_Geometry(), _Tracks(), _Features(12), frames, dataset, {"device": "cpu"})
        graph = FactorGraph(frames)
        graph.add_factors(tracker.track([0, 1, 2, 3]))
        graph.add_factors(tracker.track([2, 3, 4, 5]))
        graph.initialize_global()
        graph.optimize(graph.full_view(), [0], 1)
        for _, depth, _, _ in frames.dense.values():
            x = torch.arange(depth.shape[1], dtype=depth.dtype)[None]
            depth[:] = 1.0 + 0.05 * x
        with patch(
            "sfm.reconstruction.fit_disparity_affine",
            side_effect=lambda predicted, sparse, weights: (
                predicted.new_tensor(0.0),
                predicted.new_tensor(1.0),
                torch.ones_like(predicted, dtype=torch.bool),
            ),
        ):
            result = reconstruct(graph, frames)
        self.assertEqual(result["sparse_points"].shape[1], 3)
        self.assertEqual(
            result["sparse_frame_ids"].shape,
            result["sparse_points"].shape[:1],
        )
        self.assertEqual(
            result["sparse_obs_cnt"].shape,
            result["sparse_points"].shape[:1],
        )
        self.assertTrue(
            bool((result["sparse_obs_cnt"][result["sparse_inliers"]] >= 3).all())
        )
        self.assertTrue(
            bool((result["sparse_obs_cnt"][~result["sparse_inliers"]] == -1).all())
        )
        self.assertEqual(result["dense_points"].shape[1], 3)
        self.assertEqual(result["dense_points"].shape[0], result["dense_colors"].shape[0])
        self.assertEqual(set(result["dense_frame_ids"].tolist()), set(range(6)))

    def test_triangulation_rejects_point_without_positive_owner_depth(self):
        graph = FactorGraph(None)
        graph.point_count = 1
        graph.point_positions = torch.zeros(1, 3)
        graph.point_initialized = torch.ones(1, dtype=torch.bool)
        graph.point_references = torch.tensor([0])
        graph.intrinsics = {frame: torch.eye(3) for frame in range(4)}
        graph.observations = [(
            0,
            torch.arange(4),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, 2),
            torch.ones(4),
        )]
        graph.observation_ids = [torch.arange(4)]
        poses = torch.eye(4).repeat(4, 1, 1)
        triangulated = (
            torch.tensor([[0.0, 0.0, 4.0]]),
            torch.tensor([False, True, True, True]),
            torch.tensor([True]),
            torch.tensor([2.0]),
        )

        with patch(
            "sfm.factor_graph.triangulate_tracks", return_value=triangulated
        ):
            graph._triangulate_global(list(range(4)), poses)

        self.assertFalse(bool(graph.point_initialized[0]))
        self.assertEqual(graph.inactive_observation_ids, {0, 1, 2, 3})


class TracksFrontendTest(unittest.TestCase):
    def test_sparse_ply_contains_frame_id_and_obs_cnt(self):
        captured = {}

        class PlyElement:
            @staticmethod
            def describe(vertices, name):
                captured["names"] = vertices.dtype.names
                captured["frame_id"] = vertices["frame_id"].copy()
                captured["obs_cnt"] = vertices["obs_cnt"].copy()
                return name

        class PlyData:
            def __init__(self, elements, text):
                self.elements = elements

            def write(self, path):
                captured["path"] = path

        export = _load_export()
        with patch.object(export, "PlyElement", PlyElement), patch.object(
            export, "PlyData", PlyData
        ):
            export.save_ply(
                "sparse_tracks.ply",
                torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
                scalar_fields={
                    "frame_id": torch.tensor([2, 4]),
                    "obs_cnt": torch.tensor([3, 7]),
                },
            )
        self.assertEqual(
            captured["names"], ("x", "y", "z", "frame_id", "obs_cnt")
        )
        self.assertEqual(captured["frame_id"].tolist(), [2, 4])
        self.assertEqual(captured["obs_cnt"].tolist(), [3, 7])

    def test_camera_obj_contains_real_line_segments(self):
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[1, 0, 3] = -1
        scene_points = torch.tensor([[-1.0, -1.0, 1.0], [1.0, 1.0, 3.0]])
        export = _load_export()
        with TemporaryDirectory() as directory:
            obj_path = Path(directory) / "camera_poses.obj"
            export.save_camera_wireframes(
                obj_path, poses, scene_points, frame_ids=[3, 7]
            )
            obj_lines = obj_path.read_text(encoding="ascii").splitlines()

        self.assertEqual(
            [line for line in obj_lines if line.startswith("g ")],
            ["g camera_3", "g camera_7"],
        )
        obj_vertices = [line for line in obj_lines if line.startswith("v ")]
        self.assertEqual(len(obj_vertices), 10)
        self.assertFalse(any(line.startswith("p ") for line in obj_lines))
        self.assertAlmostEqual(float(obj_vertices[0].split()[1]), 0.0)
        self.assertAlmostEqual(float(obj_vertices[5].split()[1]), 1.0)
        obj_edges = [line for line in obj_lines if line.startswith("l ")]
        self.assertEqual(len(obj_edges), 16)
        first_edges = [int(index) for line in obj_edges[:8] for index in line.split()[1:]]
        second_edges = [int(index) for line in obj_edges[8:] for index in line.split()[1:]]
        self.assertLessEqual(max(first_edges), 5)
        self.assertGreaterEqual(min(second_edges), 6)

    def test_query_selection_uses_global_scores_without_grid_reservation(self):
        cluster = torch.stack(
            (
                torch.arange(60).remainder(8).float(),
                torch.arange(60).remainder(6).float(),
            ),
            -1,
        )
        remote = torch.tensor([[79.0, 59.0]])
        points = torch.cat((cluster, remote))
        scores = torch.cat((torch.arange(60, dtype=torch.float32), torch.tensor([-1.0])))
        selected = _top_score_indices(points, scores, 48)
        self.assertEqual(selected.numel(), 48)
        self.assertEqual(selected.unique().numel(), 48)
        self.assertNotIn(60, selected.tolist())
        self.assertEqual(
            set(selected.tolist()), set(scores.topk(48).indices.tolist())
        )

    def test_glob3r_requires_refinement_outputs(self):
        class Model:
            def match_pair(self, patch_tokens, encoder, images, reference):
                return SimpleNamespace(warp_stages=[], confidence_stages=[])

        frontend = Glob3RTracks(Model())
        frontend.prepare_window(torch.zeros(1, 3, 3, 4, 5), (None, None))
        with self.assertRaisesRegex(RuntimeError, "refinement"):
            frontend.track(0, torch.tensor([[1.0, 1.0]]))

    def test_glob3r_reencodes_matching_state_for_masked_images(self):
        class Model:
            def encode_matching_state(self, images):
                self.images = images
                return "masked_patch_tokens", "masked_encoder"

        model = Model()
        frontend = Glob3RTracks(model)
        images = torch.ones(1, 3, 3, 4, 5)
        frontend.prepare_window(images, None)

        self.assertIs(model.images, images)
        self.assertEqual(frontend.patch_tokens, "masked_patch_tokens")
        self.assertEqual(frontend.encoder, "masked_encoder")

    def test_glob3r_samples_dense_warp_at_shared_image_queries(self):
        height, width = 4, 5
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        base = torch.stack((xx, yy)).float()
        warp = torch.stack((base + 0.25, base + 0.5))[None]
        confidence = torch.ones(1, 2, 1, height, width)

        class Model:
            def match_pair(self, patch_tokens, encoder, images, reference):
                return SimpleNamespace(
                    warp_stages=[warp], confidence_stages=[confidence],
                    coarse_warp=warp, coarse_confidence=confidence,
                    target_indices=[0, 2],
                )

        frontend = Glob3RTracks(Model())
        frontend.prepare_window(torch.zeros(1, 3, 3, height, width), (None, None))
        queries = torch.tensor([[1.25, 1.5], [3.0, 2.0]])
        result = frontend.track(1, queries)
        self.assertTrue(torch.allclose(result["tracks"][1], queries))
        self.assertTrue(torch.allclose(result["tracks"][0], queries + 0.25))
        self.assertTrue(torch.allclose(result["tracks"][2], queries + 0.5))

    def test_vggsfm_reorders_the_same_queries_back_to_input_frames(self):
        class Tracker(torch.nn.Module):
            def process_images_to_fmaps(self, images):
                return images[:, :, :1]

            def forward(self, images, queries, fmaps, fine_tracking):
                offsets = torch.arange(images.shape[1], device=images.device)[None, :, None, None]
                tracks = queries[:, None] + offsets
                confidence = torch.ones(*tracks.shape[:-1], device=images.device)
                return tracks, tracks, confidence, confidence

        frontend = VGGSfMTracks(
            Tracker(), (4, 5), tracker_size=8, mixed_precision="none"
        )
        frontend.prepare_window(torch.zeros(1, 3, 3, 4, 5), None)
        queries = torch.tensor([[1.0, 1.0], [3.0, 2.0]])
        result = frontend.track(1, queries)
        self.assertTrue(torch.allclose(result["tracks"][1], queries))
        self.assertTrue(torch.allclose(result["tracks"][0], queries + 1 / 1.6))
        self.assertTrue(torch.allclose(result["tracks"][2], queries + 2 / 1.6))
        self.assertTrue(torch.equal(result["confidence"], torch.ones(3, 2)))

    def test_vggsfm_preserves_raw_visibility_and_score_for_visualization(self):
        class Tracker(torch.nn.Module):
            def process_images_to_fmaps(self, images):
                return images[:, :, :1]

            def forward(self, images, queries, fmaps, fine_tracking):
                tracks = queries[:, None].expand(-1, images.shape[1], -1, -1).clone()
                visibility = torch.tensor(
                    [[[0.3, 0.3], [0.4, 0.7], [0.8, 0.2]]],
                    device=images.device,
                )
                score = torch.tensor(
                    [[[0.2, 0.2], [0.6, 0.4], [0.9, 0.8]]],
                    device=images.device,
                )
                return tracks, tracks, visibility, score

        frontend = VGGSfMTracks(
            Tracker(), (4, 5), tracker_size=8, mixed_precision="none"
        )
        frontend.prepare_window(torch.zeros(1, 3, 3, 4, 5), None)
        result = frontend.track(1, torch.tensor([[1.0, 1.0], [3.0, 2.0]]))

        expected_visibility = torch.tensor(
            [[0.4, 0.7], [1.0, 1.0], [0.8, 0.2]]
        )
        expected_score = torch.tensor(
            [[0.6, 0.4], [1.0, 1.0], [0.9, 0.8]]
        )
        expected_confidence = torch.tensor(
            [[0.24, 0.0], [1.0, 1.0], [0.72, 0.16]]
        )
        self.assertTrue(torch.allclose(
            result["visualization_confidence"], expected_visibility
        ))
        self.assertTrue(torch.allclose(
            result["visualization_score"], expected_score
        ))
        self.assertTrue(torch.allclose(result["confidence"], expected_confidence))

    def test_vggsfm_requires_track_scores(self):
        class Tracker(torch.nn.Module):
            def process_images_to_fmaps(self, images):
                return images[:, :, :1]

            def forward(self, images, queries, fmaps, fine_tracking):
                tracks = queries[:, None].expand(-1, images.shape[1], -1, -1).clone()
                visibility = torch.ones(*tracks.shape[:-1], device=images.device)
                return tracks, tracks, visibility, None

        frontend = VGGSfMTracks(
            Tracker(), (4, 5), tracker_size=8, mixed_precision="none"
        )
        frontend.prepare_window(torch.zeros(1, 3, 3, 4, 5), None)
        with self.assertRaisesRegex(RuntimeError, "track scores"):
            frontend.track(0, torch.tensor([[1.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
