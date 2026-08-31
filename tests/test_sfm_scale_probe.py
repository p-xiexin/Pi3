from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from sfm.frontend_cache import save_frontend_cache
from sfm.probe_scale import (
    _edge_directions,
    _mst_edges,
    _normalized_bearing_linearization,
    _pi3_center_candidate,
    _retriangulate_probe_graph,
    _strong_bridges,
    _translation_direction_average,
    _window_path,
    probe,
)
from sfm.factor_graph import FactorGraph
from sfm.geometry import camera_centers
from sfm.tracker import FrameStore, WindowTracker


class ScaleProbeTest(unittest.TestCase):
    def test_mst_and_bridge_metrics(self):
        identity = torch.eye(4)
        edges = [
            (0, 2, identity, 3.0),
            (0, 1, identity, 2.0),
            (1, 2, identity, 1.0),
        ]
        selected = _mst_edges(edges)
        self.assertEqual([(edge[0], edge[1]) for edge in selected], [(0, 2), (0, 1)])
        point_frames = {0: {0, 1, 2, 3}, 1: {0, 1, 2}, 2: {2, 3}}
        self.assertEqual(_strong_bridges(point_frames, 2), [0])

    def test_window_path_reports_local_shrink(self):
        frame_ids = torch.arange(4)
        poses = torch.eye(4).repeat(4, 1, 1)
        poses[:, 0, 3] = -torch.tensor([0.0, 1.0, 2.0, 3.0])
        shrunk = poses.clone()
        shrunk[:, 0, 3] = -torch.tensor([0.0, 1.0, 1.5, 2.0])
        before = _window_path(poses, frame_ids, [1, 2, 3])
        after = _window_path(shrunk, frame_ids, [1, 2, 3])
        self.assertAlmostEqual(float(after / before), 0.5)

    def test_normalized_bearing_is_scale_invariant_with_correct_jacobian(self):
        dtype = torch.float64
        ray = torch.tensor([0.2, -0.1, 1.0], dtype=dtype)
        ray /= ray.norm()
        projector = (
            torch.eye(3, dtype=dtype) - ray[:, None] * ray[None, :]
        )[None]
        centers = torch.tensor([[0.1, -0.3, 0.2]], dtype=dtype)
        points = torch.tensor([[1.4, 0.6, 3.2]], dtype=dtype)
        ii = jj = torch.zeros(1, dtype=torch.long)
        error, _, Jx = _normalized_bearing_linearization(
            centers, points, ii, jj, projector
        )
        scaled_error, _, scaled_Jx = _normalized_bearing_linearization(
            4.0 * centers, 4.0 * points, ii, jj, projector
        )
        torch.testing.assert_close(scaled_error, error)
        torch.testing.assert_close(4.0 * scaled_Jx, Jx)

        offset = (points[0] - centers[0]).requires_grad_()
        numeric = torch.autograd.functional.jacobian(
            lambda value: projector[0] @ (value / value.norm()), offset
        )
        torch.testing.assert_close(Jx[0], numeric, atol=1.0e-10, rtol=1.0e-10)

        scale_direction = points[0] - centers[0]
        torch.testing.assert_close(
            Jx[0] @ scale_direction,
            torch.zeros(3, dtype=dtype),
            atol=1.0e-10,
            rtol=0.0,
        )

    def test_edge_translation_converts_to_oriented_world_direction(self):
        dtype = torch.float64
        angle = torch.tensor(0.7, dtype=dtype)
        rotation = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0.0],
                [torch.sin(angle), torch.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )
        centers = torch.tensor([[0.2, -0.3, 0.5], [1.4, 0.8, -0.1]], dtype=dtype)
        poses = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
        poses[1, :3, :3] = rotation
        poses[:, :3, 3] = -torch.einsum(
            "sij,sj->si", poses[:, :3, :3], centers
        )
        relative = poses[1] @ torch.linalg.inv(poses[0])
        _, _, direction, _, _ = _edge_directions(
            poses, torch.tensor([3, 8]), [(3, 8, relative, 1.0)]
        )
        expected = centers[1] - centers[0]
        expected /= expected.norm()
        torch.testing.assert_close(direction[0], expected)

    def test_direction_average_recovers_noncollinear_relative_baselines(self):
        dtype = torch.float64
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.0],
                [1.7, 0.4, 0.2],
                [2.5, 1.0, 0.3],
                [3.8, 1.7, 0.7],
                [5.0, 2.8, 1.1],
            ],
            dtype=dtype,
        )
        poses = torch.eye(4, dtype=dtype).repeat(len(centers), 1, 1)
        poses[:, :3, 3] = -centers
        edges = []
        for source in range(len(centers)):
            for target in range(source + 1, len(centers)):
                relative = poses[target] @ torch.linalg.inv(poses[source])
                relative[:3, 3] /= relative[:3, 3].norm()
                edges.append((source, target, relative, 1.0))
        candidate, near_null, _ = _translation_direction_average(
            poses, torch.arange(len(centers)), edges
        )
        recovered = camera_centers(candidate)
        scale = (recovered * centers).sum() / recovered.square().sum()
        torch.testing.assert_close(scale * recovered, centers, atol=1.0e-9, rtol=1.0e-9)
        self.assertEqual(near_null, 1)

    def test_metric_scaled_pi3_centers_chain_through_window_anchor(self):
        dtype = torch.float64
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.1, 0.0],
                [1.5, 0.3, 0.1],
                [2.4, 0.7, 0.2],
                [3.5, 1.2, 0.4],
                [4.8, 2.0, 0.7],
            ],
            dtype=dtype,
        )
        initial = torch.eye(4, dtype=dtype).repeat(len(centers), 1, 1)
        first = torch.eye(4, dtype=dtype).repeat(4, 1, 1)
        first[:, :3, 3] = centers[:4]
        second = torch.eye(4, dtype=dtype).repeat(4, 1, 1)
        second[:, :3, 3] = (centers[2:] - centers[2]) / 2.0
        candidate, diagnostics = _pi3_center_candidate(
            initial,
            torch.arange(len(centers)),
            [
                (torch.tensor([0, 1, 2, 3]), first, torch.tensor(1.0)),
                (torch.tensor([2, 3, 4, 5]), second, torch.tensor(2.0)),
            ],
        )
        torch.testing.assert_close(camera_centers(candidate), centers)
        self.assertAlmostEqual(diagnostics[1][2], 1.0)
        self.assertAlmostEqual(diagnostics[1][3], 0.0)

    def test_pi3_probe_retriangulates_from_raw_tracks(self):
        dtype = torch.float64
        centers = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=dtype,
        )
        poses = torch.eye(4, dtype=dtype).repeat(len(centers), 1, 1)
        poses[:, :3, 3] = -centers
        point = torch.tensor([0.2, 0.1, 4.0], dtype=dtype)
        uv = torch.stack(
            [(point - center)[:2] / (point - center)[2] for center in centers]
        )

        graph = FactorGraph(None)
        graph.poses = {frame: torch.eye(4, dtype=dtype) for frame in range(4)}
        graph.intrinsics = {frame: torch.eye(3, dtype=dtype) for frame in range(4)}
        graph.point_count = 1
        graph.point_references = torch.tensor([0])
        graph.point_anchors = torch.zeros(1, 3, dtype=dtype)
        graph.point_positions = torch.tensor([[99.0, 99.0, 99.0]], dtype=dtype)
        graph.point_initialized = torch.ones(1, dtype=torch.bool)
        graph.observations = [
            (
                0,
                torch.arange(4),
                torch.zeros(4, dtype=torch.long),
                uv,
                torch.ones(4, dtype=dtype),
            )
        ]
        graph.observation_ids = [torch.arange(4)]
        graph.next_observation_id = 4
        graph.inactive_observation_ids = {1}

        view = _retriangulate_probe_graph(
            graph, torch.arange(4), poses, set()
        )

        torch.testing.assert_close(view["points"][0], point, atol=1.0e-9, rtol=1.0e-9)
        torch.testing.assert_close(view["poses"], poses)
        self.assertEqual(view["ii"].numel(), 4)
        self.assertEqual(view["ii"].shape, view["jj"].shape)
        self.assertEqual(view["ii"].shape, view["weight"].shape)
        self.assertEqual(view["ii"].shape, view["observation_ids"].shape)

    def test_pi3_released_optimization_probe_smoke(self):
        from tests.test_full_sfm import (
            _Dataset,
            _Features,
            _Geometry,
            _Tracks,
            _synthetic_relative_pose,
        )

        dataset = _Dataset()
        frames = FrameStore(dataset)
        tracker = WindowTracker(
            _Geometry(), _Tracks(), _Features(12), frames, dataset, {"device": "cpu"}
        )
        with TemporaryDirectory() as root, patch(
            "sfm.geometric_verification.MIN_PAIR_MATCHES", 3
        ), patch(
            "sfm.geometric_verification.MIN_PAIR_INLIERS", 3
        ), patch(
            "sfm.geometric_verification._estimate_relative_pose",
            side_effect=_synthetic_relative_pose,
        ):
            packets = [tracker.track([0, 1, 2, 3]), tracker.track([2, 3, 4, 5])]
            path = Path(root) / "data.h5"
            save_frontend_cache(path, packets, frames)
            output = StringIO()
            with redirect_stdout(output):
                probe(path, device="cpu", bearing_iterations=1, compare=True)

        text = output.getvalue()
        self.assertIn("BASELINE_SCALE_GAUGE initialization=essential", text)
        self.assertIn("PI3_SCALE_GAUGE initialization=pi3_metric", text)
        self.assertIn("PI3_TRI points=", text)
        self.assertIn("PI3_OPT_RAY stage=fixed", text)
        self.assertIn("PI3_OPT_WIN win=1", text)
        self.assertIn("PI3_OPT_FILTER stage=final", text)


if __name__ == "__main__":
    unittest.main()
