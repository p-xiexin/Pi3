import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sfm.colmap_optimizer import optimize_view_colmap
from sfm.factor_graph import FactorGraph
from sfm.optimizer import _project


class _Rotation3d:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=np.float64).copy()

    def matrix(self):
        return self._matrix


class _Rigid3d:
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = np.asarray(translation, dtype=np.float64).copy()


class _Track:
    def __init__(self):
        self.elements = []

    def add_element(self, image_id, point2D_idx):
        self.elements.append(SimpleNamespace(image_id=image_id, point2D_idx=point2D_idx))


class _Reconstruction:
    def __init__(self):
        self.cameras = {}
        self.images = {}
        self.points3D = {}

    def add_camera(self, camera):
        self.cameras[camera.camera_id] = camera

    def add_point3D(self, xyz, track, color):
        point_id = len(self.points3D) + 1
        self.points3D[point_id] = SimpleNamespace(
            xyz=np.asarray(xyz).copy(), track=track, color=np.asarray(color).copy()
        )
        return point_id

    def add_image(self, image):
        self.images[image.image_id] = image


class _BundleAdjustmentOptions:
    def __init__(self):
        self.loss_function_scale = None
        self.loss_function_type = None
        self.refine_principal_point = False
        self.refine_extra_params = False
        self.refine_extrinsics = False
        self.refine_focal_length = False
        self.solver_options = SimpleNamespace(max_num_iterations=None)


class ColmapOptimizerTest(unittest.TestCase):
    def _fake_pycolmap(self):
        state = SimpleNamespace(called=False, reconstruction=None, options=None)

        class Camera:
            def __init__(self, model, width, height, params, camera_id):
                self.model = model
                self.width = width
                self.height = height
                self.params = np.asarray(params).copy()
                self.camera_id = camera_id

        class Point2D:
            def __init__(self, xy, point3D_id):
                self.xy = np.asarray(xy).copy()
                self.point3D_id = point3D_id

        class Image:
            def __init__(self, image_id, name, camera_id, cam_from_world, points2D):
                self.image_id = image_id
                self.name = name
                self.camera_id = camera_id
                self.cam_from_world = cam_from_world
                self.points2D = points2D
                self.registered = False

        def bundle_adjustment(reconstruction, options):
            state.called = True
            state.reconstruction = reconstruction
            state.options = options

        module = SimpleNamespace(
            Reconstruction=_Reconstruction,
            Camera=Camera,
            Track=_Track,
            Rotation3d=_Rotation3d,
            Rigid3d=_Rigid3d,
            Point2D=Point2D,
            ListPoint2D=list,
            Image=Image,
            BundleAdjustmentOptions=_BundleAdjustmentOptions,
            LossFunctionType=SimpleNamespace(CAUCHY="cauchy"),
            bundle_adjustment=bundle_adjustment,
        )
        return module, state

    def test_colmap_backend_builds_tracks_and_returns_comparable_tensors(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, -0.2, -0.4], dtype=dtype)
        points = torch.tensor([[0.1, 0.0, 4.0], [-0.2, 0.1, 5.0]], dtype=dtype)
        K = torch.tensor(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).repeat(3, 1, 1)
        ii = torch.arange(3).repeat_interleave(2)
        jj = torch.arange(2).repeat(3)
        uv, _ = _project(poses, points, K, torch.zeros(3, 4, dtype=dtype), ii, jj)
        view = {
            "frame_ids": torch.tensor([2, 4, 7]),
            "poses": poses,
            "points": points,
            "K": K,
            "distortion": torch.zeros(3, 4, dtype=dtype),
            "ii": ii,
            "jj": jj,
            "uv": uv,
            "weight": torch.ones(ii.numel(), dtype=dtype),
        }
        fake, state = self._fake_pycolmap()

        with patch("sfm.colmap_optimizer._load_pycolmap", return_value=fake):
            result = optimize_view_colmap(view, [0], 7)

        self.assertTrue(state.called)
        self.assertEqual(state.options.loss_function_scale, 2.0)
        self.assertEqual(state.options.loss_function_type, "cauchy")
        self.assertTrue(state.options.refine_principal_point)
        self.assertTrue(state.options.refine_extra_params)
        self.assertTrue(state.options.refine_extrinsics)
        self.assertTrue(state.options.refine_focal_length)
        self.assertEqual(state.options.solver_options.max_num_iterations, 7)
        self.assertEqual(len(state.reconstruction.cameras), 1)
        self.assertEqual(len(state.reconstruction.images), 3)
        self.assertEqual(len(state.reconstruction.points3D), 2)
        self.assertTrue(all(
            len(point.track.elements) == 3
            for point in state.reconstruction.points3D.values()
        ))
        self.assertEqual(result["ba_backend"], "colmap")
        self.assertTrue(torch.allclose(result["poses"], poses))
        self.assertTrue(torch.allclose(result["points"], points))
        self.assertTrue(torch.allclose(result["K"], K))
        self.assertTrue(torch.equal(result["distortion"], torch.zeros_like(result["distortion"])))
        self.assertEqual(int(result["valid_observations"]), 6)
        self.assertAlmostEqual(float(result["loss_per_pixel"]), 0.0)

    def test_duplicate_point_image_observation_is_rejected(self):
        fake, _ = self._fake_pycolmap()
        view = {
            "frame_ids": torch.tensor([0]),
            "poses": torch.eye(4).unsqueeze(0),
            "points": torch.tensor([[0.0, 0.0, 1.0]]),
            "K": torch.eye(3).unsqueeze(0),
            "distortion": torch.zeros(1, 4),
            "ii": torch.tensor([0, 0]),
            "jj": torch.tensor([0, 0]),
            "uv": torch.zeros(2, 2),
            "weight": torch.ones(2),
        }
        with patch("sfm.colmap_optimizer._load_pycolmap", return_value=fake):
            with self.assertRaisesRegex(ValueError, "at most one observation"):
                optimize_view_colmap(view, [0], 1)

    def test_factor_graph_dispatches_colmap_and_commits_intrinsics(self):
        graph = FactorGraph(None)
        graph.poses = {3: torch.eye(4)}
        graph.intrinsics = {3: torch.eye(3)}
        graph.point_positions = torch.zeros(1, 3)
        graph.point_initialized = torch.zeros(1, dtype=torch.bool)
        view = {
            "scope": "global",
            "frame_ids": torch.tensor([3]),
            "point_ids": torch.tensor([0]),
            "poses": torch.eye(4).unsqueeze(0),
            "points": torch.tensor([[0.0, 0.0, 1.0]]),
            "K": torch.eye(3).unsqueeze(0),
            "distortion": torch.zeros(1, 4),
            "ii": torch.tensor([0]),
            "jj": torch.tensor([0]),
            "uv": torch.zeros(1, 2),
            "weight": torch.ones(1),
        }
        optimized_K = torch.tensor(
            [[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]]
        )
        backend_result = {
            "poses": view["poses"].clone(),
            "points": view["points"].clone(),
            "K": optimized_K,
            "distortion": torch.zeros(1, 4),
            "loss": torch.tensor(0.0),
            "loss_per_pixel": torch.tensor(0.0),
            "valid_observations": torch.tensor(1),
            "ba_backend": "colmap",
        }

        with patch(
            "sfm.colmap_optimizer.optimize_view_colmap", return_value=backend_result
        ) as backend:
            result = graph.optimize(view, [3], 5, backend="colmap")

        backend.assert_called_once()
        self.assertEqual(result["scope"], "global")
        self.assertEqual(result["ba_backend"], "colmap")
        self.assertTrue(torch.equal(graph.intrinsics[3], optimized_K[0]))
        self.assertTrue(bool(graph.point_initialized[0]))


if __name__ == "__main__":
    unittest.main()
