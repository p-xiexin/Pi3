import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sfm.colmap_optimizer import (
    _load_pycolmap,
    optimize_view_colmap,
    optimize_view_colmap_two_rounds,
)
from sfm.factor_graph import FactorGraph
from sfm.optimizer import _project


class _Rotation3d:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=np.float64).copy()

    def matrix(self):
        return self._matrix


class _Rigid3d:
    def __init__(self, rotation, translation=None):
        if translation is None:
            matrix = np.asarray(rotation, dtype=np.float64)
            self.rotation = _Rotation3d(matrix[:3, :3])
            self.translation = matrix[:3, 3].copy()
        else:
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

    def add_camera_with_trivial_rig(self, camera):
        self.cameras[camera.camera_id] = camera

    def add_point3D(self, xyz, track, color):
        point_id = len(self.points3D) + 1
        self.points3D[point_id] = SimpleNamespace(
            xyz=np.asarray(xyz).copy(), track=track, color=np.asarray(color).copy()
        )
        return point_id

    def add_image_with_trivial_frame(self, image, cam_from_world):
        image.frame_id = image.image_id
        image._cam_from_world = cam_from_world
        image.has_pose = True
        self.images[image.image_id] = image

    def reg_image_ids(self):
        return list(self.images)


class _BundleAdjustmentOptions:
    def __init__(self):
        self.refine_principal_point = False
        self.refine_extra_params = False
        self.refine_rig_from_world = False
        self.refine_sensor_from_rig = True
        self.refine_focal_length = False
        self.ceres = SimpleNamespace(
            loss_function_scale=None,
            loss_function_type=None,
            solver_options=SimpleNamespace(max_num_iterations=None),
        )


class _BundleAdjustmentConfig:
    def __init__(self):
        self.images = set()
        self.constant_frames = set()
        self.fixed_gauge = None

    def add_image(self, image_id):
        self.images.add(image_id)

    def set_constant_rig_from_world_pose(self, frame_id):
        self.constant_frames.add(frame_id)

    def fix_gauge(self, gauge):
        self.fixed_gauge = gauge


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
            def __init__(self, image_id, name, camera_id, points2D):
                self.image_id = image_id
                self.name = name
                self.camera_id = camera_id
                self.points2D = points2D
                self.frame_id = None
                self.has_pose = False

            def cam_from_world(self):
                return self._cam_from_world

        def create_default_bundle_adjuster(options, config, reconstruction):
            state.reconstruction = reconstruction
            state.options = options
            state.config = config

            class Adjuster:
                @staticmethod
                def solve():
                    state.called = True
                    return SimpleNamespace(
                        is_solution_usable=lambda: True,
                        brief_report=lambda: "ok",
                    )

            return Adjuster()

        module = SimpleNamespace(
            Reconstruction=_Reconstruction,
            Camera=Camera,
            Track=_Track,
            Rotation3d=_Rotation3d,
            Rigid3d=_Rigid3d,
            Point2D=Point2D,
            Point2DList=list,
            Image=Image,
            BundleAdjustmentOptions=_BundleAdjustmentOptions,
            BundleAdjustmentConfig=_BundleAdjustmentConfig,
            BundleAdjustmentGauge=SimpleNamespace(TWO_CAMS_FROM_WORLD="two_cameras"),
            LossFunctionType=SimpleNamespace(CAUCHY="cauchy"),
            create_default_bundle_adjuster=create_default_bundle_adjuster,
        )
        return module, state

    def test_backend_requires_pycolmap_4_1_1(self):
        incompatible = SimpleNamespace(COLMAP_version="COLMAP 3.10")
        with patch(
            "sfm.colmap_optimizer.importlib.import_module",
            return_value=incompatible,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires pycolmap 4.1.1"):
                _load_pycolmap()

        compatible = SimpleNamespace(COLMAP_version="COLMAP 4.1.1")
        with patch(
            "sfm.colmap_optimizer.importlib.import_module",
            return_value=compatible,
        ):
            self.assertIs(_load_pycolmap(), compatible)

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
        self.assertEqual(state.options.ceres.loss_function_scale, 2.0)
        self.assertEqual(state.options.ceres.loss_function_type, "cauchy")
        self.assertTrue(state.options.refine_principal_point)
        self.assertFalse(state.options.refine_extra_params)
        self.assertTrue(state.options.refine_rig_from_world)
        self.assertFalse(state.options.refine_sensor_from_rig)
        self.assertTrue(state.options.refine_focal_length)
        self.assertEqual(state.options.ceres.solver_options.max_num_iterations, 7)
        self.assertEqual(state.config.images, {1, 2, 3})
        self.assertEqual(state.config.constant_frames, {1})
        self.assertEqual(state.config.fixed_gauge, "two_cameras")
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

    def test_two_round_colmap_uses_shared_filtering_protocol(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, -0.2, -0.4], dtype=dtype)
        points = torch.tensor(
            [[0.1, 0.0, 4.0], [-0.2, 0.1, 5.0]], dtype=dtype
        )
        K = torch.tensor(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).repeat(3, 1, 1)
        ii = torch.arange(3).repeat_interleave(2)
        jj = torch.arange(2).repeat(3)
        distortion = torch.zeros(3, 4, dtype=dtype)
        uv, _ = _project(poses, points, K, distortion, ii, jj)
        uv[-1, 0] += 4.01
        view = {
            "frame_ids": torch.tensor([2, 4, 7]),
            "poses": poses,
            "points": points,
            "K": K,
            "distortion": distortion,
            "ii": ii,
            "jj": jj,
            "uv": uv,
            "weight": torch.ones(ii.numel(), dtype=dtype),
        }
        observation_counts = []

        def passthrough(current, fixed_ids, iterations):
            observation_counts.append(int(current["ii"].numel()))
            return {
                "poses": current["poses"].clone(),
                "points": current["points"].clone(),
                "K": current["K"].clone(),
                "distortion": current["distortion"].clone(),
                "ba_backend": "colmap",
            }

        with patch(
            "sfm.colmap_optimizer.optimize_view_colmap",
            side_effect=passthrough,
        ):
            result = optimize_view_colmap_two_rounds(
                view,
                [0],
                bearing_iterations=0,
                first_iterations=7,
                second_iterations=3,
            )

        self.assertEqual(observation_counts, [6, 3])
        self.assertEqual(int(result["first_inlier_observations"]), 3)
        self.assertEqual(int(result["valid_observations"]), 3)
        self.assertTrue(
            torch.equal(result["point_inliers"], torch.tensor([True, False]))
        )
        self.assertTrue(
            torch.equal(
                result["observation_inliers"],
                torch.tensor([True, False, True, False, True, False]),
            )
        )
        self.assertEqual(result["ba_backend"], "colmap")

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
            "sfm.colmap_optimizer.optimize_view_colmap_two_rounds",
            return_value=backend_result,
        ) as backend:
            result = graph.optimize(view, [3], 5, backend="colmap")

        backend.assert_called_once()
        self.assertEqual(result["scope"], "global")
        self.assertEqual(result["ba_backend"], "colmap")
        self.assertTrue(torch.equal(graph.intrinsics[3], optimized_K[0]))
        self.assertTrue(bool(graph.point_initialized[0]))


if __name__ == "__main__":
    unittest.main()
