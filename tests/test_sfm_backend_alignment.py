import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sfm.optimizer import (
    _project,
    _select_scale_gauge,
    filter_reprojection_observations,
    optimize_view,
    optimize_view_two_rounds,
)
from sfm.reconstruction import (
    fit_disparity_affine,
    reconstruct,
    sample_map_at_points,
)


def _filter_view():
    dtype = torch.float64
    poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.tensor([0.0, -0.2, -0.4], dtype=dtype)
    points = torch.tensor(
        [[0.1, 0.0, 5.0], [-0.2, 0.1, 4.0], [0.1, 0.0, -3.0]],
        dtype=dtype,
    )
    K = torch.tensor(
        [[120.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    ).repeat(3, 1, 1)
    ii = torch.arange(3).repeat_interleave(3)
    jj = torch.arange(3).repeat(3)
    distortion = torch.zeros(3, 4, dtype=dtype)
    uv, _ = _project(poses, points, K, distortion, ii, jj)
    uv[7, 0] += 4.01
    return {
        "poses": poses,
        "points": points,
        "ii": ii,
        "jj": jj,
        "uv": uv,
        "weight": torch.ones(ii.numel(), dtype=dtype),
        "K": K,
        "distortion": distortion,
    }


class ReprojectionBackendTest(unittest.TestCase):
    def test_four_pixel_positive_depth_filter_preserves_view_indexing(self):
        view = _filter_view()
        result = filter_reprojection_observations(
            view, view["poses"], view["points"]
        )

        self.assertTrue(
            torch.equal(
                result["point_inliers"], torch.tensor([True, False, False])
            )
        )
        expected = torch.tensor(
            [True, False, False, True, False, False, True, False, False]
        )
        self.assertTrue(torch.equal(result["observation_inliers"], expected))
        self.assertAlmostEqual(float(result["reprojection_error"][7]), 4.01)
        self.assertEqual(result["observation_inliers"].numel(), view["ii"].numel())

    def test_two_round_interface_filters_between_bundle_adjustments(self):
        view = _filter_view()
        result = optimize_view_two_rounds(
            view,
            [0],
            bearing_iterations=0,
            first_iterations=0,
            second_iterations=0,
        )

        self.assertEqual(result["poses"].shape, view["poses"].shape)
        self.assertEqual(result["points"].shape, view["points"].shape)
        self.assertEqual(int(result["first_inlier_observations"]), 3)
        self.assertEqual(int(result["valid_observations"]), 3)
        self.assertTrue(
            torch.equal(
                result["point_inliers"], torch.tensor([True, False, False])
            )
        )
        self.assertTrue(torch.isfinite(result["loss_per_pixel"]))

    def test_two_rounds_select_midpoint_scale_gauge_once(self):
        view = _filter_view()
        with patch(
            "sfm.optimizer._select_scale_gauge", wraps=_select_scale_gauge
        ) as select:
            optimize_view_two_rounds(
                view,
                [0],
                bearing_iterations=0,
                first_iterations=0,
                second_iterations=0,
                scale_gauge_camera=1,
            )

        self.assertEqual(select.call_count, 1)
        self.assertEqual(select.call_args.kwargs["camera"], 1)

    def test_filtered_midpoint_camera_cannot_drop_the_persistent_gauge(self):
        view = _filter_view()
        first_filter = {
            "observation_inliers": view["ii"] != 1,
        }

        with patch(
            "sfm.optimizer._select_scale_gauge", wraps=_select_scale_gauge
        ) as select, patch(
            "sfm.optimizer.filter_reprojection_observations",
            return_value=first_filter,
        ) as filtering:
            with self.assertRaisesRegex(
                RuntimeError,
                "persistent scale gauge camera has no active observations",
            ):
                optimize_view_two_rounds(
                    view,
                    [0],
                    bearing_iterations=0,
                    first_iterations=0,
                    second_iterations=0,
                    scale_gauge_camera=1,
                )

        self.assertEqual(select.call_count, 1)
        filtering.assert_called_once()

    def test_filter_rejects_track_when_owner_observation_is_an_outlier(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(4, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, -0.2, -0.4, -0.6], dtype=dtype)
        points = torch.tensor(
            [[0.1, 0.0, 4.0], [-0.2, 0.1, 5.0]], dtype=dtype
        )
        K = torch.tensor(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).repeat(4, 1, 1)
        ii = torch.arange(4).repeat_interleave(2)
        jj = torch.arange(2).repeat(4)
        distortion = torch.zeros(4, 4, dtype=dtype)
        uv, _ = _project(poses, points, K, distortion, ii, jj)
        uv[1, 0] += 5.0
        view = {
            "poses": poses,
            "points": points,
            "references": torch.tensor([0, 0]),
            "ii": ii,
            "jj": jj,
            "uv": uv,
            "weight": torch.ones(ii.numel(), dtype=dtype),
            "K": K,
            "distortion": distortion,
        }

        result = filter_reprojection_observations(
            view, poses, points, min_observations=3
        )

        self.assertTrue(
            torch.equal(result["point_inliers"], torch.tensor([True, False]))
        )
        self.assertTrue(
            torch.equal(
                result["observation_inliers"],
                torch.tensor(
                    [True, False, True, False, True, False, True, False]
                ),
            )
        )

    def test_existing_matrix_free_optimizer_still_reduces_pixel_error(self):
        torch.manual_seed(2)
        dtype = torch.float64
        cameras, point_count = 3, 12
        truth = torch.eye(4, dtype=dtype).repeat(cameras, 1, 1)
        truth[:, 0, 3] = torch.tensor([0.0, -0.4, -0.8], dtype=dtype)
        points = torch.randn(point_count, 3, dtype=dtype)
        points *= torch.tensor([0.4, 0.3, 0.2], dtype=dtype)
        points += torch.tensor([0.0, 0.0, 4.0], dtype=dtype)
        ii = torch.arange(cameras).repeat_interleave(point_count)
        jj = torch.arange(point_count).repeat(cameras)
        K = torch.tensor(
            [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).repeat(cameras, 1, 1)
        distortion = torch.zeros(cameras, 4, dtype=dtype)
        target, _ = _project(truth, points, K, distortion, ii, jj)
        initial_poses = truth.clone()
        initial_poses[1:, 0, 3] += torch.tensor([0.08, -0.07], dtype=dtype)
        initial_points = points + 0.05 * torch.randn_like(points)
        before = (
            _project(initial_poses, initial_points, K, distortion, ii, jj)[0]
            - target
        ).norm(dim=-1).mean()
        result = optimize_view(
            {
                "poses": initial_poses,
                "points": initial_points,
                "ii": ii,
                "jj": jj,
                "uv": target,
                "weight": torch.ones(ii.numel(), dtype=dtype),
                "K": K,
                "distortion": distortion,
            },
            [0],
            4,
        )
        after = (
            _project(result["poses"], result["points"], K, distortion, ii, jj)[0]
            - target
        ).norm(dim=-1).mean()

        self.assertLess(float(after), float(before) * 1.0e-3)
        self.assertTrue(torch.isfinite(result["poses"]).all())


class DenseAlignmentTest(unittest.TestCase):
    def test_subpixel_sampling_is_bilinear(self):
        yy, xx = torch.meshgrid(
            torch.arange(5, dtype=torch.float64),
            torch.arange(6, dtype=torch.float64),
            indexing="ij",
        )
        value_map = 2.0 + 0.3 * xx + 0.7 * yy
        points = torch.tensor([[1.25, 2.5], [3.75, 1.5]], dtype=torch.float64)

        sampled = sample_map_at_points(value_map, points)[:, 0]
        expected = 2.0 + 0.3 * points[:, 0] + 0.7 * points[:, 1]

        self.assertTrue(torch.allclose(sampled, expected, atol=1.0e-12))

    def test_disparity_affine_ransac_recovers_scale_and_shift(self):
        torch.manual_seed(4)
        predicted = torch.linspace(2.0, 6.0, 16, dtype=torch.float64)
        expected_scale = predicted.new_tensor(1.7)
        expected_shift = predicted.new_tensor(0.08)
        sparse = (expected_scale / predicted + expected_shift).reciprocal()
        sparse[0] *= 1.8

        scale, shift, inliers = fit_disparity_affine(
            predicted, sparse, torch.ones_like(predicted)
        )

        self.assertTrue(torch.allclose(scale, expected_scale, atol=1.0e-10))
        self.assertTrue(torch.allclose(shift, expected_shift, atol=1.0e-10))
        self.assertFalse(bool(inliers[0]))
        self.assertEqual(int(inliers.sum()), 15)
        with self.assertRaisesRegex(RuntimeError, "at least 8"):
            fit_disparity_affine(
                predicted[:7], sparse[:7], torch.ones_like(predicted[:7])
            )

    def test_reconstruct_uses_frame_observations_and_keeps_export_fields(self):
        torch.manual_seed(7)
        dtype = torch.float64
        height, width = 20, 20
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=dtype),
            torch.arange(width, dtype=dtype),
            indexing="ij",
        )
        predicted_depth = 4.0 + 0.005 * xx + 0.003 * yy
        scale, shift = 1.4, 0.07
        uv = torch.tensor(
            [
                [2.25, 2.50],
                [4.50, 3.25],
                [7.75, 4.50],
                [10.25, 6.75],
                [13.50, 8.25],
                [16.25, 10.50],
                [3.75, 13.25],
                [6.50, 16.25],
                [11.75, 14.50],
                [15.50, 16.75],
            ],
            dtype=dtype,
        )
        sampled = sample_map_at_points(predicted_depth, uv)[:, 0]
        sparse_depth = (scale / sampled + shift).reciprocal()
        K = torch.tensor(
            [[30.0, 0.0, 9.5], [0.0, 30.0, 9.5], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        uv1 = torch.cat((uv, torch.ones_like(uv[:, :1])), -1)
        rays = torch.einsum("ij,pj->pi", torch.linalg.inv(K), uv1)
        points = rays * sparse_depth[:, None]
        point_count = points.shape[0]
        frame_count = 3
        observation_frames = torch.arange(frame_count).repeat_interleave(point_count)
        point_ids = torch.arange(point_count).repeat(frame_count)
        observation_uv = uv.repeat(frame_count, 1)
        graph = SimpleNamespace(
            poses={i: torch.eye(4, dtype=dtype) for i in range(frame_count)},
            intrinsics={i: K.clone() for i in range(frame_count)},
            point_count=point_count,
            point_initialized=torch.ones(point_count, dtype=torch.bool),
            point_positions=points.clone(),
            point_references=torch.zeros(point_count, dtype=torch.long),
            observations=[
                (
                    0,
                    observation_frames,
                    point_ids,
                    observation_uv,
                    torch.ones(observation_frames.numel(), dtype=dtype),
                )
            ],
            observation_ids=[torch.arange(observation_frames.numel())],
            inactive_observation_ids={observation_frames.numel() - 1},
        )
        image = torch.stack(
            (xx / width, yy / height, torch.full_like(xx, 0.5)), dim=0
        ).float()
        frames = SimpleNamespace(
            keyframes=set(range(frame_count)),
            dense={
                i: (
                    image.clone(),
                    torch.stack(
                        (
                            torch.zeros_like(predicted_depth),
                            torch.zeros_like(predicted_depth),
                            predicted_depth / float(i + 1),
                        ),
                        -1,
                    ),
                    torch.ones_like(predicted_depth),
                    float(i + 1),
                )
                for i in range(frame_count)
            },
        )

        with patch("builtins.print") as output:
            result = reconstruct(graph, frames)

        self.assertEqual(
            [call.args[0] for call in output.call_args_list],
            [
                "dense reconstruction frame=0 relative_scale=1 "
                "final_disparity_scale=1.4",
                "dense reconstruction frame=1 relative_scale=2 "
                "final_disparity_scale=0.7",
                "dense reconstruction frame=2 relative_scale=3 "
                "final_disparity_scale=0.466667",
            ],
        )

        self.assertEqual(
            set(result),
            {
                "sparse_points",
                "sparse_point_ids",
                "sparse_frame_ids",
                "sparse_obs_cnt",
                "sparse_inliers",
                "dense_points",
                "dense_colors",
                "dense_frame_ids",
            },
        )
        self.assertTrue(bool(result["sparse_inliers"].all()))
        self.assertTrue(bool((result["sparse_obs_cnt"][:-1] == 3).all()))
        self.assertEqual(int(result["sparse_obs_cnt"][-1]), 2)
        self.assertEqual(result["dense_points"].shape[1], 3)
        self.assertEqual(result["dense_points"].shape[0], result["dense_colors"].shape[0])
        self.assertEqual(set(result["dense_frame_ids"].tolist()), {0, 1, 2})
        median_depth = [
            result["dense_points"][result["dense_frame_ids"] == i, 2].median()
            for i in range(frame_count)
        ]
        expected_median_depth = (
            scale / predicted_depth + shift
        ).reciprocal().median()
        for actual in median_depth:
            self.assertTrue(
                torch.allclose(actual, expected_median_depth, atol=1.0e-10)
            )
        dense_depth = [
            result["dense_points"][result["dense_frame_ids"] == i, 2]
            for i in range(frame_count)
        ]
        self.assertTrue(torch.allclose(dense_depth[0], dense_depth[1], atol=1.0e-10))
        self.assertTrue(torch.allclose(dense_depth[0], dense_depth[2], atol=1.0e-10))


if __name__ == "__main__":
    unittest.main()
