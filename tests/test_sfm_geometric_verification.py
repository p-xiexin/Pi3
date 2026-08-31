import unittest
from unittest.mock import patch

import numpy as np
import torch

from sfm.geometric_verification import (
    VerifiedTracks,
    _estimate_relative_pose,
    triangulate_tracks,
    validate_landmarks,
    verify_packet,
)
from sfm.optimizer import _project


def _relative_pose(source_points, target_points, source_K, target_K):
    translation = source_points.new_tensor([1.0, 0.0, 0.0])
    return (
        torch.eye(3, device=source_points.device, dtype=source_points.dtype),
        translation,
        torch.ones(source_points.shape[0], device=source_points.device, dtype=torch.bool),
    )


def _two_frame_packet(kind="loop", include_track_ids=True):
    point_count = 4
    frame_ids = torch.tensor([10, 90])
    query = torch.stack(
        (torch.arange(point_count, dtype=torch.float32), torch.ones(point_count)), dim=-1
    )
    target = query.clone()
    target[:, 0] -= 0.1
    part = {
        "reference": 10,
        "keys": torch.arange(point_count),
        "anchors": torch.cat((query, torch.ones(point_count, 1)), dim=-1),
        "obs_frames": frame_ids.repeat_interleave(point_count),
        "obs_points": torch.arange(point_count).repeat(2),
        "obs_uv": torch.cat((query, target)),
        "obs_weights": torch.ones(point_count * 2),
        "edge_source": [10],
        "edge_target": [90],
        "edge_weight": [point_count],
    }
    if include_track_ids:
        part["track_ids"] = torch.tensor([101, 205, 999, 4000])
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 2, 3] = 7.0
    K = torch.tensor(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    ).repeat(2, 1, 1)
    return {
        "kind": kind,
        "frame_ids": frame_ids,
        "keyframes": torch.tensor([10]),
        "poses": poses,
        "K": K,
        "parts": [part],
        "edges": [(10, 90, torch.eye(4), point_count)],
    }


class GeometricVerificationTest(unittest.TestCase):
    def test_calibrated_ransac_uses_one_pixel_normalized_threshold(self):
        class Cv2Stub:
            RANSAC = 8

            def __init__(self):
                self.find_call = None

            def findEssentialMat(
                self, source, target, K, method, prob, threshold
            ):
                self.find_call = source, target, K, method, prob, threshold
                return np.eye(3), np.ones((source.shape[0], 1), dtype=np.uint8)

            def recoverPose(self, essential, source, target, K, mask):
                return (
                    source.shape[0],
                    np.eye(3),
                    np.array([[1.0], [0.0], [0.0]]),
                    mask,
                )

        point_count = 16
        source = torch.stack(
            (torch.arange(point_count, dtype=torch.float64) + 20, torch.ones(point_count) * 30),
            dim=-1,
        )
        target = source.clone()
        target[:, 0] -= 1
        source_K = torch.tensor(
            [[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        target_K = torch.tensor(
            [[200.0, 0.0, 20.0], [0.0, 200.0, 30.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        cv2_stub = Cv2Stub()

        result = _estimate_relative_pose(
            source, target, source_K, target_K, cv2_module=cv2_stub
        )

        self.assertIsNotNone(result)
        normalized_source, _, camera_matrix, method, probability, threshold = (
            cv2_stub.find_call
        )
        self.assertTrue(np.allclose(normalized_source[0], [0.0, 0.0]))
        self.assertTrue(np.array_equal(camera_matrix, np.eye(3)))
        self.assertEqual(method, cv2_stub.RANSAC)
        self.assertEqual(probability, 0.999)
        self.assertAlmostEqual(threshold, 1.0 / 150.0)

    def test_loop_packet_accepts_two_view_edge_and_preserves_track_ids(self):
        packet = _two_frame_packet()

        def reject_last(source_points, target_points, source_K, target_K):
            rotation, translation, inliers = _relative_pose(
                source_points, target_points, source_K, target_K
            )
            inliers[-1] = False
            return rotation, translation, inliers

        with (
            patch("sfm.geometric_verification.MIN_PAIR_MATCHES", 3),
            patch("sfm.geometric_verification.MIN_PAIR_INLIERS", 3),
            patch(
                "sfm.geometric_verification._estimate_relative_pose",
                side_effect=reject_last,
            ),
        ):
            verified = verify_packet(packet)

        self.assertEqual(verified["verified_track_ids"].tolist(), [101, 205, 999])
        self.assertEqual(verified["parts"][0]["track_ids"].tolist(), [101, 205, 999])
        self.assertEqual(
            verified["parts"][0]["obs_points"].tolist(), [0, 1, 2, 0, 1, 2]
        )
        self.assertEqual(verified["edges"][0][:2], (10, 90))
        self.assertEqual(int(verified["geometry_observations"]), 6)
        self.assertAlmostEqual(float(verified["geometry_inlier_ratio"]), 0.75)
        self.assertTrue(torch.equal(verified["poses"], packet["poses"]))

    def test_sliding_packet_does_not_promote_two_view_tracks(self):
        packet = _two_frame_packet(kind="sliding")
        with (
            patch("sfm.geometric_verification.MIN_PAIR_MATCHES", 3),
            patch("sfm.geometric_verification.MIN_PAIR_INLIERS", 3),
            patch(
                "sfm.geometric_verification._estimate_relative_pose",
                side_effect=_relative_pose,
            ),
            self.assertRaisesRegex(RuntimeError, "no pose edges"),
        ):
            verify_packet(packet)

    def test_packet_verification_requires_preassigned_stable_ids(self):
        packet = _two_frame_packet(include_track_ids=False)
        with self.assertRaisesRegex(RuntimeError, "stable track IDs"):
            verify_packet(packet)

    def test_owner_gate_removes_target_observations_of_rejected_tracks(self):
        packet = _two_frame_packet()
        verified_tracks = VerifiedTracks(
            mask=torch.tensor(
                [[True, False, True, False], [True, True, True, True]]
            ),
            keep=torch.ones(4, dtype=torch.bool),
            source=torch.tensor([0]),
            target=torch.tensor([1]),
            relative=torch.eye(4).unsqueeze(0),
            weight=torch.tensor([4.0]),
            pair_matches=torch.tensor([4]),
            pair_inliers=torch.tensor([4]),
        )

        with patch(
            "sfm.geometric_verification.verify_tracks",
            return_value=verified_tracks,
        ):
            verified = verify_packet(packet)

        part = verified["parts"][0]
        self.assertEqual(part["track_ids"].tolist(), [101, 999])
        self.assertEqual(part["obs_points"].tolist(), [0, 1, 0, 1])
        self.assertEqual(part["obs_frames"].tolist(), [10, 10, 90, 90])
        self.assertTrue(bool((part["obs_points"] >= 0).all()))

    def test_multiview_dlt_rejects_low_parallax(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, -0.01, -0.02], dtype=dtype)
        points = torch.tensor([[0.2, -0.1, 10.0]], dtype=dtype)
        ii = torch.arange(3)
        jj = torch.zeros(3, dtype=torch.long)
        K = torch.tensor(
            [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).repeat(3, 1, 1)
        uv, _ = _project(
            poses, points, K, torch.zeros(3, 4, dtype=dtype), ii, jj
        )

        _, observation_mask, point_mask, angles = triangulate_tracks(
            poses, K, ii, jj, uv, 1
        )

        self.assertFalse(bool(point_mask[0]))
        self.assertFalse(bool(observation_mask.any()))
        self.assertEqual(float(angles[0]), 0.0)

    def test_landmark_validation_requires_three_positive_depth_views(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, -0.5, -1.0], dtype=dtype)
        poses[2, 2, 3] = -10.0
        points = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype)
        ii = torch.arange(3)
        jj = torch.zeros(3, dtype=torch.long)

        observation_mask, point_mask, _ = validate_landmarks(
            poses, points, ii, jj
        )

        self.assertFalse(bool(point_mask[0]))
        self.assertFalse(bool(observation_mask.any()))


if __name__ == "__main__":
    unittest.main()
