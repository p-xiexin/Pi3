import sys
import types
import unittest
from unittest.mock import patch

import torch

from sfm.features import (
    ImageQueryFeatures,
    _top_score_indices,
    build_query_features,
)
from sfm.keyframes import select_keyframes_eq4_window
from sfm.tracker import FrameStore, WindowTracker


class _FakeExtractor:
    points = torch.tensor([[1.0, 1.0]])
    scores = torch.tensor([1.0])

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def extract(self, images, invalid_mask=None):
        return {
            "keypoints": self.points.to(images)[None],
            "keypoint_scores": self.scores.to(images)[None],
        }


class _FakeSuperPoint(_FakeExtractor):
    points = torch.tensor([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])
    scores = torch.tensor([0.3, 0.2, 0.1])


class _FakeSIFT(_FakeExtractor):
    points = torch.tensor([[1.3, 1.3], [63.0, 47.0]])
    scores = torch.tensor([0.4, 0.05])


def _fake_lightglue():
    module = types.ModuleType("lightglue")
    module.SIFT = _FakeSIFT
    module.SuperPoint = _FakeSuperPoint
    return module


def _geometry(frame_count, height=2, width=3):
    yy, xx = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()
    points = points.unsqueeze(0).repeat(frame_count, 1, 1, 1)
    confidence = torch.ones(frame_count, height, width)
    poses = torch.eye(4).repeat(frame_count, 1, 1)
    intrinsics = torch.eye(3).repeat(frame_count, 1, 1)
    valid_mask = torch.ones(frame_count, height, width, dtype=torch.bool)
    return points, confidence, poses, intrinsics, valid_mask


class QueryFeatureAlignmentTests(unittest.TestCase):
    def test_both_factories_use_the_same_superpoint_sift_queries(self):
        with patch.dict(sys.modules, {"lightglue": _fake_lightglue()}):
            features = {
                name: build_query_features(
                    {"tracks_model": name, "device": "cpu"}
                )
                for name in ("glob3r", "vgg")
            }
        for frontend in features.values():
            self.assertIsInstance(frontend, ImageQueryFeatures)
            self.assertEqual(frontend.max_points, 2048)
            self.assertEqual(len(frontend.extractors), 2)
            self.assertEqual(
                frontend.extractors[0].kwargs["max_num_keypoints"], 8192
            )
            self.assertEqual(
                frontend.extractors[0].kwargs["detection_threshold"], 0.005
            )
            self.assertEqual(
                frontend.extractors[1].kwargs["max_num_keypoints"], 8192
            )

    def test_global_top_scores_do_not_reserve_spatial_grid_cells(self):
        points = torch.tensor(
            [[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [79.0, 59.0]]
        )
        scores = torch.tensor([0.1, 0.3, 0.2, 0.05])
        selected = _top_score_indices(points, scores, max_points=2)
        self.assertEqual(selected.tolist(), [1, 2])

    def test_shared_queries_merge_detectors_before_global_topk(self):
        with patch.dict(sys.modules, {"lightglue": _fake_lightglue()}):
            features = ImageQueryFeatures(max_points=3, device="cpu")
            queries = features.extract(torch.zeros(3, 48, 64))
        self.assertEqual(queries.shape, (3, 2))
        self.assertTrue(torch.equal(queries[0], torch.tensor([1.3, 1.3])))
        self.assertTrue(torch.equal(queries[1], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(queries[2], torch.tensor([1.1, 1.1])))


class FrontendMaskAlignmentTests(unittest.TestCase):
    def test_both_frontends_use_the_same_pi3_mask_everywhere(self):
        confidence = torch.tensor(
            [
                [[0.8, 0.2], [0.8, 0.8]],
                [[0.2, 0.8], [0.8, 0.8]],
                [[0.8, 0.8], [0.8, 0.8]],
            ]
        )
        vgg_mask = confidence > 0.3

        class Dataset:
            paths = ["0", "1", "2"]
            K = torch.eye(3)
            valid_mask = None

            @staticmethod
            def read(frame_ids):
                return torch.ones(len(frame_ids), 3, 2, 2)

        class Geometry:
            def eval(self):
                return self

            def infer_window(self, images):
                yy, xx = torch.meshgrid(
                    torch.arange(2), torch.arange(2), indexing="ij"
                )
                points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()
                return {
                    "local_points": points[None, None].repeat(1, 3, 1, 1, 1),
                    "camera_poses": torch.eye(4).repeat(1, 3, 1, 1),
                    "conf": torch.logit(confidence)[None, ..., None],
                }, "unmasked_matching_state"

        class Features:
            def extract(self, image, valid_mask=None):
                self.image = image.detach().clone()
                self.valid_mask = valid_mask.detach().clone()
                yy, xx = torch.meshgrid(
                    torch.arange(2), torch.arange(2), indexing="ij"
                )
                queries = torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(image)
                return queries[valid_mask.reshape(-1)]

        class Tracks:
            def prepare_window(self, images, _state):
                self.prepared = images.detach().clone()
                self.state = _state
                self.frame_count = images.shape[1]

            def track(self, _reference, queries):
                return {
                    "tracks": queries[None].repeat(self.frame_count, 1, 1),
                    "confidence": queries.new_ones(self.frame_count, queries.shape[0]),
                }

        def run(frontend):
            dataset = Dataset()
            frames = FrameStore(dataset)
            features = Features()
            tracks = Tracks()
            tracker = WindowTracker(
                Geometry(),
                tracks,
                features,
                frames,
                dataset,
                {"device": "cpu", "tracks_model": frontend},
            )
            with patch(
                "sfm.tracker.select_keyframes_eq4_window",
                return_value=torch.tensor([0]),
            ) as selection, patch(
                "sfm.tracker.verify_packet", side_effect=lambda packet, **_: packet
            ):
                packet = tracker.track([0, 1, 2])
            return types.SimpleNamespace(
                packet=packet,
                frames=frames,
                features=features,
                tracks=tracks,
                selection_mask=selection.call_args.args[5],
                threshold=tracker.pi3_mask_confidence_threshold,
            )

        for result in (run("vgg"), run("glob3r")):
            self.assertEqual(result.threshold, 0.3)
            self.assertTrue(
                torch.equal(result.features.valid_mask, vgg_mask[0])
            )
            self.assertTrue(torch.equal(result.selection_mask, vgg_mask))
            self.assertEqual(result.frames.anchors[0][0].numel(), 3)
            self.assertEqual(
                int((result.packet["parts"][0]["obs_frames"] == 1).sum()),
                2,
            )
            target = result.packet["parts"][0]["obs_uv"][
                result.packet["parts"][0]["obs_frames"] == 1
            ]
            self.assertTrue(
                torch.equal(target, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
            )
            self.assertTrue(
                torch.equal(
                    result.tracks.prepared[0, 0, :, 0, 1], torch.zeros(3)
                )
            )
            self.assertEqual(float(result.frames.dense[0][2][0, 1]), 0.0)
            self.assertIsNone(result.tracks.state)


class Eq4WindowAlignmentTests(unittest.TestCase):
    def test_identity_geometry_promotes_only_at_five_frame_interval(self):
        geometry = _geometry(7)
        selected = select_keyframes_eq4_window(range(7), *geometry)
        self.assertEqual(selected.tolist(), [0, 5])

    def test_low_projection_coverage_promotes_target(self):
        points, confidence, poses, intrinsics, valid_mask = _geometry(3)
        poses[1, 0, 3] = 10.0
        selected = select_keyframes_eq4_window(
            range(3), points, confidence, poses, intrinsics, valid_mask
        )
        self.assertEqual(selected.tolist(), [0, 1])

    def test_default_seventy_percent_threshold_rejects_sixty_percent_coverage(self):
        points, confidence, poses, intrinsics, valid_mask = _geometry(
            2, height=1, width=10
        )
        valid_mask[0, 0, 6:] = False

        selected_default = select_keyframes_eq4_window(
            range(2), points, confidence, poses, intrinsics, valid_mask
        )
        selected_half = select_keyframes_eq4_window(
            range(2),
            points,
            confidence,
            poses,
            intrinsics,
            valid_mask,
            projection_threshold=0.5,
        )

        self.assertEqual(selected_default.tolist(), [0, 1])
        self.assertEqual(selected_half.tolist(), [0])

    def test_existing_keyframes_define_processed_prefix(self):
        geometry = _geometry(8)
        selected = select_keyframes_eq4_window(
            range(8, 16), *geometry, existing_keyframes={10, 15}
        )
        self.assertEqual(selected.tolist(), [2, 7])

    def test_selection_resumes_forward_from_latest_existing_keyframe(self):
        geometry = _geometry(8)
        selected = select_keyframes_eq4_window(
            range(8, 16), *geometry, existing_keyframes={10}
        )
        self.assertEqual(selected.tolist(), [2, 7])

    def test_frame_ids_must_be_chronological(self):
        geometry = _geometry(3)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_keyframes_eq4_window([0, 2, 1], *geometry)


if __name__ == "__main__":
    unittest.main()
