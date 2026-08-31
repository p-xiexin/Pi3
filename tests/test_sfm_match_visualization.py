import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image
import torch

from sfm.match_visualization import (
    _matrix_groups,
    _raw_masks,
    _row_label,
    _sparse_confidence_panel,
    _track_color,
    save_match_images,
)


def _verified_packet():
    return {
        "kind": "sliding",
        "frame_ids": torch.tensor([0, 1]),
        "parts": [{
            "reference": 0,
            "track_ids": torch.tensor([10]),
            "obs_frames": torch.tensor([0, 1]),
            "obs_points": torch.tensor([0, 0]),
            "obs_uv": torch.tensor([[1.0, 2.5], [2.0, 2.5]]),
            "obs_weights": torch.tensor([1.0, 0.9]),
        }],
    }


class MatchVisualizationTest(unittest.TestCase):
    def _frames(self):
        depth = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        confidence = torch.full((4, 4), 0.75)
        return SimpleNamespace(
            dense={
                0: (torch.zeros(3, 4, 4), depth, confidence),
                1: (torch.zeros(3, 4, 4), depth, confidence),
            }
        )

    def _dataset(self):
        return SimpleNamespace(
            read=lambda frame_ids: torch.zeros(len(frame_ids), 3, 4, 4)
        )

    def test_old_packet_without_raw_payload_keeps_matrix_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            paths = save_match_images(
                root,
                [_verified_packet()],
                self._frames(),
                self._dataset(),
                cell_width=4,
            )
            self.assertEqual([path.name for path in paths], ["reference_0000.png"])
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, (20, 8))
        self.assertEqual(
            _row_label(0, 1, None, None, 7, 5),
            "reference 0000  target 0001  shown 5 / valid 7",
        )

    def test_raw_payload_draws_gray_before_verified_color_and_labels_stages(self):
        packet = _verified_packet()
        packet["visualization"] = [{
            "reference": 0,
            "frame_ids": torch.tensor([0, 1]),
            "query_points": torch.tensor(
                [[1.0, 2.5], [3.0, 3.0], [float("nan"), 0.0]]
            ),
            "raw_tracks": torch.tensor([
                [[1.0, 2.5], [3.0, 3.0], [0.0, 0.0]],
                [[2.0, 2.5], [3.0, 3.0], [-1.0, 0.0]],
            ]),
            "frontend_valid": torch.tensor([
                [True, True, True],
                [True, False, False],
            ]),
            "visualization_confidence": torch.tensor([
                [1.0, 0.5, 0.0],
                [0.9, 0.2, 0.0],
            ]),
            "visualization_confidence_label": "vgg visibility",
        }]

        groups = _matrix_groups([packet, packet])
        self.assertEqual(set(groups[0]["raw"]), {0, 1})
        raw = groups[0]["raw"][1]
        query_mask, target_mask, frontend_mask = _raw_masks(raw, 4, 4)
        self.assertEqual(int(query_mask.sum()), 2)
        self.assertEqual(int(target_mask.sum()), 2)
        self.assertEqual(int(frontend_mask.sum()), 1)
        self.assertEqual(
            _row_label(0, 1, 2, 1, 1, 1),
            "reference 0000  target 0001  raw 2  frontend 1  "
            "geometry 1  shown 1",
        )

        with tempfile.TemporaryDirectory() as root:
            path = save_match_images(
                root,
                [packet, packet],
                self._frames(),
                self._dataset(),
                cell_width=40,
            )[0]
            with Image.open(path) as rendered:
                # The second row is target frame 1.  A verified point covers the
                # gray raw point, while an unverified raw point remains gray.
                verified = rendered.getpixel((13, 40 + 32))
                raw_query = rendered.getpixel((39, 40 + 39))
                raw_target = rendered.getpixel((40 + 39, 40 + 39))
                vgg_confidence = rendered.getpixel((2 * 40 + 39, 40 + 39))
                pi3_depth = rendered.getpixel((3 * 40 + 20, 40 + 30))
                pi3_confidence = rendered.getpixel((4 * 40 + 20, 40 + 30))
        self.assertEqual(verified, _track_color(10))
        self.assertEqual(raw_query[0], raw_query[1])
        self.assertEqual(raw_query[1], raw_query[2])
        self.assertGreater(raw_query[0], 0)
        self.assertEqual(raw_target[0], raw_target[1])
        self.assertEqual(raw_target[1], raw_target[2])
        self.assertGreater(raw_target[0], 0)
        self.assertNotEqual(vgg_confidence, (0, 0, 0))
        self.assertNotEqual(pi3_depth, (0, 0, 0))
        self.assertNotEqual(pi3_confidence, (0, 0, 0))

    def test_vgg_panel_uses_visibility_for_color_and_score_for_alpha(self):
        panel = _sparse_confidence_panel(
            torch.tensor([[0.5, 0.5], [2.5, 0.5], [0.5, 2.5], [2.5, 2.5]]),
            torch.tensor([0.5, 0.5, 0.1, 0.9]),
            source_width=4,
            source_height=4,
            width=40,
            height=40,
            opacity=torch.tensor([0.2, 0.8, 0.8, 0.8]),
        )
        low_score = panel.getpixel((7, 7))
        high_score = panel.getpixel((32, 7))
        low_visible = panel.getpixel((7, 32))
        high_visible = panel.getpixel((32, 32))
        self.assertGreater(sum(high_score), sum(low_score))
        self.assertNotEqual(low_visible, high_visible)

    def test_glob3r_dense_confidence_occupies_the_third_column(self):
        packet = _verified_packet()
        packet["frontend"] = "glob3r"
        packet["visualization"] = [{
            "frontend": "glob3r",
            "reference": 0,
            "frame_ids": torch.tensor([0, 1]),
            "query_points": torch.tensor([[1.0, 2.5]]),
            "raw_tracks": torch.tensor([
                [[1.0, 2.5]],
                [[2.0, 2.5]],
            ]),
            "frontend_valid": torch.tensor([[True], [True]]),
            "visualization_confidence": torch.stack((
                torch.full((4, 4), 0.2),
                torch.full((4, 4), 0.8),
            )),
            "visualization_confidence_label": "glob3r confidence",
        }]

        with tempfile.TemporaryDirectory() as root:
            path = save_match_images(
                root,
                [packet],
                self._frames(),
                self._dataset(),
                cell_width=40,
            )[0]
            with Image.open(path) as rendered:
                self.assertEqual(rendered.size, (200, 80))
                dense_confidence = rendered.getpixel((2 * 40 + 20, 40 + 30))
        self.assertNotEqual(dense_confidence, (0, 0, 0))

    def test_duplicate_pair_prefers_one_more_complete_frontend_result(self):
        first = _verified_packet()
        first["visualization"] = [{
            "reference": 0,
            "frame_ids": torch.tensor([0, 1]),
            "query_points": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "raw_tracks": torch.tensor([
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [float("nan"), 1.0]],
            ]),
            "frontend_valid": torch.tensor([[True, True], [True, False]]),
        }]
        second = _verified_packet()
        second["visualization"] = [{
            "reference": 0,
            "frame_ids": torch.tensor([0, 1]),
            "query_points": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "raw_tracks": torch.tensor([
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]),
            "frontend_valid": torch.tensor([[True, True], [True, True]]),
        }]

        raw = _matrix_groups([first, second])[0]["raw"][1]
        self.assertEqual(int(raw["frontend_valid"].sum()), 2)
        self.assertEqual(raw["target_points"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
