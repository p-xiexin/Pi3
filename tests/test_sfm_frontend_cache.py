import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch

from sfm.frontend_cache import (
    _save_payload,
    load_frontend_cache,
    load_loop_cache,
    packet_to_device,
    restore_frames,
    save_frontend_cache,
    save_loop_cache,
)


class FrontendCacheTest(unittest.TestCase):
    def test_pre_geometry_cache_requires_both_stage_files_to_be_deleted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data.h5"
            _save_payload(path, {"packets": [], "frames": {}})
            with self.assertRaisesRegex(
                RuntimeError, "delete data.h5 and data_loop.h5"
            ):
                load_frontend_cache(path)

    def test_pre_geometry_loop_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data_loop.h5"
            _save_payload(path, {"packets": []})
            with self.assertRaisesRegex(
                RuntimeError, "delete data.h5 and data_loop.h5"
            ):
                load_loop_cache(path)

    def test_round_trip_uses_one_payload_and_restores_frame_store(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data.h5"
            packet = {
                "kind": "sliding",
                "frame_ids": torch.tensor([0, 1, 2], device="cpu"),
                "poses": torch.eye(4).repeat(3, 1, 1),
                "pi3_T_WCs": torch.eye(4).repeat(3, 1, 1),
                "metric_scale": torch.tensor(1.25),
                "parts": [{
                    "reference": 0,
                    "keys": torch.tensor([7]),
                    "track_ids": torch.tensor([0]),
                }],
                "edges": [(0, 1, torch.eye(4), 2.0)],
                "visualization": [{
                    "reference": 0,
                    "frame_ids": torch.tensor([0, 1, 2]),
                    "query_points": torch.tensor([[0.25, 0.5]]),
                    "raw_tracks": torch.tensor([
                        [[0.25, 0.5]],
                        [[0.5, 0.5]],
                        [[0.75, 0.5]],
                    ]),
                    "frontend_valid": torch.tensor([[True], [False], [True]]),
                    "visualization_confidence": torch.tensor(
                        [[1.0], [0.25], [0.75]]
                    ),
                    "visualization_score": torch.tensor(
                        [[1.0], [0.4], [0.8]]
                    ),
                    "visualization_confidence_label": "vgg visible=color score=alpha",
                }],
            }
            frames = SimpleNamespace(
                keyframes={0},
                dense={0: (
                    torch.ones(3, 2, 2),
                    torch.ones(2, 2),
                    torch.ones(2, 2),
                    1.25,
                )},
                anchors={0: tuple(torch.ones(1) for _ in range(4))},
                track_ids={0: torch.tensor([0])},
                next_track_id=1,
            )

            save_frontend_cache(path, [packet], frames)
            loaded = load_frontend_cache(path)

            self.assertTrue(torch.equal(loaded["packets"][0]["frame_ids"], packet["frame_ids"]))
            self.assertTrue(torch.equal(
                loaded["packets"][0]["pi3_T_WCs"], packet["pi3_T_WCs"]
            ))
            self.assertEqual(float(loaded["packets"][0]["metric_scale"]), 1.25)
            diagnostic = loaded["packets"][0]["visualization"][0]
            self.assertTrue(torch.equal(
                diagnostic["raw_tracks"], packet["visualization"][0]["raw_tracks"]
            ))
            self.assertTrue(torch.equal(
                diagnostic["frontend_valid"],
                packet["visualization"][0]["frontend_valid"],
            ))
            self.assertTrue(torch.equal(
                diagnostic["visualization_confidence"],
                packet["visualization"][0]["visualization_confidence"],
            ))
            self.assertTrue(torch.equal(
                diagnostic["visualization_score"],
                packet["visualization"][0]["visualization_score"],
            ))
            self.assertEqual(
                diagnostic["visualization_confidence_label"],
                "vgg visible=color score=alpha",
            )
            with h5py.File(path, "r") as handle:
                self.assertEqual(list(handle.keys()), ["payload"])
            restored = SimpleNamespace(keyframes=set(), dense={}, anchors={})
            restore_frames(restored, loaded["frames"])
            self.assertEqual(restored.keyframes, {0})
            self.assertTrue(torch.equal(restored.dense[0][1], frames.dense[0][1]))
            self.assertEqual(restored.dense[0][1].ndim, 2)
            self.assertEqual(restored.dense[0][3], 1.25)
            self.assertTrue(torch.equal(restored.track_ids[0], torch.tensor([0])))
            self.assertEqual(restored.next_track_id, 1)
            moved = packet_to_device(loaded["packets"][0], "cpu")
            self.assertTrue(torch.equal(moved["edges"][0][2], torch.eye(4)))
            self.assertFalse(path.with_suffix(".h5.tmp").exists())

    def test_loop_cache_contains_only_incremental_loop_packets(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data_loop.h5"
            packet = {
                "kind": "loop",
                "frame_ids": torch.tensor([2, 9]),
                "parts": [{
                    "reference": 2,
                    "track_ids": torch.tensor([4]),
                    "obs_uv": torch.ones(3, 2),
                }],
                "edges": [(2, 9, torch.eye(4), 3.0)],
            }

            save_loop_cache(path, [packet])
            loaded = load_loop_cache(path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["kind"], "loop")
            self.assertTrue(torch.equal(loaded[0]["frame_ids"], packet["frame_ids"]))
            with h5py.File(path, "r") as handle:
                self.assertEqual(list(handle.keys()), ["payload"])
            self.assertFalse(path.with_suffix(".h5.tmp").exists())


if __name__ == "__main__":
    unittest.main()
