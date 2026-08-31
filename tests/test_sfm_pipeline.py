import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sfm.main import run
from sfm.pipeline import CachedFrontend, FrontendPackets, GraphBackend


class FrontendPacketsTest(unittest.TestCase):
    def test_consume_transfers_packet_ownership(self):
        sliding = {"kind": "sliding"}
        loop = {"kind": "loop"}
        packets = FrontendPackets([sliding], [loop])

        self.assertEqual(packets.consume(), [sliding, loop])
        self.assertEqual(packets.sliding, [])
        self.assertEqual(packets.loop, [])


class PipelineBoundaryTest(unittest.TestCase):
    def _frontend(self, root, loop=True):
        config = {
            "data_h5": str(Path(root) / "data.h5"),
            "data_loop_h5": str(Path(root) / "data_loop.h5"),
            "window_size": 20,
            "device": "cpu",
            "loop": loop,
        }
        frames = SimpleNamespace(keyframes=set(), dense={}, anchors={})
        return CachedFrontend(config, object(), frames)

    def test_cache_misses_render_before_committing_each_h5(self):
        with tempfile.TemporaryDirectory() as root:
            frontend = self._frontend(root)
            sliding, loops = [{"kind": "sliding"}], [{"kind": "loop"}]
            events = []
            with (
                patch.object(frontend, "_build_sliding_packets", return_value=sliding),
                patch.object(frontend, "_detect_loop_windows", return_value=([[0, 1]], set())),
                patch.object(frontend, "_track_loop_windows", return_value=loops),
                patch(
                    "sfm.pipeline.save_match_images",
                    side_effect=lambda *args: events.append("draw sliding") or [],
                ),
                patch(
                    "sfm.pipeline.save_frontend_cache",
                    side_effect=lambda *args: events.append("save sliding"),
                ),
                patch(
                    "sfm.pipeline.save_loop_match_images",
                    side_effect=lambda *args: events.append("draw loop") or [],
                ),
                patch(
                    "sfm.pipeline.save_loop_cache",
                    side_effect=lambda *args: events.append("save loop"),
                ),
            ):
                cached_sliding, rebuilt = frontend._sliding_packets()
                cached_loop = frontend._loop_packets(cached_sliding, rebuilt)

            self.assertIs(cached_sliding, sliding)
            self.assertIs(cached_loop, loops)
            self.assertEqual(
                events,
                ["draw sliding", "save sliding", "draw loop", "save loop"],
            )

    def test_sliding_frontend_reports_each_chunk_relative_scale(self):
        with tempfile.TemporaryDirectory() as root:
            config = {
                "data_h5": str(Path(root) / "data.h5"),
                "data_loop_h5": str(Path(root) / "data_loop.h5"),
                "window_size": 2,
            }
            dataset = SimpleNamespace(windows=lambda _: [[0, 1], [1, 2]])
            frames = SimpleNamespace(keyframes=set(), dense={}, anchors={})
            frontend = CachedFrontend(config, dataset, frames)
            chunk_scales = iter((1.0, 1.75))

            def track(window_ids):
                frames.dense[window_ids[-1]] = (None, None, None, next(chunk_scales))
                return {"keyframes": torch.tensor([window_ids[0]])}

            tracker = SimpleNamespace(track=track)
            with (
                patch.object(frontend, "_tracker", return_value=tracker),
                patch("builtins.print") as output,
            ):
                frontend._build_sliding_packets()

            self.assertEqual(
                [call.args[0] for call in output.call_args_list],
                [
                    "frontend 1/2 frames=0..1 keyframes=[0] relative_scale=1",
                    "frontend 2/2 frames=1..2 keyframes=[1] relative_scale=1.75",
                ],
            )

    def test_cache_hits_do_not_render_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            frontend = self._frontend(root)
            frontend.cache_path.touch()
            frontend.loop_cache_path.touch()
            sliding, loops = [{"visualization": [object()]}], [{"kind": "loop"}]
            with (
                patch(
                    "sfm.pipeline.load_frontend_cache",
                    return_value={"packets": sliding, "frames": {}},
                ),
                patch("sfm.pipeline.restore_frames"),
                patch("sfm.pipeline.load_loop_cache", return_value=loops),
                patch("sfm.pipeline.save_match_images") as save_sliding,
                patch("sfm.pipeline.save_loop_match_images") as save_loop,
            ):
                cached_sliding, rebuilt = frontend._sliding_packets()
                cached_loop = frontend._loop_packets(cached_sliding, rebuilt)

            self.assertFalse(rebuilt)
            self.assertIs(cached_sliding, sliding)
            self.assertIs(cached_loop, loops)
            save_sliding.assert_not_called()
            save_loop.assert_not_called()

    def test_empty_loop_windows_do_not_load_tracking_models(self):
        config = {
            "data_h5": "data.h5",
            "data_loop_h5": "data_loop.h5",
        }
        frontend = CachedFrontend(config, object(), object())
        with patch.object(frontend, "_tracker") as tracker:
            self.assertEqual(frontend._track_loop_windows([]), [])
        tracker.assert_not_called()

    def test_graph_backend_rejects_an_empty_frontend(self):
        backend = GraphBackend({"device": "cpu"}, object(), None, ".")
        with self.assertRaisesRegex(RuntimeError, "no factor packets"):
            backend._build_graph(FrontendPackets([], []))

    def test_main_delegates_to_the_pipeline_composition_root(self):
        config = {"random_seed": 3}
        with patch("sfm.main.SfMPipeline") as pipeline:
            pipeline.return_value.run.return_value = {"loss": 1.0}
            self.assertEqual(run(config), {"loss": 1.0})
        pipeline.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
