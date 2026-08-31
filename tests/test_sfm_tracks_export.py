import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import torch

from sfm.tracks_export import save_ba_tracks


def _read_binary_ply(path, dtype):
    with Path(path).open("rb") as stream:
        count = None
        while True:
            line = stream.readline().decode("utf-8").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line == "end_header":
                break
        if count is None:
            raise AssertionError("PLY has no vertex count")
        body = stream.read()
        if len(body) != count * dtype.itemsize:
            raise AssertionError("PLY body size does not match its declared vertex count")
        return np.frombuffer(body, dtype=dtype, count=count)


def _read_ply_header(path):
    lines = []
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline().decode("utf-8").strip()
            lines.append(line)
            if line == "end_header":
                return lines


class TracksExportTest(unittest.TestCase):
    def test_legacy_bundle_layout_ids_pixels_and_colors(self):
        image0 = torch.zeros(3, 4, 4)
        image0[:, 1, 1] = torch.tensor([0.5, 0.25, 1.0])
        image2 = torch.zeros(3, 4, 4)
        image2[:, 2, 2] = torch.tensor([0.0, 1.0, 0.5])
        frames = SimpleNamespace(
            dense={
                0: (image0, torch.empty(0), torch.empty(0)),
                2: (image2, torch.empty(0), torch.empty(0)),
            }
        )
        view = {
            "frame_ids": torch.tensor([0, 2]),
            "points": torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]),
            "references": torch.tensor([0, 1]),
            "ii": torch.tensor([0, 1, 0, 1]),
            "jj": torch.tensor([0, 0, 1, 1]),
            "uv": torch.tensor([[1.0, 1.0], [0.5, 1.5], [3.0, 0.0], [2.0, 2.0]]),
        }
        transforms = torch.tensor(
            [
                [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "p99.ply"
            stale.write_bytes(b"stale projection")
            summary = save_ba_tracks(directory, view, frames, transforms)
            root = Path(directory)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"doc.xml", "points0.ply", "tracks.ply", "p0.ply", "p2.ply"},
            )
            self.assertFalse(stale.exists())
            self.assertEqual(summary["camera_count"], 2)
            self.assertEqual(summary["point_count"], 2)
            self.assertEqual(summary["observation_count"], 4)
            self.assertEqual(
                _read_ply_header(root / "points0.ply"),
                [
                    "ply", "format binary_little_endian 1.0", "element vertex 2",
                    "property float x", "property float y", "property float z",
                    "property int id", "end_header",
                ],
            )
            self.assertEqual(
                _read_ply_header(root / "tracks.ply"),
                [
                    "ply", "format binary_little_endian 1.0", "element vertex 2",
                    "property uchar red", "property uchar green", "property uchar blue",
                    "end_header",
                ],
            )
            self.assertEqual(
                _read_ply_header(root / "p0.ply"),
                [
                    "ply", "format binary_little_endian 1.0", "element vertex 2",
                    "property float x", "property float y", "property float size",
                    "property int id", "property float z", "property uint64 keypoint",
                    "end_header",
                ],
            )

            points = _read_binary_ply(
                root / "points0.ply",
                np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("id", "<u4")]),
            )
            self.assertEqual(points["id"].tolist(), [1, 2])
            np.testing.assert_allclose(
                np.column_stack((points["x"], points["y"], points["z"])),
                view["points"].numpy(),
            )

            colors = _read_binary_ply(
                root / "tracks.ply",
                np.dtype([("red", "u1"), ("green", "u1"), ("blue", "u1")]),
            )
            self.assertEqual(
                np.column_stack((colors["red"], colors["green"], colors["blue"])).tolist(),
                [[127, 63, 255], [0, 255, 127]],
            )

            projection_dtype = np.dtype(
                [
                    ("x", "<f4"), ("y", "<f4"), ("size", "<f4"), ("id", "<u4"),
                    ("z", "<f4"), ("keypoint", "<u8"),
                ]
            )
            p0 = _read_binary_ply(root / "p0.ply", projection_dtype)
            p2 = _read_binary_ply(root / "p2.ply", projection_dtype)
            np.testing.assert_allclose(
                np.column_stack((p0["x"], p0["y"])), [[12.0, 23.0], [16.0, 20.0]]
            )
            np.testing.assert_allclose(
                np.column_stack((p2["x"], p2["y"])), [[0.5, 1.5], [2.0, 2.0]]
            )
            self.assertEqual(p0["id"].tolist(), [1, 2])
            self.assertEqual(p2["id"].tolist(), [1, 2])
            self.assertEqual(p0["size"].tolist(), [1.0, 1.0])
            self.assertEqual(p0["z"].tolist(), [-1.0, -1.0])
            self.assertEqual(p2["z"].tolist(), [-1.0, -1.0])
            self.assertEqual(p0["keypoint"].tolist(), [0, 0])
            self.assertEqual(p2["keypoint"].tolist(), [0, 0])

            document = ElementTree.parse(root / "doc.xml").getroot()
            self.assertEqual(document.tag, "point_cloud")
            self.assertEqual(document.find("tracks").attrib, {"path": "tracks.ply", "count": "2"})
            self.assertEqual(
                document.find("points").attrib,
                {"component_id": "0", "path": "points0.ply", "count": "2"},
            )
            self.assertEqual(
                [projection.attrib for projection in document.findall("projections")],
                [
                    {"camera_id": "0", "path": "p0.ply", "count": "2"},
                    {"camera_id": "2", "path": "p2.ply", "count": "2"},
                ],
            )

    def test_missing_reference_observation_is_rejected(self):
        frames = SimpleNamespace(dense={0: (torch.zeros(3, 2, 2), None, None)})
        view = {
            "frame_ids": torch.tensor([0]),
            "points": torch.tensor([[0.0, 0.0, 1.0]]),
            "references": torch.tensor([0]),
            "ii": torch.tensor([], dtype=torch.long),
            "jj": torch.tensor([], dtype=torch.long),
            "uv": torch.empty(0, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no reference observation"):
                save_ba_tracks(directory, view, frames)


if __name__ == "__main__":
    unittest.main()
