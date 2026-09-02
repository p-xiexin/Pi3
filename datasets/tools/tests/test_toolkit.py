from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from datasets.tools.engine import extract_archive, verify_rules
from datasets.tools.registry import load_catalog


class CatalogTests(unittest.TestCase):
    def test_catalog_has_both_profiles(self) -> None:
        catalog = load_catalog()
        self.assertGreaterEqual(len(catalog.datasets), 13)
        for dataset in catalog.datasets.values():
            self.assertEqual(set(dataset["profiles"]), {"minimal", "train"})

    def test_profile_aliases(self) -> None:
        catalog = load_catalog()
        self.assertEqual(catalog.profile("sintel", "test")[1], "minimal")
        self.assertEqual(catalog.profile("sintel", "full")[1], "train")

    def test_hypersim_minimal_uses_selective_download(self) -> None:
        catalog = load_catalog()
        profile, _ = catalog.profile("hypersim", "minimal")
        self.assertEqual(profile["recipe"]["type"], "remote-zip")
        self.assertEqual(len(profile["recipe"]["artifacts"]), 1)
        self.assertEqual(len(profile["recipe"]["artifacts"][0]["entries"]), 21)

    def test_sequence_datasets_generate_explicit_indexes(self) -> None:
        catalog = load_catalog()
        indexed = {
            "tum_rgbd",
            "redwood",
            "eth3d_slam",
            "hypersim",
            "sintel",
        }
        for dataset_id in indexed:
            dataset = catalog.get(dataset_id)
            self.assertIn(
                {"glob": "pi3_index.npy", "min_count": 1}, dataset["verify"]
            )
            for profile_name in ("minimal", "train"):
                profile, _ = catalog.profile(dataset_id, profile_name)
                command = profile["postprocess"][0]
                self.assertEqual(command[:4], [
                    "{python}", "-m", "datasets.tools.build_index", dataset_id
                ])


class EngineTests(unittest.TestCase):
    def test_extract_archive_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("scene/rgb/0001.png", b"image")
            output = root / "output"
            extract_archive(archive, output)
            results = verify_rules(output, [{"glob": "*/rgb/*.png", "min_count": 1}])
            self.assertTrue(results[0]["ok"])

    def test_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escaped.txt", b"unsafe")
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
