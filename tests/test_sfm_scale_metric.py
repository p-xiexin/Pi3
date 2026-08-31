import unittest

import torch

from sfm.scale_metric import estimate_chunk_scale


class ScaleMetricTest(unittest.TestCase):
    def test_top_confidence_samples_and_overlap_frames_are_averaged(self):
        reference = torch.full((2, 3, 3), 2.0)
        current = torch.stack((torch.full((3, 3), 3.0), torch.full((3, 3), 2.0)))
        reference_scale = torch.tensor([3.0, 4.0])
        reference_confidence = torch.ones_like(reference)
        current_confidence = torch.ones_like(current)

        reference[0, 0, 0] = 100.0
        reference_confidence[0, 0, 0] = 0.01
        current_confidence[0, 0, 0] = 0.01
        estimate = estimate_chunk_scale(
            current,
            reference,
            reference_scale,
            current_confidence,
            reference_confidence,
            sample_points=4,
            minimum_points=4,
        )

        self.assertAlmostEqual(float(estimate), 3.0)

    def test_reference_scale_is_accumulated(self):
        estimate = estimate_chunk_scale(
            torch.ones(1, 3, 3),
            torch.full((1, 3, 3), 2.0),
            torch.tensor([3.0]),
            torch.ones(1, 3, 3),
            torch.ones(1, 3, 3),
            sample_points=4,
            minimum_points=4,
        )

        self.assertAlmostEqual(float(estimate), 6.0)

    def test_missing_high_confidence_samples_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "high-confidence Pi3 depth pixels"):
            estimate_chunk_scale(
                torch.ones(1, 2, 2),
                torch.ones(1, 2, 2),
                torch.ones(1),
                torch.zeros(1, 2, 2),
                torch.ones(1, 2, 2),
                sample_points=2,
                minimum_points=2,
            )


if __name__ == "__main__":
    unittest.main()
