"""Run standalone Glob3R matching-head forward and backward smoke tests."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pi3.models.glob3r.model import Glob3RMatchingHead
from pi3.models.glob3r.loss import Glob3RMatchingLoss


BATCH_SIZE = 1
FRAME_COUNT = 2
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 56
PATCH_SIZE = 14
ENCODER_DIM = 64
PATCH_COUNT = (IMAGE_HEIGHT // PATCH_SIZE) * (IMAGE_WIDTH // PATCH_SIZE)


def make_synthetic_inputs(device: torch.device):
    images = torch.rand(
        BATCH_SIZE, FRAME_COUNT, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=device
    )
    geometry_tokens = torch.randn(
        BATCH_SIZE,
        FRAME_COUNT,
        PATCH_COUNT,
        2 * ENCODER_DIM,
        device=device,
    )
    encoder_features = [
        torch.randn(
            BATCH_SIZE, FRAME_COUNT, PATCH_COUNT, ENCODER_DIM, device=device
        )
        for _ in range(4)
    ]
    return geometry_tokens, encoder_features, images


def make_synthetic_batch(device: torch.device):
    intrinsics = torch.eye(3, device=device).expand(BATCH_SIZE, -1, -1).clone()
    camera_pose = torch.eye(4, device=device).expand(BATCH_SIZE, -1, -1).clone()
    return [
        {
            "depthmap": torch.ones(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, device=device),
            "camera_intrinsics": intrinsics,
            "camera_pose": camera_pose,
        }
        for _ in range(FRAME_COUNT)
    ]


def run_smoke_test(
    device: torch.device, enable_refinement: bool, check_backward: bool = False
) -> None:
    stage = "coarse+refinement" if enable_refinement else "coarse"
    print(f"Running Glob3R {stage} smoke test on {device}...")
    model = Glob3RMatchingHead(
        encoder_dim=ENCODER_DIM,
        geometry_dim=2 * ENCODER_DIM,
        match_dim=ENCODER_DIM,
        patch_size=PATCH_SIZE,
        enable_refinement=enable_refinement,
    ).to(device)
    model.train(check_backward)
    geometry_tokens, encoder_features, images = make_synthetic_inputs(device)

    gradient_context = nullcontext() if check_backward else torch.no_grad()
    with gradient_context:
        output = model(
            geometry_tokens,
            encoder_features,
            images,
            reference_index=0,
        )

    if check_backward:
        predictions = {
            "match_similarity": output.similarity,
            "coarse_warp": output.coarse_warp,
            "coarse_match_confidence": output.coarse_confidence,
            "warp_stages": output.warp_stages,
            "match_confidence_stages": output.confidence_stages,
            "target_indices": output.target_indices,
            "reference_index": 0,
        }
        loss, _ = Glob3RMatchingLoss()(predictions, make_synthetic_batch(device))
        loss.backward()

    print(f"PASS: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("coarse", "refinement", "all"),
        default="all",
        help="which standalone matching-head path to test",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device; CPU is the intended lightweight default",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="also backpropagate a synthetic objective",
    )
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device(args.device)

    if args.mode in {"coarse", "all"}:
        run_smoke_test(device, enable_refinement=False, check_backward=args.backward)
    if args.mode in {"refinement", "all"}:
        run_smoke_test(device, enable_refinement=True, check_backward=args.backward)
    print("Glob3R matching-head smoke tests passed.")


if __name__ == "__main__":
    main()
