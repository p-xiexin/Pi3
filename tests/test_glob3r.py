"""CPU unit tests for Glob3R equations that do not require a pretrained Pi3X."""

import torch

from check_glob3r_equations import FORMULA_IMPLEMENTATIONS
from pi3.models.glob3r.geometry import (
    build_ground_truth_warp,
    bundle_adjustment_objective,
    motion_averaging_objective,
)
from pi3.models.glob3r.loss import (
    auxiliary_nll_loss,
    confidence_loss,
    generalized_charbonnier_loss,
)
from pi3.models.glob3r.model import MultiViewMatchEmbedding


def test_every_paper_equation_is_registered():
    assert set(FORMULA_IMPLEMENTATIONS) == set(range(1, 36))


def test_identity_camera_produces_identity_warp_and_positive_confidence():
    batch, targets, height, width = 1, 1, 4, 6
    depth = torch.ones(batch, height, width)
    target_depth = depth[:, None].expand(-1, targets, -1, -1)
    intrinsics = torch.eye(3).unsqueeze(0)
    target_intrinsics = intrinsics[:, None].expand(-1, targets, -1, -1)
    transform = torch.eye(4).reshape(1, 1, 4, 4)
    supervision = build_ground_truth_warp(
        depth, target_depth, intrinsics, target_intrinsics, transform
    )
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    expected = torch.stack((x, y), dim=-1).float().reshape(1, 1, height, width, 2)
    torch.testing.assert_close(supervision.warp, expected)
    assert supervision.confidence.all()
    assert supervision.mask.all()


def test_multiview_similarity_and_fourier_embedding_shapes():
    module = MultiViewMatchEmbedding(dim=16)
    tokens = torch.randn(2, 3, 6, 16)
    similarity, embedding, targets, indices = module(tokens, 2, 3, reference_index=1)
    assert similarity.shape == (2, 2, 6, 6)
    assert embedding.shape == (2, 2, 6, 16)
    assert targets.shape == (2, 2, 6, 16)
    assert indices == [0, 2]


def test_matching_losses_are_finite_and_zero_residual_is_minimal():
    similarity = torch.eye(4).reshape(1, 1, 4, 4) * 10
    labels = torch.arange(4).reshape(1, 1, 4)
    assert torch.isfinite(auxiliary_nll_loss(similarity, labels))

    warp = torch.zeros(1, 1, 2, 2, 2)
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    zero = generalized_charbonnier_loss(warp, warp, valid)
    nonzero = generalized_charbonnier_loss(warp + 1, warp, valid)
    assert zero < nonzero

    confidence = torch.full((1, 1, 1, 2, 2), 0.9)
    assert torch.isfinite(confidence_loss(confidence, valid, valid))


def test_global_objectives_are_zero_for_exact_geometry():
    centers = torch.zeros(1, 3)
    points = torch.tensor([[0.0, 0.0, 2.0]])
    depths = torch.tensor([2.0])
    rotations = torch.eye(3).unsqueeze(0)
    rays = torch.tensor([[0.0, 0.0, 1.0]])
    indices = torch.zeros(1, dtype=torch.long)
    confidence = torch.ones(1)
    motion = motion_averaging_objective(
        centers, points, depths, rotations, rays, indices, indices, confidence
    )
    torch.testing.assert_close(motion, torch.tensor(0.0))

    world_to_camera = torch.eye(4).unsqueeze(0)
    intrinsics = torch.eye(3).unsqueeze(0)
    observation = torch.tensor([[0.0, 0.0]])
    ba = bundle_adjustment_objective(
        points,
        world_to_camera,
        intrinsics,
        observation,
        indices,
        indices,
        confidence,
    )
    torch.testing.assert_close(ba, torch.tensor(0.0))


def test_matching_overview_renders_prediction_grid():
    from utils.glob3r_visualization import render_matching_overview

    images = torch.zeros(1, 2, 3, 4, 6)
    y, x = torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij")
    warp = torch.stack((x, y), dim=0).float().reshape(1, 1, 2, 4, 6)
    prediction = {
        "warp": warp,
        "warp_confidence": torch.ones(1, 1, 1, 4, 6),
        "target_indices": [1],
        "reference_index": 0,
    }
    views = [
        {
            "depthmap": torch.ones(1, 4, 6),
            "camera_intrinsics": torch.eye(3).unsqueeze(0),
            "camera_pose": torch.eye(4).unsqueeze(0),
        }
        for _ in range(2)
    ]
    rendered = render_matching_overview(images, prediction, views, cell_width=6)
    assert rendered.mode == "RGB"
    assert rendered.size == (68, 16)
