import torch
from types import SimpleNamespace

from pi3.models.glob3r.geometry import bundle_adjustment_objective, motion_averaging_objective
from local_opt.optimization import (
    bundle_adjust,
    maximum_spanning_tree_initialization,
    robust_rotation_averaging,
    translation_averaging,
)
from local_opt.sfm import Glob3RSfMConfig, Glob3RSfMPipeline
from local_opt.visualization import render_keyframe_matching_overview


def _synthetic_scene():
    dtype = torch.float64
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.1, 0.0]], dtype=dtype)
    points = torch.tensor(
        [[-0.4, -0.2, 4.0], [0.2, 0.3, 5.0], [1.0, -0.1, 6.0], [1.5, 0.4, 5.5]],
        dtype=dtype,
    )
    rotations = torch.eye(3, dtype=dtype).repeat(3, 1, 1)
    world_to_camera = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    world_to_camera[:, :3, 3] = -centers
    intrinsics = torch.tensor(
        [[400.0, 0.0, 256.0], [0.0, 400.0, 192.0], [0.0, 0.0, 1.0]], dtype=dtype
    ).repeat(3, 1, 1)
    observation_camera = torch.arange(3).repeat_interleave(4)
    observation_point = torch.arange(4).repeat(3)
    camera_points = points[observation_point] - centers[observation_camera]
    normalized_rays = camera_points / camera_points[:, 2:3]
    observations = torch.einsum(
        "oij,oj->oi", intrinsics[observation_camera], normalized_rays
    )[:, :2]
    depths = camera_points[:, 2]
    confidence = torch.ones(observation_camera.numel(), dtype=dtype)
    return (
        centers,
        points,
        rotations,
        world_to_camera,
        intrinsics,
        observations,
        normalized_rays,
        depths,
        observation_camera,
        observation_point,
        confidence,
    )


def test_mst_and_rotation_averaging_recover_relative_poses():
    (_, _, _, world_to_camera, _, _, _, _, _, _, _) = _synthetic_scene()
    source = torch.tensor([0, 1, 0])
    target = torch.tensor([1, 2, 2])
    relative = world_to_camera[target] @ torch.linalg.inv(world_to_camera[source])
    weight = torch.tensor([10.0, 8.0, 2.0], dtype=world_to_camera.dtype)
    initialized = maximum_spanning_tree_initialization(3, source, target, relative, weight)
    averaged = robust_rotation_averaging(initialized, source, target, relative, weight)
    assert torch.allclose(initialized, world_to_camera)
    assert torch.allclose(averaged, world_to_camera[:, :3, :3], atol=1.0e-8)


def test_eq5_then_eq6_reduce_their_paper_objectives():
    (
        centers,
        points,
        rotations,
        world_to_camera,
        intrinsics,
        observations,
        rays,
        depths,
        observation_camera,
        observation_point,
        confidence,
    ) = _synthetic_scene()
    initial_centers = centers.clone()
    initial_centers[1:, 0] += torch.tensor([0.15, -0.12], dtype=centers.dtype)
    initial_depths = depths * 1.03
    initial_points = points.clone()
    before_motion = motion_averaging_objective(
        initial_centers,
        initial_points,
        initial_depths,
        rotations,
        rays,
        observation_camera,
        observation_point,
        confidence,
    )
    motion = translation_averaging(
        rotations,
        rays,
        observation_camera,
        observation_point,
        confidence,
        initial_centers,
        initial_depths,
    )
    assert motion.objective < before_motion

    initial_world_to_camera = world_to_camera.clone()
    initial_world_to_camera[:, :3, 3] = -motion.camera_centers
    initial_ba_points = motion.points_3d.clone()
    initial_ba_points[1:, 0] += 0.02
    before_ba = bundle_adjustment_objective(
        initial_ba_points,
        initial_world_to_camera,
        intrinsics,
        observations,
        observation_camera,
        observation_point,
        confidence,
        torch.zeros(3, 4, dtype=points.dtype),
    )
    ba = bundle_adjust(
        initial_world_to_camera,
        initial_ba_points,
        intrinsics,
        observations,
        observation_camera,
        observation_point,
        confidence,
        shared_intrinsics=True,
    )
    assert ba.objective < before_ba
    assert torch.isfinite(ba.world_to_camera).all()
    assert torch.isfinite(ba.points_3d).all()


class _FakeGlob3R:
    def eval(self):
        return self

    def infer_window(self, images, reference_indices):
        _, frames, _, height, width = images.shape
        dtype, device = images.dtype, images.device
        y, x = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )
        focal = 8.0
        points = torch.stack(
            ((x - (width - 1) / 2) / focal * 4, (y - (height - 1) / 2) / focal * 4, torch.full_like(x, 4)),
            dim=-1,
        ).repeat(frames, 1, 1, 1)
        geometry = {
            "local_points": points[None],
            "camera_poses": torch.eye(4, dtype=dtype, device=device).repeat(1, frames, 1, 1),
            "conf": torch.full((1, frames, height, width, 1), 10.0, dtype=dtype, device=device),
            "metric": None,
        }
        references = reference_indices(geometry) if callable(reference_indices) else reference_indices
        matches = {}
        grid = torch.stack((x, y), dim=0)
        for reference in references:
            targets = [index for index in range(frames) if index != reference]
            warp = grid.repeat(1, len(targets), 1, 1, 1)
            confidence = torch.ones(1, len(targets), 1, height, width, dtype=dtype, device=device)
            matches[reference] = SimpleNamespace(
                target_indices=targets,
                coarse_warp=warp,
                coarse_confidence=confidence,
                warp_stages=[warp],
                confidence_stages=[confidence],
            )
        return {"geometry": geometry, "matches": matches}


def test_local_sfm_pipeline_runs_from_one_window_through_dense_fusion():
    images = torch.rand(4, 3, 6, 8, dtype=torch.float32)
    intrinsics = torch.tensor(
        [[8.0, 0.0, 3.5], [0.0, 8.0, 2.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    overviews = []

    def render_matching(callback_images, matches):
        reference, matching = next(iter(matches.items()))
        overviews.append(
            render_keyframe_matching_overview(
                callback_images, matching, reference, target_offset=0
            )
        )

    config = Glob3RSfMConfig(
        tracking_points_per_keyframe=8,
        rotation_iterations=2,
        translation_iterations=3,
        ba_iterations=2,
        dense_depth_ransac_iterations=8,
    )
    result = Glob3RSfMPipeline(_FakeGlob3R(), config).run(
        images, intrinsics, matching_callback=render_matching
    )
    assert len(overviews) == 1
    assert overviews[0].mode == "RGB"
    assert result.world_to_camera.shape == (4, 4, 4)
    assert result.observations.shape[1] == 2
    assert result.raw_points.shape[1] == 3
    assert result.keyframes == [0]
    assert result.raw_points.shape[0] == images.shape[-2] * images.shape[-1]
    assert torch.isfinite(result.raw_points).all()
    assert torch.allclose(result.intrinsics, intrinsics[None].expand(4, -1, -1))
    assert torch.count_nonzero(result.distortion) == 0
    assert result.dense_points is not None
    assert torch.isfinite(result.points_3d).all()
