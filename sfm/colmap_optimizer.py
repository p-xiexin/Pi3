"""PyCOLMAP bundle-adjustment backend adapted from image_matcher/colmap_ba.py."""

import importlib

import numpy as np
import torch

from .optimizer import (
    MAX_REPROJECTION_ERROR_PX,
    MIN_TRACK_OBSERVATIONS,
    _bearing_adjust,
    _fixed_camera_mask,
    _project,
    _robust_loss,
    filter_reprojection_observations,
)


def _load_pycolmap():
    """Import the optional backend and enforce the supported binding version."""
    try:
        pycolmap = importlib.import_module("pycolmap")
    except ImportError as error:
        raise ImportError(
            "global_ba_backend=colmap requires pycolmap==4.1.1; install project requirements"
        ) from error
    version = getattr(pycolmap, "COLMAP_version", None)
    if version != "COLMAP 4.1.1":
        raise RuntimeError(
            f"global_ba_backend=colmap requires pycolmap 4.1.1, found {version!r}"
        )
    return pycolmap


def _build_reconstruction(view, pycolmap):
    """Convert one compact factor-graph view into a valid COLMAP reconstruction."""
    poses = view["poses"].detach().cpu().double().numpy()
    points = view["points"].detach().cpu().double().numpy()
    intrinsics = view["K"].detach().cpu().double().numpy()
    observation_cameras = view["ii"].detach().cpu().long().numpy()
    observation_points = view["jj"].detach().cpu().long().numpy()
    observations = view["uv"].detach().cpu().double().numpy()
    camera_count, point_count = poses.shape[0], points.shape[0]
    if observations.shape[0] == 0:
        raise ValueError("COLMAP BA requires at least one observation")
    pair_ids = observation_cameras.astype(np.int64) * point_count + observation_points
    if np.unique(pair_ids).size != pair_ids.size:
        raise ValueError("COLMAP BA requires at most one observation per point and image")

    reconstruction = pycolmap.Reconstruction()
    mean_K = intrinsics.mean(axis=0)
    mean_distortion = view["distortion"].detach().cpu().double().numpy().mean(axis=0)
    camera_id = 1
    camera = pycolmap.Camera(
        model="OPENCV",
        width=max(1, int(round(2.0 * mean_K[0, 2]))),
        height=max(1, int(round(2.0 * mean_K[1, 2]))),
        params=np.array(
            [
                mean_K[0, 0],
                mean_K[1, 1],
                mean_K[0, 2],
                mean_K[1, 2],
                *mean_distortion.tolist(),
            ],
            dtype=np.float64,
        ),
        camera_id=camera_id,
    )
    reconstruction.add_camera_with_trivial_rig(camera)

    point_ids = []
    for xyz in points:
        point_ids.append(int(reconstruction.add_point3D(
            xyz, pycolmap.Track(), np.zeros(3, dtype=np.uint8)
        )))

    image_ids = {}
    for local_camera in range(camera_count):
        selected = np.nonzero(observation_cameras == local_camera)[0]
        if selected.size == 0:
            continue
        points2d = pycolmap.Point2DList([
            pycolmap.Point2D(observations[index], point_ids[observation_points[index]])
            for index in selected
        ])
        image_id = len(image_ids) + 1
        image = pycolmap.Image(
            image_id=image_id,
            name=f"{int(view['frame_ids'][local_camera])}.png",
            camera_id=camera_id,
            points2D=points2d,
        )
        for point2d_index, observation_index in enumerate(selected):
            reconstruction.points3D[point_ids[observation_points[observation_index]]].track.add_element(
                image_id, point2d_index
            )
        reconstruction.add_image_with_trivial_frame(
            image, pycolmap.Rigid3d(poses[local_camera, :3, :])
        )
        image_ids[local_camera] = image_id
    return reconstruction, camera_id, point_ids, image_ids


def _validate_reconstruction(reconstruction):
    """Check both directions of every COLMAP image-to-track association."""
    registered_image_ids = set(reconstruction.reg_image_ids())
    for image_id, image in reconstruction.images.items():
        if image_id not in registered_image_ids or not image.has_pose:
            raise RuntimeError(f"COLMAP image {image_id} is not registered")
        if image.camera_id not in reconstruction.cameras:
            raise RuntimeError(f"COLMAP image {image_id} references a missing camera")
        for point2d in image.points2D:
            if point2d.point3D_id not in reconstruction.points3D:
                raise RuntimeError(
                    f"COLMAP image {image_id} references missing point {point2d.point3D_id}"
                )
    for point_id, point in reconstruction.points3D.items():
        for element in point.track.elements:
            if element.image_id not in reconstruction.images:
                raise RuntimeError(f"COLMAP point {point_id} references a missing image")
            image = reconstruction.images[element.image_id]
            if element.point2D_idx >= len(image.points2D):
                raise RuntimeError(f"COLMAP point {point_id} has an invalid point2D index")
            if image.points2D[element.point2D_idx].point3D_id != point_id:
                raise RuntimeError(f"COLMAP point {point_id} has an inconsistent reverse link")


def _bundle_adjustment_options(pycolmap, iterations):
    """Build the nested COLMAP 4.1.1 Ceres options."""
    options = pycolmap.BundleAdjustmentOptions()
    options.ceres.loss_function_scale = 2.0
    options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.refine_principal_point = True
    # The rest of this SfM pipeline uses a pinhole camera model. Keeping the
    # OPENCV distortion terms fixed at zero makes COLMAP and native BA consume
    # the same projection model and keeps dense unprojection consistent.
    options.refine_extra_params = False
    options.refine_rig_from_world = True
    options.refine_sensor_from_rig = False
    options.refine_focal_length = True
    options.ceres.solver_options.max_num_iterations = int(iterations)
    return options


def _bundle_adjustment_config(pycolmap, reconstruction, image_ids, fixed_ids):
    """Select all images and preserve the requested camera and scale gauge."""
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in image_ids.values():
        config.add_image(image_id)
    for local_camera in fixed_ids:
        image_id = image_ids[int(local_camera)]
        config.set_constant_rig_from_world_pose(
            reconstruction.images[image_id].frame_id
        )
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    return config


def _active_fixed_ids(view, fixed_ids):
    """Keep requested gauge cameras that have measurements in this BA round."""
    _fixed_camera_mask(view["poses"], fixed_ids)
    active = torch.unique(view["ii"]).tolist()
    if not active:
        raise RuntimeError("COLMAP BA requires at least one observed camera")
    active_set = {int(camera) for camera in active}
    resolved = [int(camera) for camera in fixed_ids if int(camera) in active_set]
    return resolved if resolved else [int(active[0])]


def _extract_result(view, reconstruction, camera_id, point_ids, image_ids):
    """Return optimized COLMAP state in the same tensors as the native backend."""
    device, dtype = view["points"].device, view["points"].dtype
    poses = view["poses"].detach().cpu().double().numpy().copy()
    for local_camera, image_id in image_ids.items():
        image = reconstruction.images[image_id]
        cam_from_world = image.cam_from_world()
        poses[local_camera, :3, :3] = np.asarray(cam_from_world.rotation.matrix())
        poses[local_camera, :3, 3] = np.asarray(cam_from_world.translation)
    points = np.stack([
        np.asarray(reconstruction.points3D[point_id].xyz) for point_id in point_ids
    ])
    params = np.asarray(reconstruction.cameras[camera_id].params, dtype=np.float64)
    if params.shape != (8,):
        raise RuntimeError(f"COLMAP OPENCV camera returned {params.size} parameters")
    K = np.array(
        [[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]],
        dtype=np.float64,
    )
    K = torch.as_tensor(K, device=device, dtype=dtype).expand(view["K"].shape[0], -1, -1).clone()
    distortion = torch.as_tensor(params[4:8], device=device, dtype=dtype).expand(
        view["K"].shape[0], -1
    ).clone()
    poses = torch.as_tensor(poses, device=device, dtype=dtype)
    points = torch.as_tensor(points, device=device, dtype=dtype)
    projection, valid = _project(
        poses, points, K, distortion, view["ii"], view["jj"]
    )
    valid_observations = valid.sum()
    if not bool(valid_observations):
        raise RuntimeError("COLMAP BA produced no valid projected observations")
    loss = _robust_loss(
        (view["uv"] - projection)[valid], view["weight"][valid], 2.0
    )
    return {
        "poses": poses,
        "points": points,
        "K": K,
        "distortion": distortion,
        "loss": loss,
        "loss_per_pixel": loss / valid_observations.to(loss.dtype),
        "valid_observations": valid_observations,
        "ba_backend": "colmap",
    }


@torch.no_grad()
def optimize_view_colmap(view, fixed_ids, iterations):
    """Run the COLMAP 4.1.1 global BA policy on the factor-graph view."""
    pycolmap = _load_pycolmap()
    reconstruction, camera_id, point_ids, image_ids = _build_reconstruction(view, pycolmap)
    _validate_reconstruction(reconstruction)
    options = _bundle_adjustment_options(pycolmap, iterations)
    config = _bundle_adjustment_config(
        pycolmap, reconstruction, image_ids, _active_fixed_ids(view, fixed_ids)
    )
    adjuster = pycolmap.create_default_bundle_adjuster(
        options, config, reconstruction
    )
    summary = adjuster.solve()
    if not summary.is_solution_usable():
        raise RuntimeError(f"COLMAP BA failed: {summary.brief_report()}")
    _validate_reconstruction(reconstruction)
    return _extract_result(view, reconstruction, camera_id, point_ids, image_ids)


@torch.no_grad()
def optimize_view_colmap_two_rounds(
    view,
    fixed_ids,
    bearing_iterations=15,
    first_iterations=20,
    second_iterations=10,
    max_error_px=MAX_REPROJECTION_ERROR_PX,
    min_observations=MIN_TRACK_OBSERVATIONS,
):
    """Run the same Eq. (5), filtering, and two-round policy as native BA."""
    poses = view["poses"].clone()
    points = view["points"].clone()
    fixed = _fixed_camera_mask(poses, fixed_ids)
    poses, points = _bearing_adjust(
        poses, points, view, fixed, bearing_iterations
    )
    first_view = dict(view, poses=poses, points=points)
    first_result = optimize_view_colmap(
        first_view, fixed_ids, first_iterations
    )
    first_evaluation = dict(
        view,
        K=first_result["K"],
        distortion=first_result["distortion"],
    )
    first = filter_reprojection_observations(
        first_evaluation,
        first_result["poses"],
        first_result["points"],
        max_error_px=max_error_px,
        min_observations=min_observations,
    )
    first_ids = torch.nonzero(
        first["observation_inliers"], as_tuple=False
    ).squeeze(-1)
    selected = {
        name: first_evaluation[name][first_ids]
        for name in ("ii", "jj", "uv", "weight")
    }
    second_view = dict(
        first_evaluation,
        poses=first_result["poses"],
        points=first_result["points"],
        **selected,
    )
    if "observation_ids" in view:
        second_view["observation_ids"] = view["observation_ids"][first_ids]
    second_result = optimize_view_colmap(
        second_view, fixed_ids, second_iterations
    )
    final_evaluation = dict(
        view,
        K=second_result["K"],
        distortion=second_result["distortion"],
    )
    final = filter_reprojection_observations(
        final_evaluation,
        second_result["poses"],
        second_result["points"],
        max_error_px=max_error_px,
        min_observations=min_observations,
        observation_ids=first_ids,
    )
    final_ids = torch.nonzero(
        final["observation_inliers"], as_tuple=False
    ).squeeze(-1)
    projection, _ = _project(
        second_result["poses"],
        second_result["points"],
        second_result["K"],
        second_result["distortion"],
        view["ii"][final_ids],
        view["jj"][final_ids],
    )
    loss = _robust_loss(
        view["uv"][final_ids] - projection,
        view["weight"][final_ids],
        2.0,
    )
    valid_observations = final["observation_inliers"].sum()
    return {
        "poses": second_result["poses"],
        "points": second_result["points"],
        "K": second_result["K"],
        "distortion": second_result["distortion"],
        "loss": loss,
        "loss_per_pixel": loss / valid_observations.to(loss.dtype),
        "valid_observations": valid_observations,
        "input_observations": torch.as_tensor(
            view["ii"].numel(), device=poses.device, dtype=torch.long
        ),
        "first_inlier_observations": first["observation_inliers"].sum(),
        "observation_inliers": final["observation_inliers"],
        "first_point_inliers": first["point_inliers"],
        "point_inliers": final["point_inliers"],
        "reprojection_error": final["reprojection_error"],
        "ba_backend": "colmap",
    }


__all__ = ["optimize_view_colmap", "optimize_view_colmap_two_rounds"]
