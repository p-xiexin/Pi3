"""Dataset-to-model adapter for ordered ABot-Recon training sequences.

Appendix A.1 requires every video sample to preserve a locally continuous
temporal order and constructs pseudo-sequences for unordered multiview data.
Pi3's dataloader emits ``list[view_dict]`` batches, so this module establishes
the ordered tensor contract consumed jointly by the model and loss.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import torch


def _sample_value(value: Any, batch_index: int) -> Any:
    """Read one sample's metadata from a collated tensor or Python sequence."""

    if torch.is_tensor(value):
        item = value[batch_index]
        return item.item() if item.numel() == 1 else item
    if isinstance(value, (list, tuple)):
        return value[batch_index]
    return value


def _numeric_frame_key(view: Dict[str, Any], batch_index: int) -> float | None:
    """Recover a sortable time index from ``frame_id`` or ``instance``."""

    for field in ("frame_id", "instance"):
        if field not in view:
            continue
        value = _sample_value(view[field], batch_index)
        try:
            return float(value)
        except (TypeError, ValueError):
            match = re.search(r"(\d+)(?!.*\d)", str(value))
            if match is not None:
                return float(match.group(1))
    return None


def _frame_orders(views: Sequence[Dict[str, Any]], batch_size: int) -> torch.Tensor:
    """Construct an independent view permutation for every batch element."""

    orders: List[List[int]] = []
    for batch_index in range(batch_size):
        keys = [_numeric_frame_key(view, batch_index) for view in views]
        if all(key is not None for key in keys):
            orders.append(sorted(range(len(views)), key=lambda index: keys[index]))
        else:
            orders.append(list(range(len(views))))
    return torch.tensor(orders, dtype=torch.long)


def _stack_and_reorder(
    views: Sequence[Dict[str, Any]], field: str, orders: torch.Tensor
) -> torch.Tensor:
    """Stack one view field and apply the same temporal permutation as RGB."""

    stacked = torch.stack([view[field] for view in views], dim=1)
    batch_index = torch.arange(stacked.shape[0], device=stacked.device)[:, None]
    return stacked[batch_index, orders.to(stacked.device)]


def prepare_abot_batch(views: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert Pi3's view list into the ordered sequence required by Appendix A.1.

    BaseDataset may shuffle the view list.  When frame_id or a numeric instance is
    available, ordering is recovered independently for every batch element.  For
    graph-ordered multiview datasets without numeric ids, the dataset must set
    shuffle=false so its path order is preserved.
    """
    if not views:
        raise ValueError("ABot-Recon requires at least one view")
    if "img" not in views[0]:
        raise KeyError("Every ABot-Recon view must contain img")

    batch_size = int(views[0]["img"].shape[0])
    orders = _frame_orders(views, batch_size)
    # Eq. (1) consumes the ordered image stream I_0..I_(N-1).  The identical
    # permutation must be applied to every geometric target or the point and pose
    # objectives would supervise a different time index than the RGB input.
    result: Dict[str, Any] = {
        "imgs": _stack_and_reorder(views, "img", orders),
        "orders": orders,
    }
    optional_fields = {
        "pts3d": "world_points",
        "valid_mask": "valid_masks",
        "camera_pose": "camera_poses",
        "camera_intrinsics": "intrinsics",
    }
    for source, target in optional_fields.items():
        if all(source in view for view in views):
            result[target] = _stack_and_reorder(views, source, orders)

    dataset = views[0].get("dataset")
    if dataset is not None:
        result["dataset_names"] = dataset
    return result


def validate_training_batch(batch: Dict[str, Any]) -> None:
    """Validate the tensor contract needed by Eq. (1) and losses (12)-(15)."""

    required = ("imgs", "world_points", "valid_masks", "camera_poses")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError("ABot-Recon training batch is missing " + ", ".join(missing))
    imgs = batch["imgs"]
    points = batch["world_points"]
    masks = batch["valid_masks"]
    poses = batch["camera_poses"]
    if imgs.ndim != 5 or points.ndim != 5 or masks.ndim != 4 or poses.ndim != 4:
        raise ValueError(
            "Expected imgs [B,N,3,H,W], world_points [B,N,H,W,3], "
            "valid_masks [B,N,H,W], camera_poses [B,N,4,4]"
        )
    if imgs.shape[:2] != points.shape[:2] or masks.shape[:2] != imgs.shape[:2]:
        raise ValueError("Image, point and mask sequence dimensions do not agree")
    if poses.shape[:2] != imgs.shape[:2] or poses.shape[-2:] != (4, 4):
        raise ValueError("camera_poses must have shape [B,N,4,4]")
