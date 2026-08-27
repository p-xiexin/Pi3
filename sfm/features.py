"""Shared image-derived queries for the two tracks frontends."""

import torch


QUERY_CANDIDATE_MULTIPLIER = 4
SUPERPOINT_DETECTION_THRESHOLD = 0.005
POINTS_PER_KEYFRAME = 2048


def _top_score_indices(points, scores, max_points):
    """Keep the global top-scoring detector outputs without spatial bucketing."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("keypoints must have shape [P,2]")
    if scores.shape != points.shape[:1]:
        raise ValueError("keypoint scores must have shape [P]")
    if int(max_points) < 1:
        raise ValueError("max_points must be positive")
    if points.shape[0] <= int(max_points):
        return torch.arange(points.shape[0], device=points.device)
    return scores.topk(int(max_points)).indices


def _filtered_features(extractor, image, valid_mask):
    invalid = None if valid_mask is None else (~valid_mask.bool())[None]
    features = extractor.extract(image[None], invalid_mask=invalid)
    points = features["keypoints"][0]
    scores = features["keypoint_scores"][0]
    finite = torch.isfinite(points).all(-1) & torch.isfinite(scores)
    points, scores = points[finite], scores[finite]
    if valid_mask is not None and points.numel():
        pixels = points.round().long()
        pixels[:, 0].clamp_(0, image.shape[-1] - 1)
        pixels[:, 1].clamp_(0, image.shape[-2] - 1)
        keep = valid_mask[pixels[:, 1], pixels[:, 0]]
        points, scores = points[keep], scores[keep]
    return points, scores


class ImageQueryFeatures:
    """Extract the same global-top SuperPoint and SIFT queries for both frontends."""

    def __init__(
        self,
        max_points=POINTS_PER_KEYFRAME,
        device="cpu",
        detection_threshold=SUPERPOINT_DETECTION_THRESHOLD,
    ):
        try:
            from lightglue import SIFT, SuperPoint
        except ImportError as error:
            raise ImportError(
                "lightglue is required for SuperPoint and SIFT track queries"
            ) from error
        self.max_points = int(max_points)
        candidate_count = self.max_points * QUERY_CANDIDATE_MULTIPLIER
        self.extractors = (
            SuperPoint(
                max_num_keypoints=candidate_count,
                detection_threshold=float(detection_threshold),
            ).to(device).eval(),
            SIFT(max_num_keypoints=candidate_count).to(device).eval(),
        )

    @torch.no_grad()
    def extract(self, image, valid_mask=None):
        """Merge detector outputs and keep the global top scores."""
        if valid_mask is not None and valid_mask.shape != image.shape[-2:]:
            raise ValueError("valid_mask must have shape [H,W]")
        feature_parts = [
            _filtered_features(extractor, image, valid_mask)
            for extractor in self.extractors
        ]
        point_parts = [points for points, _ in feature_parts if points.numel()]
        score_parts = [scores for points, scores in feature_parts if points.numel()]
        if not point_parts:
            raise RuntimeError("SuperPoint and SIFT found no valid query points")
        points = torch.cat(point_parts)
        scores = torch.cat(score_parts)
        selected = _top_score_indices(points, scores, self.max_points)
        return points[selected].to(device=image.device, dtype=image.dtype)


def build_query_features(config):
    """Build the shared query extractor used by Glob3R and VGGSfM."""
    tracks_model = config["tracks_model"]
    if tracks_model not in {"glob3r", "vgg"}:
        raise ValueError("tracks_model must be glob3r or vgg")
    return ImageQueryFeatures(
        max_points=int(config.get("points_per_keyframe", POINTS_PER_KEYFRAME)),
        device=config["device"],
        detection_threshold=float(
            config.get(
                "superpoint_detection_threshold",
                SUPERPOINT_DETECTION_THRESHOLD,
            )
        ),
    )


__all__ = [
    "ImageQueryFeatures",
    "POINTS_PER_KEYFRAME",
    "QUERY_CANDIDATE_MULTIPLIER",
    "SUPERPOINT_DETECTION_THRESHOLD",
    "build_query_features",
]
