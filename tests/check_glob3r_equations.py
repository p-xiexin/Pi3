"""Static completeness check for the Glob3R equation-to-code mapping."""

import ast
from pathlib import Path
import re


FORMULA_IMPLEMENTATIONS = {
    1: "model.pack_geometry_prediction",
    2: "model.Glob3RMatchingHead.forward",
    3: "loss.Glob3RMatchingLoss.forward",
    4: "geometry.keyframe_projection_count",
    5: "geometry.motion_averaging_objective",
    6: "geometry.bundle_adjustment_objective",
    7: "model.Glob3RMatchingHead.forward:encoder_features",
    8: "model.Glob3RMatchingHead.forward:geometry_tokens",
    9: "model.MatchingDecoder.forward",
    10: "model.MultiViewMatchEmbedding.forward:reference",
    11: "model.MultiViewMatchEmbedding.forward:targets",
    12: "model.MultiViewMatchEmbedding.forward:similarity",
    13: "model.MultiViewMatchEmbedding.forward:cosine_similarity",
    14: "model.MultiViewMatchEmbedding.forward:similarity_stack",
    15: "model.MultiViewMatchEmbedding.fourier_target_coordinates",
    16: "model.MultiViewMatchEmbedding.forward:embedding_aggregation",
    17: "model.MultiViewMatchEmbedding.forward:embedding_stack",
    18: "model.DPTMatchingHead.forward:pair_projection",
    19: "model.DPTMatchingHead.forward:pair_stack",
    20: "model.DPTMatchingHead.forward",
    21: "model.DPTMatchingHead.forward:output_contract",
    22: "model.FineFeaturePyramid.forward",
    23: "model.WarpRefinement.forward:refine_feature",
    24: "model.DepthwiseRefinementBlock.forward",
    25: "model.WarpRefinement.forward:warp_update",
    26: "geometry.build_ground_truth_warp:back_projection",
    27: "geometry.build_ground_truth_warp:projection",
    28: "geometry.build_ground_truth_warp:ground_truth_warp",
    29: "geometry.build_ground_truth_warp:confidence",
    30: "geometry.build_ground_truth_warp:mask",
    31: "loss.auxiliary_nll_loss",
    32: "loss.generalized_charbonnier_loss:residual",
    33: "loss.generalized_charbonnier_loss:penalty",
    34: "loss.confidence_loss",
    35: "loss.Glob3RMatchingLoss.forward:total",
}


def main() -> None:
    expected = set(range(1, 36))
    mapped = set(FORMULA_IMPLEMENTATIONS)
    if mapped != expected:
        raise SystemExit(f"equation registry mismatch: missing={expected-mapped}, extra={mapped-expected}")

    implementation_sources = []
    for path in Path("pi3/models/glob3r").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        implementation_sources.append(source)
    cited = {
        int(number)
        for number in re.findall(r"(?:Eq|Eqs)\.\s*\((\d+)\)", "\n".join(implementation_sources))
    }
    if cited != expected:
        raise SystemExit(f"equation comment mismatch: missing={expected-cited}, extra={cited-expected}")
    print(f"Glob3R equation audit passed: {len(mapped)} / {len(expected)} formulas mapped")


if __name__ == "__main__":
    main()
