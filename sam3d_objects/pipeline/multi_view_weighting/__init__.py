from __future__ import annotations

from sam3d_objects.pipeline.multi_view_weighting.attention import (
    AttentionCollector,
    SSAttentionCollector,
    _compute_attention_weights,
    _compute_dense_attention,
    _compute_slat_patch_entropy_vector,
    _compute_ss_attention_scores,
    _compute_ss_attention_weights,
    _compute_ss_patch_entropy_vector,
    _reduce_attention_entropy_scalar,
)
from sam3d_objects.pipeline.multi_view_weighting.fusion import (
    WeightedMultiViewFusion,
    _apply_weight_to_prediction,
    _lookup_weight_for_view,
    compute_ss_joint_attention_mass_vector_from_scores,
    compute_ss_joint_attention_mass_weights,
    fuse_predictions,
    weighted_fusion_sparse,
)
from sam3d_objects.pipeline.multi_view_weighting.injection import (
    _build_view_condition_args,
    inject_generator_multi_view_with_collector,
    inject_ss_generator_with_collector,
    inject_weighted_multi_view_with_precomputed_weights,
)

__all__ = [
    "AttentionCollector",
    "SSAttentionCollector",
    "WeightedMultiViewFusion",
    "_apply_weight_to_prediction",
    "_build_view_condition_args",
    "_compute_attention_weights",
    "_compute_dense_attention",
    "_compute_slat_patch_entropy_vector",
    "_compute_ss_attention_scores",
    "_compute_ss_attention_weights",
    "_compute_ss_patch_entropy_vector",
    "_lookup_weight_for_view",
    "_reduce_attention_entropy_scalar",
    "compute_ss_joint_attention_mass_vector_from_scores",
    "compute_ss_joint_attention_mass_weights",
    "fuse_predictions",
    "inject_generator_multi_view_with_collector",
    "inject_ss_generator_with_collector",
    "inject_weighted_multi_view_with_precomputed_weights",
    "weighted_fusion_sparse",
]
