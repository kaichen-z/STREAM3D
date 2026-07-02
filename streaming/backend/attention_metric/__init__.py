from __future__ import annotations

from .config import SLAT_CONDITION_LAYOUT, AttentionWeightingConfig
from .manager import AttentionWeightManager
from .metric import ConditionMetric, ConditionMetricInput, ConditionMetricMode
from .metric_entropy import EntropyConditionMetric
from .metric_jam import JointAttentionMassConditionMetric
from .metric_mass_relative import MassRelativeConditionMetric
from .registry import (
    AttentionMetricFactory,
    build_condition_metric,
    build_weight_metric,
    ss_metric_vector_from_warmup,
    stage1_score_by_view_from_warmup,
    stage2_score_by_view_from_attention_scores,
)


__all__ = [
    "AttentionMetricFactory",
    "AttentionWeightManager",
    "AttentionWeightingConfig",
    "ConditionMetric",
    "ConditionMetricInput",
    "ConditionMetricMode",
    "EntropyConditionMetric",
    "JointAttentionMassConditionMetric",
    "MassRelativeConditionMetric",
    "SLAT_CONDITION_LAYOUT",
    "build_condition_metric",
    "build_weight_metric",
    "ss_metric_vector_from_warmup",
    "stage1_score_by_view_from_warmup",
    "stage2_score_by_view_from_attention_scores",
]
