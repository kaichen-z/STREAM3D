from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from streaming.backend.attention_metric.metric import (
    ConditionMetric,
    ConditionMetricInput,
    ConditionMetricMode,
)
from streaming.backend.attention_metric.metric_entropy import EntropyConditionMetric
from streaming.backend.attention_metric.metric_jam import (
    JointAttentionMassConditionMetric,
)
from streaming.backend.attention_metric.metric_mass_relative import (
    MassRelativeConditionMetric,
)


class AttentionMetricFactory:
    metric_types: dict[ConditionMetricMode, type[ConditionMetric]] = {
        ConditionMetricMode.ENTROPY: EntropyConditionMetric,
        ConditionMetricMode.JOINT_ATTENTION_MASS: JointAttentionMassConditionMetric,
        ConditionMetricMode.MASS_RELATIVE: MassRelativeConditionMetric,
    }

    @classmethod
    def mode(cls, metric: object) -> ConditionMetricMode:
        if isinstance(metric, ConditionMetric):
            return metric.metric
        return ConditionMetricMode(str(metric).strip())

    @classmethod
    def modes(cls, metrics: object) -> tuple[ConditionMetricMode, ...]:
        if metrics is None:
            return ()
        if isinstance(metrics, str) or isinstance(metrics, ConditionMetric):
            values = [metrics]
        else:
            values = list(metrics)
        return tuple(
            dict.fromkeys(cls.mode(value) for value in values if str(value).strip())
        )

    @classmethod
    def build(cls, metric: object) -> ConditionMetric:
        return cls.metric_types[cls.mode(metric)]()

    @classmethod
    def build_many(cls, metrics: object) -> tuple[ConditionMetric, ...]:
        return tuple(cls.build(metric) for metric in cls.modes(metrics))

    @classmethod
    def build_all(cls) -> dict[ConditionMetricMode, ConditionMetric]:
        return {
            mode: metric_type()
            for mode, metric_type in cls.metric_types.items()
        }


def build_condition_metric(metric: object) -> ConditionMetric:
    return AttentionMetricFactory.build(metric)


def build_weight_metric(metric: object) -> ConditionMetric:
    return AttentionMetricFactory.build(metric)


def ss_metric_vector_from_warmup(
    *,
    metric: object,
    attention: torch.Tensor,
    patch_start: int,
    patch_end: int,
) -> torch.Tensor:
    return AttentionMetricFactory.build(metric).compute(
        ConditionMetricInput(
            attention=attention,
            patch_start=int(patch_start),
            patch_end=int(patch_end),
        )
    )


def stage1_score_by_view_from_warmup(
    *,
    warmup: Mapping[str, Any],
    metric: object,
    patch_start: int,
    patch_end: int,
    kappa: float = 1.0,
) -> dict[int, torch.Tensor]:
    metric_impl = AttentionMetricFactory.build(metric)
    return metric_impl.score_by_view(
        {
            int(view_idx): attention_scores
            for view_idx, attention_scores in warmup["attention_scores"].items()
        },
        patch_start=int(patch_start),
        patch_end=int(patch_end),
        kappa=float(kappa),
    )


def stage2_score_by_view_from_attention_scores(
    *,
    attention_scores_by_view: Mapping[int, torch.Tensor],
    metric: object,
    patch_start: int,
    patch_end: int,
    kappa: float,
) -> dict[int, torch.Tensor]:
    metric_impl = AttentionMetricFactory.build(metric)
    return metric_impl.score_by_view(
        {int(view): scores for view, scores in attention_scores_by_view.items()},
        patch_start=int(patch_start),
        patch_end=int(patch_end),
        kappa=float(kappa),
    )
