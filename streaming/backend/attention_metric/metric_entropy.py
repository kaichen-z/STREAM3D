from __future__ import annotations

import torch

from streaming.backend.attention_metric.metric import (
    ConditionMetric,
    ConditionMetricInput,
    ConditionMetricMode,
)


class EntropyConditionMetric(ConditionMetric):
    metric = ConditionMetricMode.ENTROPY

    def compute(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        _, entropy_confidence = self.patch_mass_and_entropy_confidence(metric_input)
        return entropy_confidence
