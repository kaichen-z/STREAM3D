from __future__ import annotations

import torch

from streaming.backend.attention_metric.metric import (
    ConditionMetric,
    ConditionMetricInput,
    ConditionMetricMode,
)


class JointAttentionMassConditionMetric(ConditionMetric):
    metric = ConditionMetricMode.JOINT_ATTENTION_MASS

    def compute(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        patch_mass, entropy_confidence = self.patch_mass_and_entropy_confidence(
            metric_input
        )
        return (patch_mass * entropy_confidence).clamp(min=0.0, max=1.0)
