from __future__ import annotations

from dataclasses import dataclass, field

from streaming.backend.attention_metric.metric import ConditionMetric
from streaming.backend.attention_metric.registry import AttentionMetricFactory


SLAT_CONDITION_LAYOUT = {
    "image_cropped": (0, 1374),
    "mask_cropped": (1374, 2748),
    "image_full": (2748, 4122),
    "mask_full": (4122, 5496),
}


@dataclass
class AttentionWeightingConfig:
    weight_source: str | list[str]
    jam_alpha: float
    jam_kappa: float
    uniform_blend: float
    use_patch_mass: bool
    patch_mass_gamma: float
    final_temperature: float
    min_weight: float
    attention_layer: int
    attention_step: int
    patch_start: int
    patch_end: int
    weight_metrics: tuple[ConditionMetric, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.weight_metrics = AttentionMetricFactory.build_many(self.weight_source)
        self.weight_source = [metric.metric.value for metric in self.weight_metrics]

    @property
    def uses_attention(self) -> bool:
        return bool(self.weight_metrics)
