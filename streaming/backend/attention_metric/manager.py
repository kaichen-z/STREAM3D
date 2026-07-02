from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from streaming.backend.attention_metric.config import AttentionWeightingConfig
from streaming.backend.attention_metric.metric import (
    ConditionMetric,
    ConditionMetricInput,
    ConditionMetricMode,
)
from streaming.backend.attention_metric.metric_jam import (
    JointAttentionMassConditionMetric,
)

logger = logging.getLogger(__name__)


class AttentionWeightManager:
    def __init__(self, config: AttentionWeightingConfig):
        self.config = config
        self._metrics = {
            metric.metric: metric for metric in self.config.weight_metrics
        }
        self._patch_mass_metric = JointAttentionMassConditionMetric()
        self._view_attention_scores: Dict[int, torch.Tensor] = {}
        self._metric_scores: Dict[ConditionMetricMode, Dict[int, torch.Tensor]] = {
            metric: {} for metric in ConditionMetricMode
        }
        self._computed_weights: Optional[Dict[int, torch.Tensor]] = None
        self._expanded_weights: Optional[Dict[int, torch.Tensor]] = None
        self._analysis_data: Dict = {}
        self._downsample_idx: Optional[torch.Tensor] = None
        self._original_coords: Optional[torch.Tensor] = None
        self._downsampled_coords: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._view_attention_scores.clear()
        for scores_by_view in self._metric_scores.values():
            scores_by_view.clear()
        self._computed_weights = None
        self._expanded_weights = None
        self._analysis_data.clear()
        self._downsample_idx = None
        self._original_coords = None
        self._downsampled_coords = None

    def set_downsample_mapping(
        self,
        idx: Optional[torch.Tensor],
        original_coords: Optional[torch.Tensor] = None,
        downsampled_coords: Optional[torch.Tensor] = None,
    ) -> None:
        self._downsample_idx = idx
        self._original_coords = original_coords
        self._downsampled_coords = downsampled_coords
        if idx is not None:
            logger.info(
                f"[AttentionWeightManager] Downsample mapping set: "
                f"{idx.shape[0]} original -> {idx.max().item() + 1} downsampled"
            )

    def add_view_attention_scores(
        self,
        view_idx: int,
        attention_scores: torch.Tensor,
        step: int = 0,
    ) -> None:
        if step != self.config.attention_step:
            logger.debug(
                f"[AttentionWeightManager] Skipping step {step} "
                f"(target: {self.config.attention_step})"
            )
            return

        view_idx = int(view_idx)
        self._view_attention_scores[view_idx] = attention_scores.detach()
        metric_input = ConditionMetricInput(
            attention=attention_scores,
            patch_start=self.config.patch_start,
            patch_end=self.config.patch_end,
        )
        for metric in self.config.weight_metrics:
            if metric.metric is ConditionMetricMode.MASS_RELATIVE:
                continue
            metric_value = metric.compute(metric_input)
            self._metric_scores[metric.metric][view_idx] = metric_value
            self._analysis_data.setdefault(f"{metric.value_key}_per_view", {})[
                view_idx
            ] = metric_value
            logger.info(
                f"[AttentionWeightManager] view {view_idx}: {metric.value_key} "
                f"min={metric_value.min():.4f}, max={metric_value.max():.4f}, "
                f"mean={metric_value.mean():.4f}"
            )

    def add_view_attention(
        self,
        view_idx: int,
        attention: torch.Tensor,
        step: int = 0,
    ) -> None:
        if step != self.config.attention_step:
            logger.debug(
                f"[AttentionWeightManager] Skipping step {step} "
                f"(target: {self.config.attention_step})"
            )
            return
        if self.config.use_patch_mass:
            confidence = self._patch_mass_metric.patch_mass_from_attention(
                attention,
                patch_start=self.config.patch_start,
                patch_end=self.config.patch_end,
            ).pow(float(self.config.patch_mass_gamma))
            self._metric_scores[ConditionMetricMode.JOINT_ATTENTION_MASS][
                int(view_idx)
            ] = confidence

    def compute_weights(self) -> Dict[int, torch.Tensor]:
        self._update_mass_relative_scores()
        self._computed_weights = self._attention_metric_weights()
        self._log_weight_statistics()
        return self._computed_weights

    def get_weights(self) -> Optional[Dict[int, torch.Tensor]]:
        if self._computed_weights is None:
            return self.compute_weights()
        return self._computed_weights

    def get_expanded_weights(self) -> Optional[Dict[int, torch.Tensor]]:
        if self._computed_weights is None:
            self.compute_weights()
        if self._computed_weights is None:
            return None
        if self._expanded_weights is not None:
            return self._expanded_weights
        if self._downsample_idx is None:
            logger.warning(
                "[AttentionWeightManager] No downsample mapping, returning original weights. "
                "This may cause dimension mismatch!"
            )
            return self._computed_weights

        self._expanded_weights = {
            view_idx: weight[self._downsample_idx]
            for view_idx, weight in self._computed_weights.items()
        }
        logger.info(
            f"[AttentionWeightManager] Expanded weights: "
            f"{next(iter(self._computed_weights.values())).shape[0]} -> "
            f"{self._downsample_idx.shape[0]}"
        )
        return self._expanded_weights

    def get_original_coords(self) -> Optional[torch.Tensor]:
        return self._original_coords

    def get_downsampled_coords(self) -> Optional[torch.Tensor]:
        return self._downsampled_coords

    def get_analysis_data(self) -> Dict:
        return {
            "config": self.config,
            "weights": self._computed_weights,
            "expanded_weights": self._expanded_weights,
            "joint_attention_mass_per_view": self._analysis_data.get(
                "joint_attention_mass_per_view", {}
            ),
            "entropy_confidence_per_view": self._analysis_data.get(
                "entropy_confidence_per_view", {}
            ),
            "mass_relative_per_view": self._analysis_data.get(
                "mass_relative_per_view", {}
            ),
            "confidences": self._metric_scores[
                ConditionMetricMode.JOINT_ATTENTION_MASS
            ],
            "entropy_confidences": self._metric_scores[ConditionMetricMode.ENTROPY],
            "mass_relative_confidences": self._metric_scores[
                ConditionMetricMode.MASS_RELATIVE
            ],
            "downsample_idx": self._downsample_idx,
            "original_coords": self._original_coords,
            "downsampled_coords": self._downsampled_coords,
        }

    def _update_mass_relative_scores(self) -> None:
        metric = self._metrics.get(ConditionMetricMode.MASS_RELATIVE)
        if metric is None:
            return
        self._metric_scores[ConditionMetricMode.MASS_RELATIVE] = metric.score_by_view(
            self._view_attention_scores,
            patch_start=self.config.patch_start,
            patch_end=self.config.patch_end,
            kappa=float(self.config.jam_kappa),
        )
        self._analysis_data["mass_relative_per_view"] = self._metric_scores[
            ConditionMetricMode.MASS_RELATIVE
        ]
        logger.info(
            f"[AttentionWeightManager] mass_relative weights computed "
            f"(kappa={float(self.config.jam_kappa):.3f})"
        )

    def _attention_metric_weights(self) -> Dict[int, torch.Tensor]:
        if not self.config.weight_metrics:
            logger.info(
                "[AttentionWeightManager] No weight metrics configured; "
                "using original multi-diffusion."
            )
            return {}

        weights_by_metric: list[Dict[int, torch.Tensor]] = []
        for metric in self.config.weight_metrics:
            score_by_view = dict(self._metric_scores.get(metric.metric, {}))
            if not score_by_view:
                logger.warning(
                    f"[AttentionWeightManager] No {metric.value_key} scores collected!"
                )
                return {}
            weights_by_metric.append(self._metric_weights(metric, score_by_view))

        if len(weights_by_metric) == 1:
            return weights_by_metric[0]
        return self._combined_metric_weights(weights_by_metric)

    def _metric_weights(
        self,
        metric: ConditionMetric,
        score_by_view: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        return metric.normalize_scores(
            score_by_view,
            exponent=(
                float(self.config.jam_alpha)
                / float(self.config.final_temperature)
            ),
            uniform_blend=float(self.config.uniform_blend),
            min_weight=float(self.config.min_weight),
        )

    def _combined_metric_weights(
        self,
        weights_by_metric: list[Dict[int, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        common_views = set(weights_by_metric[0])
        for metric_weights in weights_by_metric[1:]:
            common_views &= set(metric_weights)
        common_views = sorted(common_views)
        if not common_views:
            logger.warning(
                "[AttentionWeightManager] No common views across configured weight metrics!"
            )
            return {}

        scale = 1.0 / len(weights_by_metric)
        combined = {
            view: sum(metric_weights[view] for metric_weights in weights_by_metric)
            * scale
            for view in common_views
        }
        return self._renormalize_combined(combined)

    def _renormalize_combined(
        self,
        combined: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        if self.config.min_weight <= 0 or len(combined) <= 1:
            return combined
        views = sorted(combined)
        weight_stack = torch.stack([combined[view] for view in views], dim=0)
        weight_stack = weight_stack.clamp(min=float(self.config.min_weight))
        weight_stack = weight_stack / weight_stack.sum(dim=0, keepdim=True).clamp(
            min=1e-10
        )
        return {view: weight_stack[index] for index, view in enumerate(views)}

    def _log_weight_statistics(self) -> None:
        if not self._computed_weights:
            return
        views = sorted(self._computed_weights)
        weights_stack = torch.stack([self._computed_weights[view] for view in views])
        logger.info(f"[AttentionWeightManager] Weight statistics ({len(views)} views):")
        for view in views:
            weight = self._computed_weights[view]
            logger.info(
                f"  View {view}: mean={weight.mean():.4f}, std={weight.std():.4f}, "
                f"min={weight.min():.4f}, max={weight.max():.4f}"
            )
        weight_std = weights_stack.std(dim=0)
        logger.info(
            f"  Cross-view std: mean={weight_std.mean():.4f}, max={weight_std.max():.4f}"
        )
