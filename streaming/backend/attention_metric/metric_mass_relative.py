from __future__ import annotations

from collections.abc import Mapping

import torch

from streaming.backend.attention_metric.metric import (
    ConditionMetric,
    ConditionMetricInput,
    ConditionMetricMode,
)


class MassRelativeConditionMetric(ConditionMetric):
    metric = ConditionMetricMode.MASS_RELATIVE

    def compute(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        raise ValueError(
            "mass_relative is a cross-view metric; call score_by_view() instead."
        )

    def score_by_view(
        self,
        attention_scores_by_view: Mapping[int, torch.Tensor],
        *,
        patch_start: int,
        patch_end: int,
        kappa: float = 1.0,
    ) -> dict[int, torch.Tensor]:
        patch_mass_by_view = {}
        entropy_confidence_by_view = {}
        for view, attention_scores in sorted(attention_scores_by_view.items()):
            patch_mass, entropy_confidence = self.components(
                attention_scores,
                patch_start=patch_start,
                patch_end=patch_end,
            )
            patch_mass_by_view[int(view)] = patch_mass
            entropy_confidence_by_view[int(view)] = entropy_confidence
        return self.from_components(
            patch_mass_by_view,
            entropy_confidence_by_view,
            kappa=float(kappa),
        )

    def components(
        self,
        attention_scores: torch.Tensor,
        *,
        patch_start: int,
        patch_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patch_mass_and_entropy_confidence(
            ConditionMetricInput(
                attention=attention_scores,
                patch_start=int(patch_start),
                patch_end=min(int(patch_end), int(attention_scores.shape[-1])),
            )
        )

    def from_components(
        self,
        patch_mass_by_view: Mapping[int, torch.Tensor],
        entropy_confidence_by_view: Mapping[int, torch.Tensor],
        *,
        kappa: float = 1.0,
    ) -> dict[int, torch.Tensor]:
        views = sorted(set(patch_mass_by_view) & set(entropy_confidence_by_view))
        if not views:
            return {}

        patch_mass_stack = torch.stack(
            [
                patch_mass_by_view[view].detach().cpu().to(torch.float32).flatten()
                for view in views
            ],
            dim=0,
        )
        entropy_stack = torch.stack(
            [
                entropy_confidence_by_view[view]
                .detach()
                .cpu()
                .to(torch.float32)
                .flatten()
                for view in views
            ],
            dim=0,
        )
        relative_mass = 1.0 + float(kappa) * (
            patch_mass_stack - patch_mass_stack.mean(dim=0, keepdim=True)
        )
        evidence = relative_mass.clamp(min=1e-6) * entropy_stack
        return {
            view: evidence[position].contiguous()
            for position, view in enumerate(views)
        }
