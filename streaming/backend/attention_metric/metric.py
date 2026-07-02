from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Tuple

import torch


@dataclass(frozen=True)
class ConditionMetricInput:
    # Stores pre-softmax attention scores. Metrics softmax each head before
    # reducing, matching the verified mass-relative path.
    attention: torch.Tensor
    patch_start: int
    patch_end: int


class ConditionMetricMode(StrEnum):
    # All condition metrics expose a score where larger values are preferred.
    ENTROPY = "entropy"
    JOINT_ATTENTION_MASS = "joint_attention_mass"
    MASS_RELATIVE = "mass_relative"

    @property
    def value_key(self) -> str:
        if self is type(self).ENTROPY:
            return "entropy_confidence"
        return self.value

    @property
    def best_summary_keys(self) -> Tuple[str, str, str]:
        value_key = self.value_key
        return (
            f"mean_best_{value_key}",
            f"min_best_{value_key}",
            f"max_best_{value_key}",
        )


class ConditionMetric(ABC):
    metric: ConditionMetricMode

    @property
    def value_key(self) -> str:
        return self.metric.value_key

    def __call__(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        return self.compute(metric_input)

    @abstractmethod
    def compute(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        """Return a flattened larger-is-better vector for one view."""

    def score_by_view(
        self,
        attention_scores_by_view: Mapping[int, torch.Tensor],
        *,
        patch_start: int,
        patch_end: int,
        **_: object,
    ) -> dict[int, torch.Tensor]:
        return {
            int(view): self.compute(
                ConditionMetricInput(
                    attention=attention_scores,
                    patch_start=int(patch_start),
                    patch_end=int(patch_end),
                )
            )
            for view, attention_scores in sorted(attention_scores_by_view.items())
        }

    def reduce_to_vector(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() > 1:
            value = value.mean(dim=tuple(range(value.dim() - 1)))
        return value.detach().cpu().to(torch.float32).flatten().contiguous()

    def attention(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        return torch.softmax(metric_input.attention.to(torch.float32), dim=-1)

    def patch_attention(self, metric_input: ConditionMetricInput) -> torch.Tensor:
        attention = self.attention(metric_input)
        patch_attention = attention[
            ..., int(metric_input.patch_start) : int(metric_input.patch_end)
        ]
        self._check_patch_count(patch_attention)
        return patch_attention

    def patch_mass_and_entropy_confidence(
        self,
        metric_input: ConditionMetricInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_attention = self.patch_attention(metric_input)
        num_patches = int(patch_attention.shape[-1])
        patch_mass = patch_attention.sum(dim=-1)
        patch_distribution = patch_attention / patch_mass.unsqueeze(-1).clamp(
            min=1e-10
        )

        if num_patches == 1:
            entropy_confidence = torch.ones_like(patch_mass)
        else:
            entropy = -(
                patch_distribution * torch.log(patch_distribution + 1e-10)
            ).sum(dim=-1)
            entropy_confidence = (1.0 - entropy / math.log(num_patches)).clamp(
                min=0.0,
                max=1.0,
            )

        return (
            self.reduce_to_vector(patch_mass),
            self.reduce_to_vector(entropy_confidence),
        )

    def patch_mass_from_attention(
        self,
        attention: torch.Tensor,
        *,
        patch_start: int,
        patch_end: int,
    ) -> torch.Tensor:
        patch_attention = attention[..., int(patch_start) : int(patch_end)]
        self._check_patch_count(patch_attention)
        total_mass = attention.sum(dim=-1).clamp(min=1e-10)
        return self.reduce_to_vector(patch_attention.sum(dim=-1) / total_mass)

    def normalize_scores(
        self,
        score_by_view: Mapping[int, torch.Tensor],
        *,
        exponent: float,
        uniform_blend: float,
        min_weight: float,
    ) -> dict[int, torch.Tensor]:
        views = sorted(int(view) for view in score_by_view)
        if not views:
            return {}
        if len(views) == 1:
            view = views[0]
            return {view: torch.ones_like(score_by_view[view])}

        score_stack = torch.stack([score_by_view[view] for view in views], dim=0)
        weights = self.normalize_score_stack(
            score_stack,
            exponent=float(exponent),
            uniform_blend=float(uniform_blend),
            min_weight=float(min_weight),
        )
        return {view: weights[index] for index, view in enumerate(views)}

    def normalize_score_stack(
        self,
        score_stack: torch.Tensor,
        *,
        exponent: float,
        uniform_blend: float,
        min_weight: float,
    ) -> torch.Tensor:
        if score_stack.dim() < 2:
            raise ValueError(
                f"Expected score stack [V, ...], got {tuple(score_stack.shape)}."
            )
        if float(exponent) <= 0:
            raise ValueError(f"exponent must be positive, got {exponent}.")
        if not 0.0 <= float(uniform_blend) <= 1.0:
            raise ValueError(
                f"uniform_blend must be in [0, 1], got {uniform_blend}."
            )

        score = score_stack.to(torch.float32).clamp(min=0.0)
        if float(exponent) != 1.0:
            score = score.pow(float(exponent))

        denom = score.sum(dim=0, keepdim=True)
        uniform = torch.full_like(score, 1.0 / int(score.shape[0]))
        weights = torch.where(denom > 1e-10, score / denom.clamp(min=1e-10), uniform)
        if float(uniform_blend) > 0.0:
            weights = (1.0 - float(uniform_blend)) * weights + float(
                uniform_blend
            ) * uniform
        if float(min_weight) > 0.0:
            weights = weights.clamp(min=float(min_weight))
            weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-10)
        return weights

    def _check_patch_count(self, patch_attention: torch.Tensor) -> None:
        num_patches = int(patch_attention.shape[-1])
        if num_patches <= 0:
            raise ValueError(f"Need at least one patch token, got {num_patches}.")
