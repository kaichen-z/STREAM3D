from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Sequence

import torch

from streaming.backend.attention_metric.metric import ConditionMetricMode


class ViewConditionSelectionMode(StrEnum):
    """Finite set of supported view-condition selection modes."""

    TOKEN_VOTE = "token_vote"


@dataclass
class Stage2SelectionConfig:
    enabled: bool
    metric: ConditionMetricMode
    topk: int
    attention_layer: int
    attention_step: int
    jam_kappa: float
    candidate_batch_size: int
    final_stage2_weighting: bool
    patch_start: int
    patch_end: int

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.metric = ConditionMetricMode(self.metric)
        self.topk = int(self.topk)
        self.attention_layer = int(self.attention_layer)
        self.attention_step = int(self.attention_step)
        self.jam_kappa = float(self.jam_kappa)
        self.candidate_batch_size = int(self.candidate_batch_size)
        self.final_stage2_weighting = bool(self.final_stage2_weighting)
        self.patch_start = int(self.patch_start)
        self.patch_end = int(self.patch_end)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "metric": self.metric.value,
            "metric_score_name": self.metric.value_key,
            "topk": int(self.topk),
            "attention_layer": int(self.attention_layer),
            "attention_step": int(self.attention_step),
            "jam_kappa": float(self.jam_kappa),
            "candidate_batch_size": int(self.candidate_batch_size),
            "final_stage2_weighting": bool(self.final_stage2_weighting),
            "patch_start": int(self.patch_start),
            "patch_end": int(self.patch_end),
            "same_stage2_warmup_noise": True,
        }


@dataclass
class ViewConditionCacheConfig:
    selection_mode: ViewConditionSelectionMode
    metric: ConditionMetricMode
    attention_layer: int
    warmup_steps: int
    topk: int
    memory_depth: int
    token_vote_update_margin: float
    jam_kappa: float
    patch_start: int
    patch_end: int
    # Stage-1 selection strategy among the seen views (Task-30/42).
    #   "va_div"  -> distance-aware VA selection (DEFAULT since Task-43): greedy frame-distance MMR
    #                over the VA vote mass; de-clusters redundant viewpoints at long horizons.
    #   "vote"    -> top-k by token-vote evidence (previous default; clusters at large horizons)
    #   "random"  -> k random seen views (baseline)
    #   "diverse" -> coverage-aware: split the seen frame-index span (≈azimuth on the
    #                spiral) into k bins and take the highest-vote view per bin.
    selection_strategy: str = "va_div"
    selection_random_seed: int = 0
    # Task-42/43: distance-aware VA selection ("va_div"). Greedy frame-distance MMR over the VA vote mass:
    #   score_v = rel_norm_v - selection_div_lambda * max_{u in S}(1 - |gfi_v-gfi_u|/span).
    # frame-index == camera trajectory position on render_spiral_100 (angular dist ~1.8 deg/frame), so this
    # penalizes viewpoint-redundant (clustered) views. selection_div_lambda=0.1 is the Task-42 optimum;
    # lambda=0 reduces exactly to plain "vote" (Task-41).
    selection_div_lambda: float = 0.1

    def __post_init__(self) -> None:
        self.selection_mode = ViewConditionSelectionMode(self.selection_mode)
        self.metric = ConditionMetricMode(self.metric)
        self.attention_layer = int(self.attention_layer)
        self.warmup_steps = int(self.warmup_steps)
        self.topk = int(self.topk)
        self.memory_depth = int(self.memory_depth)
        self.token_vote_update_margin = float(self.token_vote_update_margin)
        self.jam_kappa = float(self.jam_kappa)
        self.patch_start = int(self.patch_start)
        self.patch_end = int(self.patch_end)
        self.selection_strategy = str(self.selection_strategy)
        self.selection_random_seed = int(self.selection_random_seed)
        self.selection_div_lambda = float(self.selection_div_lambda)

    def to_metadata(self) -> Dict[str, Any]:
        metadata = asdict(self)
        metadata["selection_mode"] = self.selection_mode.value
        metadata["metric"] = self.metric.value
        metadata["metric_name"] = self.metric.value
        metadata["metric_score_name"] = self.metric.value_key
        metadata["bundle_size"] = int(self.topk)
        metadata["memory_depth"] = int(self.memory_depth)
        if self.metric is ConditionMetricMode.ENTROPY:
            metadata["metric_score"] = "1_minus_normalized_entropy"
        metadata["same_warmup_noise"] = True
        return metadata


@dataclass
class TokenVoteMemory:
    score: Optional[torch.Tensor] = None
    global_frame_index: Optional[torch.Tensor] = None
    view_records: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ViewScoreBatch:
    scores: torch.Tensor
    global_frame_indices: torch.Tensor
    view_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SelectorVoteResult:
    memory: TokenVoteMemory
    selected_views: List[Dict[str, Any]]


@dataclass
class ViewConditionCacheState:
    token_memory: TokenVoteMemory = field(default_factory=TokenVoteMemory)
    fixed_initial_noise: Any = None
    fixed_latent_shape_spec: Any = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class Stage1SelectionResult:
    selected_views: List[Dict[str, Any]]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class Stage2SelectionResult:
    selected_views: List[Dict[str, Any]]
    warnings: List[str]
    candidate_pool_size: int
    attention_view_count: int
    collector_steps: List[int]
    view_vote_report: List[Dict[str, Any]] = field(default_factory=list)


class ViewConditionSelector(ABC):
    def __init__(
        self,
        config: ViewConditionCacheConfig,
        state: ViewConditionCacheState,
    ) -> None:
        self.config = config
        self.state = state

    @property
    @abstractmethod
    def needs_ss_warmup(self) -> bool:
        pass

    @abstractmethod
    def stage1_select(
        self,
        *,
        chunk_spec: Dict[str, Any],
        warmup_chunk_spec: Dict[str, Any],
        warmup: Optional[Dict[str, Any]],
        score_by_view: Optional[Dict[int, torch.Tensor]],
        frame_names: Sequence[str],
        chunk_index: int,
        chunk_name: str,
    ) -> Stage1SelectionResult:
        pass

    @abstractmethod
    def stage2_select(
        self,
        *,
        seen_chunk_specs: Sequence[Dict[str, Any]],
        chunk_index: int,
        chunk_name: str,
        score_batches: Sequence[ViewScoreBatch],
        attention_view_count: int,
        collector_steps: Sequence[int],
        warnings: Sequence[str],
        config: Stage2SelectionConfig,
    ) -> Stage2SelectionResult:
        pass
