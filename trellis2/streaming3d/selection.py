from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from types import MethodType
from typing import Any, Iterator
import inspect
import math
import random as _random

import torch
from PIL import Image


class SelectionMethod(StrEnum):
    TRELLIS_ATTENTION = "trellis_attention"
    MASS_RELATIVE = "mass_relative"
    TOKEN_VOTE = "token_vote"
    RANDOM = "random"
    # Task-43: distance-aware VA selection — token-vote evidence + frame-distance MMR de-clustering.
    # Default Trellis selection policy. lambda=0 reduces exactly to TOKEN_VOTE.
    VA_DIV = "va_div"
    FEATURE_PROXY = "feature_proxy"


@dataclass(frozen=True)
class SelectionConfig:
    topk: int = 8
    method: SelectionMethod = SelectionMethod.TRELLIS_ATTENTION
    warmup_steps: int = 1
    q_chunk_size: int = 128
    attention_layer: int = 6
    jam_kappa: float = 1.0
    memory_depth: int = 1
    update_margin: float = 0.0
    # Task-43: diversity strength for VA_DIV selection (frame-distance MMR). 0.1 is the Task-42 optimum;
    # 0.0 makes VA_DIV identical to TOKEN_VOTE.
    selection_div_lambda: float = 0.1
    random_seed: int = 0


@dataclass(frozen=True)
class ViewScore:
    view_index: int
    joint_attention_mass: float | None = None
    mass_relative: float | None = None
    trellis_feature_energy: float | None = None


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: list[int]
    candidate_indices: list[int]
    scores: list[ViewScore]
    metadata: dict[str, Any]


@dataclass
class TokenVoteMemory:
    evidence: torch.Tensor | None = None
    view_index: torch.Tensor | None = None
    view_records: dict[int, dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.view_records is None:
            self.view_records = {}


class CrossAttentionMassCollector:
    def __init__(
        self,
        view_indices: list[int],
        token_spans: list[tuple[int, int]],
        q_chunk_size: int,
    ):
        self.view_indices = view_indices
        self.token_spans = token_spans
        self.q_chunk_size = q_chunk_size
        self.total_mass = torch.zeros(len(view_indices), dtype=torch.float64)
        self.records: list[dict[str, Any]] = []

    def record(self, module_name: str, module: Any, x: torch.Tensor, context: Any) -> None:
        if not isinstance(context, torch.Tensor):
            return
        if context.ndim != 3 or context.shape[0] != 1:
            return
        if context.shape[1] < self.token_spans[-1][1]:
            return

        with torch.no_grad():
            q = module.to_q(x)
            kv = module.to_kv(context)
            batch_size, num_queries, _ = q.shape
            num_context_tokens = context.shape[1]
            q = q.reshape(batch_size, num_queries, module.num_heads, -1)
            kv = kv.reshape(batch_size, num_context_tokens, 2, module.num_heads, -1)
            k, _ = kv.unbind(dim=2)
            if module.qk_rms_norm:
                q = module.q_rms_norm(q)
                k = module.k_rms_norm(k)

            q = q.float().permute(0, 2, 1, 3)
            k = k.float().permute(0, 2, 1, 3)
            scale = 1.0 / math.sqrt(q.shape[-1])
            layer_mass = torch.zeros(len(self.view_indices), device=q.device, dtype=torch.float64)
            normalizer = q.shape[0] * q.shape[1] * q.shape[2]

            for start in range(0, q.shape[2], self.q_chunk_size):
                q_chunk = q[:, :, start:start + self.q_chunk_size, :]
                logits = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
                probs = torch.softmax(logits, dim=-1)
                token_mass = probs.sum(dim=(0, 1, 2)).double()
                for view_offset, (span_start, span_end) in enumerate(self.token_spans):
                    layer_mass[view_offset] += token_mass[span_start:span_end].sum()

            layer_mass = layer_mass.cpu() / normalizer
            self.total_mass += layer_mass
            self.records.append(
                {
                    "module": module_name,
                    "num_query_tokens": int(num_queries),
                    "num_context_tokens": int(num_context_tokens),
                    "mass_by_view": {
                        str(view_index): float(layer_mass[offset])
                        for offset, view_index in enumerate(self.view_indices)
                    },
                }
            )

    def scores(self) -> list[ViewScore]:
        if not self.records:
            raise RuntimeError("No TRELLIS dense cross-attention records were collected")
        normalizer = float(self.total_mass.sum())
        if normalizer <= 0:
            raise RuntimeError("Collected TRELLIS cross-attention mass is zero")
        normalized = self.total_mass / normalizer
        return [
            ViewScore(view_index=view_index, joint_attention_mass=float(normalized[offset]))
            for offset, view_index in enumerate(self.view_indices)
        ]


class MassRelativeAttentionCollector:
    def __init__(
        self,
        view_indices: list[int],
        token_spans: list[tuple[int, int]],
        q_chunk_size: int,
        attention_layer: int,
        kappa: float,
    ):
        self.view_indices = view_indices
        self.token_spans = token_spans
        self.q_chunk_size = q_chunk_size
        self.attention_layer = attention_layer
        self.kappa = kappa
        self.total_evidence = torch.zeros(len(view_indices), dtype=torch.float64)
        self.total_votes = torch.zeros(len(view_indices), dtype=torch.int64)
        self.total_evidence_by_query: torch.Tensor | None = None
        self.num_query_tokens = 0
        self.records: list[dict[str, Any]] = []

    def record(self, module_name: str, module: Any, x: torch.Tensor, context: Any) -> None:
        if module_name != f"blocks.{self.attention_layer}.cross_attn":
            return
        if not isinstance(context, torch.Tensor):
            return
        if context.ndim != 3 or context.shape[0] != 1:
            return
        if context.shape[1] < self.token_spans[-1][1]:
            return

        with torch.no_grad():
            q = module.to_q(x)
            kv = module.to_kv(context)
            batch_size, num_queries, _ = q.shape
            num_context_tokens = context.shape[1]
            q = q.reshape(batch_size, num_queries, module.num_heads, -1)
            kv = kv.reshape(batch_size, num_context_tokens, 2, module.num_heads, -1)
            k, _ = kv.unbind(dim=2)
            if module.qk_rms_norm:
                q = module.q_rms_norm(q)
                k = module.k_rms_norm(k)

            q = q.float().permute(0, 2, 1, 3)
            k = k.float().permute(0, 2, 1, 3)
            scale = 1.0 / math.sqrt(q.shape[-1])
            layer_evidence = torch.zeros(len(self.view_indices), device=q.device, dtype=torch.float64)
            layer_evidence_by_query = torch.empty(
                len(self.view_indices),
                q.shape[2],
                device=q.device,
                dtype=torch.float64,
            )
            layer_votes = torch.zeros(len(self.view_indices), device=q.device, dtype=torch.int64)

            for start in range(0, q.shape[2], self.q_chunk_size):
                q_chunk = q[:, :, start:start + self.q_chunk_size, :]
                logits = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
                probs = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
                view_mass = torch.empty(
                    len(self.view_indices),
                    probs.shape[0],
                    device=q.device,
                    dtype=torch.float32,
                )
                view_confidence = torch.empty_like(view_mass)
                for view_offset, (span_start, span_end) in enumerate(self.token_spans):
                    span_probs = probs[:, span_start:span_end]
                    mass = span_probs.sum(dim=-1)
                    distribution = span_probs / mass.clamp_min(1e-12).unsqueeze(-1)
                    entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=-1)
                    entropy = entropy / math.log(span_end - span_start)
                    view_mass[view_offset] = mass
                    view_confidence[view_offset] = 1.0 - entropy

                relative_mass = view_mass - view_mass.mean(dim=0, keepdim=True)
                evidence = (1.0 + self.kappa * relative_mass).clamp_min(1e-6) * view_confidence
                layer_evidence += evidence.double().sum(dim=1)
                layer_evidence_by_query[:, start:start + q_chunk.shape[2]] = evidence.double()
                layer_votes += evidence.argmax(dim=0).bincount(
                    minlength=len(self.view_indices)
                ).to(layer_votes.device)

            self.total_evidence += layer_evidence.cpu()
            if self.total_evidence_by_query is None:
                self.total_evidence_by_query = layer_evidence_by_query.cpu()
            else:
                self.total_evidence_by_query += layer_evidence_by_query.cpu()
            self.total_votes += layer_votes.cpu()
            self.num_query_tokens += int(num_queries)
            self.records.append(
                {
                    "module": module_name,
                    "num_query_tokens": int(num_queries),
                    "num_context_tokens": int(num_context_tokens),
                    "kappa": float(self.kappa),
                    "evidence_by_view": {
                        str(view_index): float(layer_evidence[offset].detach().cpu())
                        for offset, view_index in enumerate(self.view_indices)
                    },
                    "token_votes_by_view": {
                        str(view_index): int(layer_votes[offset].detach().cpu())
                        for offset, view_index in enumerate(self.view_indices)
                    },
                }
            )

    def scores(self) -> list[ViewScore]:
        if not self.records:
            raise RuntimeError(
                f"No TRELLIS cross-attention records were collected at sparse-structure layer {self.attention_layer}"
            )
        normalizer = float(self.total_evidence.sum())
        if normalizer <= 0:
            raise RuntimeError("Collected TRELLIS mass_relative evidence is zero")
        normalized = self.total_evidence / normalizer
        return [
            ViewScore(view_index=view_index, mass_relative=float(normalized[offset]))
            for offset, view_index in enumerate(self.view_indices)
        ]


@contextmanager
def collect_dense_cross_attention(
    model: torch.nn.Module,
    collector: CrossAttentionMassCollector | MassRelativeAttentionCollector,
) -> Iterator[None]:
    from trellis2.modules.attention.modules import MultiHeadAttention

    originals = []
    for module_name, module in model.named_modules():
        if not isinstance(module, MultiHeadAttention) or module._type != "cross":
            continue
        original_forward = module.forward

        def wrapped_forward(self, x, context=None, phases=None, *, _name=module_name, _forward=original_forward):
            collector.record(_name, self, x, context)
            return _forward(x, context=context, phases=phases)

        module.forward = MethodType(wrapped_forward, module)
        originals.append((module, original_forward))

    if not originals:
        raise RuntimeError(f"No TRELLIS dense cross-attention modules found in {type(model).__name__}")
    try:
        yield
    finally:
        for module, original_forward in originals:
            module.forward = original_forward


def select_views(
    pipeline: Any,
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
) -> SelectionResult:
    _validate_inputs(candidate_indices, candidate_images, config)
    if config.method == SelectionMethod.RANDOM:
        return select_views_random(candidate_indices, config)
    if config.method == SelectionMethod.FEATURE_PROXY:
        return select_views_by_feature_proxy(pipeline, candidate_indices, candidate_images, config)
    if config.method == SelectionMethod.TRELLIS_ATTENTION:
        return select_views_by_trellis_attention(pipeline, candidate_indices, candidate_images, config)
    if config.method == SelectionMethod.MASS_RELATIVE:
        return select_views_by_trellis_mass_relative(pipeline, candidate_indices, candidate_images, config)
    if config.method in (SelectionMethod.TOKEN_VOTE, SelectionMethod.VA_DIV):
        return update_token_vote_selection(
            pipeline,
            candidate_indices,
            candidate_images,
            config,
            TokenVoteMemory(),
        )[1]
    raise ValueError(f"Unsupported selection method: {config.method}")


def _va_div_reselect(report: list[dict[str, Any]], k: int, div_lambda: float) -> list[int]:
    """Task-43 distance-aware VA selection for the TRELLIS token-vote path. Greedy frame-distance MMR over
    the VA vote mass (token_count): score_v = rel_norm_v - lambda * max_{u in S}(1 - |fi_v-fi_u|/win),
    win = span/k. `view_index` == camera trajectory frame index (≈azimuth on the spiral), so the penalty
    de-clusters viewpoint-redundant picks. Mirrors the SAM3D selector_vote._va_div_select. Returns the
    selected view_index ints (ascending). lambda=0 ⇒ plain top-k vote."""
    rows = list(report)
    if len(rows) <= k:
        return [int(r["view_index"]) for r in rows[:k]]
    fi = {id(r): int(r["view_index"]) for r in rows}
    rel = {id(r): float(r.get("token_count", 0) or 0.0) for r in rows}
    rmax = max(rel.values()) or 1.0
    pool = [fi[id(r)] for r in rows]
    span = max(max(pool) - min(pool), 1)
    win = max(span / float(max(k, 1)), 1.0)
    def overlap(a: dict, b: dict) -> float:
        return max(0.0, 1.0 - abs(fi[id(a)] - fi[id(b)]) / win)
    chosen = [sorted(rows, key=lambda r: (-rel[id(r)], fi[id(r)]))[0]]
    chosen_ids = {id(chosen[0])}
    while len(chosen) < k:
        best, best_s = None, float("-inf")
        for r in rows:
            if id(r) in chosen_ids:
                continue
            omax = max(overlap(r, c) for c in chosen)
            s = rel[id(r)] / rmax - float(div_lambda) * omax
            if s > best_s or (s == best_s and (best is None or fi[id(r)] < fi[id(best)])):
                best_s, best = s, r
        if best is None:
            break
        chosen.append(best)
        chosen_ids.add(id(best))
    return [fi[id(r)] for r in sorted(chosen, key=lambda r: fi[id(r)])[:k]]


def update_token_vote_selection(
    pipeline: Any,
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
    memory: TokenVoteMemory,
) -> tuple[TokenVoteMemory, SelectionResult]:
    _validate_inputs(candidate_indices, candidate_images, config)
    scores, evidence_by_view, metadata = collect_mass_relative_evidence(
        pipeline,
        candidate_indices,
        candidate_images,
        config,
        method_name="trellis_cross_attention_token_vote",
    )
    updated_memory = update_token_vote_memory(
        memory,
        candidate_indices=candidate_indices,
        evidence_by_view=evidence_by_view,
        scores=scores,
        memory_depth=config.memory_depth,
        update_margin=config.update_margin,
    )
    report = token_vote_report(updated_memory)
    # Task-43: VA_DIV re-ranks the vote report with frame-distance MMR (de-clustering); TOKEN_VOTE keeps
    # plain count-descending top-k. lambda=0 makes the two identical.
    if config.method == SelectionMethod.VA_DIV and float(config.selection_div_lambda) > 0.0:
        selected = _va_div_reselect(report, int(config.topk), float(config.selection_div_lambda))
        selection_rule = "va_div_frame_distance_mmr"
    else:
        selected = [int(row["view_index"]) for row in report[:config.topk]]
        selection_rule = "token_vote_count_desc"
    result_metadata = {
        **metadata,
        "method": str(config.method),
        "score_name": "mass_relative",
        "selection_rule": selection_rule,
        "selection_div_lambda": float(config.selection_div_lambda),
        "memory_depth": config.memory_depth,
        "update_margin": config.update_margin,
        "view_vote_report": report,
    }
    return updated_memory, SelectionResult(
        selected_indices=selected,
        candidate_indices=list(candidate_indices),
        scores=scores,
        metadata=result_metadata,
    )


def select_views_random(
    candidate_indices: list[int],
    config: SelectionConfig,
) -> SelectionResult:
    start_time = perf_counter()
    k = min(int(config.topk), len(candidate_indices))
    if len(candidate_indices) <= k:
        selected = list(candidate_indices)
        reason = "candidate_count_le_topk"
    else:
        selected = sorted(_random.Random(int(config.random_seed)).sample(candidate_indices, k))
        reason = "seeded_sample"
    scores = [
        ViewScore(view_index=int(view_index))
        for view_index in candidate_indices
    ]
    return SelectionResult(
        selected_indices=[int(index) for index in selected],
        candidate_indices=[int(index) for index in candidate_indices],
        scores=scores,
        metadata={
            "method": "random",
            "selection_rule": "seeded_random_sample",
            "reason": reason,
            "topk": int(config.topk),
            "random_seed": int(config.random_seed),
            "elapsed_seconds": perf_counter() - start_time,
        },
    )


def collect_mass_relative_evidence(
    pipeline: Any,
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
    *,
    method_name: str,
) -> tuple[list[ViewScore], dict[int, torch.Tensor], dict[str, Any]]:
    start_time = perf_counter()
    tokens = _get_image_tokens(pipeline, candidate_images)
    num_views, tokens_per_view, channels = tokens.shape
    token_spans = [
        (view_offset * tokens_per_view, (view_offset + 1) * tokens_per_view)
        for view_offset in range(num_views)
    ]
    context = tokens.reshape(1, num_views * tokens_per_view, channels)
    flow_model = pipeline.models["sparse_structure_flow_model"]
    resolution = flow_model.resolution
    noise = torch.randn(
        1,
        flow_model.in_channels,
        resolution,
        resolution,
        resolution,
        device=pipeline.device,
    )
    sampler_kwargs = {
        **pipeline.sparse_structure_sampler_params,
        "steps": config.warmup_steps,
        "verbose": False,
        "tqdm_desc": "Streaming3D TRELLIS token-vote warmup",
    }
    sampler_signature = inspect.signature(pipeline.sparse_structure_sampler.sample)
    if "neg_cond" in sampler_signature.parameters:
        sampler_kwargs["neg_cond"] = torch.zeros_like(tokens[:1])
    if "guidance_strength" in sampler_signature.parameters:
        sampler_kwargs["guidance_strength"] = 1.0

    collector = MassRelativeAttentionCollector(
        candidate_indices,
        token_spans,
        config.q_chunk_size,
        config.attention_layer,
        config.jam_kappa,
    )
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    try:
        with collect_dense_cross_attention(flow_model, collector):
            pipeline.sparse_structure_sampler.sample(
                flow_model,
                noise,
                cond=context,
                **sampler_kwargs,
            )
    finally:
        if pipeline.low_vram:
            flow_model.cpu()

    scores = collector.scores()
    if collector.total_evidence_by_query is None:
        raise RuntimeError("No token-level TRELLIS mass_relative evidence was collected")
    evidence_by_view = {
        int(view_index): collector.total_evidence_by_query[offset].to(torch.float32).contiguous()
        for offset, view_index in enumerate(candidate_indices)
    }
    metadata = {
        "method": method_name,
        "score_name": "mass_relative",
        "topk": config.topk,
        "warmup_steps": config.warmup_steps,
        "attention_layer": config.attention_layer,
        "jam_kappa": config.jam_kappa,
        "tokens_shape": list(tokens.shape),
        "q_chunk_size": config.q_chunk_size,
        "collector_records": collector.records,
        "token_vote_counts": {
            str(view_index): int(collector.total_votes[offset])
            for offset, view_index in enumerate(candidate_indices)
        },
        "elapsed_seconds": perf_counter() - start_time,
    }
    return scores, evidence_by_view, metadata


def select_views_by_trellis_attention(
    pipeline: Any,
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
) -> SelectionResult:
    start_time = perf_counter()
    tokens = _get_image_tokens(pipeline, candidate_images)
    if len(candidate_indices) <= config.topk:
        scores = [
            ViewScore(view_index=view_index, joint_attention_mass=1.0 / len(candidate_indices))
            for view_index in candidate_indices
        ]
        return SelectionResult(
            selected_indices=list(candidate_indices),
            candidate_indices=list(candidate_indices),
            scores=scores,
            metadata={
                "method": "trellis_cross_attention_joint_mass",
                "score_name": "joint_attention_mass",
                "reason": "candidate_count_le_topk",
                "topk": config.topk,
                "warmup_steps": 0,
                "tokens_shape": list(tokens.shape),
                "collector_records": [],
                "elapsed_seconds": perf_counter() - start_time,
            },
        )

    num_views, tokens_per_view, channels = tokens.shape
    token_spans = [
        (view_offset * tokens_per_view, (view_offset + 1) * tokens_per_view)
        for view_offset in range(num_views)
    ]
    context = tokens.reshape(1, num_views * tokens_per_view, channels)
    flow_model = pipeline.models["sparse_structure_flow_model"]
    resolution = flow_model.resolution
    noise = torch.randn(
        1,
        flow_model.in_channels,
        resolution,
        resolution,
        resolution,
        device=pipeline.device,
    )
    sampler_kwargs = {
        **pipeline.sparse_structure_sampler_params,
        "steps": config.warmup_steps,
        "verbose": False,
        "tqdm_desc": "Streaming3D TRELLIS attention warmup",
    }
    sampler_signature = inspect.signature(pipeline.sparse_structure_sampler.sample)
    if "neg_cond" in sampler_signature.parameters:
        sampler_kwargs["neg_cond"] = torch.zeros_like(tokens[:1])
    if "guidance_strength" in sampler_signature.parameters:
        sampler_kwargs["guidance_strength"] = 1.0

    collector = CrossAttentionMassCollector(candidate_indices, token_spans, config.q_chunk_size)
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    try:
        with collect_dense_cross_attention(flow_model, collector):
            pipeline.sparse_structure_sampler.sample(
                flow_model,
                noise,
                cond=context,
                **sampler_kwargs,
            )
    finally:
        if pipeline.low_vram:
            flow_model.cpu()

    scores = collector.scores()
    selected = _select_topk(scores, config.topk, "joint_attention_mass")
    return SelectionResult(
        selected_indices=selected,
        candidate_indices=list(candidate_indices),
        scores=scores,
        metadata={
            "method": "trellis_cross_attention_joint_mass",
            "score_name": "joint_attention_mass",
            "topk": config.topk,
            "warmup_steps": config.warmup_steps,
            "tokens_shape": list(tokens.shape),
            "q_chunk_size": config.q_chunk_size,
            "collector_records": collector.records,
            "elapsed_seconds": perf_counter() - start_time,
        },
    )


def select_views_by_trellis_mass_relative(
    pipeline: Any,
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
) -> SelectionResult:
    start_time = perf_counter()
    tokens = _get_image_tokens(pipeline, candidate_images)
    if len(candidate_indices) <= config.topk:
        scores = [
            ViewScore(view_index=view_index, mass_relative=1.0 / len(candidate_indices))
            for view_index in candidate_indices
        ]
        return SelectionResult(
            selected_indices=list(candidate_indices),
            candidate_indices=list(candidate_indices),
            scores=scores,
            metadata={
                "method": "trellis_cross_attention_mass_relative",
                "score_name": "mass_relative",
                "reason": "candidate_count_le_topk",
                "topk": config.topk,
                "warmup_steps": 0,
                "attention_layer": config.attention_layer,
                "jam_kappa": config.jam_kappa,
                "tokens_shape": list(tokens.shape),
                "collector_records": [],
                "elapsed_seconds": perf_counter() - start_time,
            },
        )

    num_views, tokens_per_view, channels = tokens.shape
    token_spans = [
        (view_offset * tokens_per_view, (view_offset + 1) * tokens_per_view)
        for view_offset in range(num_views)
    ]
    context = tokens.reshape(1, num_views * tokens_per_view, channels)
    flow_model = pipeline.models["sparse_structure_flow_model"]
    resolution = flow_model.resolution
    noise = torch.randn(
        1,
        flow_model.in_channels,
        resolution,
        resolution,
        resolution,
        device=pipeline.device,
    )
    sampler_kwargs = {
        **pipeline.sparse_structure_sampler_params,
        "steps": config.warmup_steps,
        "verbose": False,
        "tqdm_desc": "Streaming3D TRELLIS mass_relative warmup",
    }
    sampler_signature = inspect.signature(pipeline.sparse_structure_sampler.sample)
    if "neg_cond" in sampler_signature.parameters:
        sampler_kwargs["neg_cond"] = torch.zeros_like(tokens[:1])
    if "guidance_strength" in sampler_signature.parameters:
        sampler_kwargs["guidance_strength"] = 1.0

    collector = MassRelativeAttentionCollector(
        candidate_indices,
        token_spans,
        config.q_chunk_size,
        config.attention_layer,
        config.jam_kappa,
    )
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    try:
        with collect_dense_cross_attention(flow_model, collector):
            pipeline.sparse_structure_sampler.sample(
                flow_model,
                noise,
                cond=context,
                **sampler_kwargs,
            )
    finally:
        if pipeline.low_vram:
            flow_model.cpu()

    scores = collector.scores()
    selected = _select_topk(scores, config.topk, "mass_relative")
    return SelectionResult(
        selected_indices=selected,
        candidate_indices=list(candidate_indices),
        scores=scores,
        metadata={
            "method": "trellis_cross_attention_mass_relative",
            "score_name": "mass_relative",
            "topk": config.topk,
            "warmup_steps": config.warmup_steps,
            "attention_layer": config.attention_layer,
            "jam_kappa": config.jam_kappa,
            "tokens_shape": list(tokens.shape),
            "q_chunk_size": config.q_chunk_size,
            "collector_records": collector.records,
            "token_vote_counts": {
                str(view_index): int(collector.total_votes[offset])
                for offset, view_index in enumerate(candidate_indices)
            },
            "elapsed_seconds": perf_counter() - start_time,
        },
    )


def update_token_vote_memory(
    memory: TokenVoteMemory,
    *,
    candidate_indices: list[int],
    evidence_by_view: dict[int, torch.Tensor],
    scores: list[ViewScore],
    memory_depth: int,
    update_margin: float,
) -> TokenVoteMemory:
    memory_depth = int(memory_depth)
    if memory_depth <= 0:
        raise ValueError("memory_depth must be positive")
    if not evidence_by_view:
        return memory

    views = [int(index) for index in candidate_indices]
    num_tokens = int(evidence_by_view[views[0]].numel())
    if memory.evidence is None:
        evidence = torch.full((num_tokens, memory_depth), float("-inf"), dtype=torch.float32)
        view_index = torch.full((num_tokens, memory_depth), -1, dtype=torch.long)
    else:
        evidence = memory.evidence.detach().cpu().to(torch.float32).clone()
        view_index = memory.view_index.detach().cpu().to(torch.long).clone()
        if tuple(evidence.shape) != (num_tokens, memory_depth):
            raise ValueError(
                f"Token-vote memory shape changed: expected {tuple(evidence.shape)}, "
                f"got {(num_tokens, memory_depth)}."
            )

    score_by_view = {int(score.view_index): score for score in scores}
    view_records = {int(index): dict(record) for index, record in memory.view_records.items()}
    candidate_values = [evidence]
    candidate_frames = [view_index]
    for view_idx in views:
        metric_values = evidence_by_view[view_idx].detach().cpu().to(torch.float32).flatten().contiguous()
        if int(metric_values.numel()) != num_tokens:
            raise ValueError(
                f"Token-vote evidence shape changed for view {view_idx}: "
                f"expected {num_tokens}, got {int(metric_values.numel())}."
            )
        score = score_by_view[view_idx]
        view_records[view_idx] = {
            "view_index": view_idx,
            "mass_relative": score.mass_relative,
        }
        candidate_values.append(metric_values[:, None])
        candidate_frames.append(torch.full((num_tokens, 1), view_idx, dtype=torch.long))

    combined_values = torch.cat(candidate_values, dim=1)
    combined_frames = torch.cat(candidate_frames, dim=1)
    updated_values = torch.full((num_tokens, memory_depth), float("-inf"), dtype=torch.float32)
    updated_frames = torch.full((num_tokens, memory_depth), -1, dtype=torch.long)
    for token_idx in range(num_tokens):
        best_by_view: dict[int, float] = {}
        for slot_idx in range(int(combined_frames.shape[1])):
            view_idx = int(combined_frames[token_idx, slot_idx].item())
            if view_idx < 0:
                continue
            value = float(combined_values[token_idx, slot_idx].item())
            old_value = best_by_view.get(view_idx)
            if old_value is None or value > old_value + float(update_margin):
                best_by_view[view_idx] = value

        ranked = sorted(
            best_by_view.items(),
            key=lambda item: (-item[1], int(item[0])),
        )[:memory_depth]
        for rank, (view_idx, value) in enumerate(ranked):
            updated_values[token_idx, rank] = float(value)
            updated_frames[token_idx, rank] = int(view_idx)

    return TokenVoteMemory(
        evidence=updated_values.contiguous(),
        view_index=updated_frames.contiguous(),
        view_records=view_records,
    )


def token_vote_report(memory: TokenVoteMemory) -> list[dict[str, Any]]:
    if memory.evidence is None or memory.view_index is None:
        return []

    evidence = memory.evidence.detach().cpu().to(torch.float32)
    view_index = memory.view_index.detach().cpu().to(torch.long)
    valid_mask = view_index >= 0
    if not bool(valid_mask.any()):
        return []

    total_slots = int(valid_mask.sum().item())
    flat_views = view_index[valid_mask]
    report = []
    for view_idx in sorted(int(value) for value in flat_views.unique().tolist()):
        view_mask = view_index == view_idx
        view_metric = evidence[view_mask]
        record = memory.view_records[view_idx]
        token_count = int(view_mask.sum().item())
        report.append(
            {
                **record,
                "token_count": token_count,
                "token_fraction": float(token_count / total_slots),
                "mean_best_mass_relative": float(view_metric.mean().item()),
                "min_best_mass_relative": float(view_metric.min().item()),
                "max_best_mass_relative": float(view_metric.max().item()),
            }
        )

    report = sorted(
        report,
        key=lambda row: (
            -int(row["token_count"]),
            -float(row["mean_best_mass_relative"]),
            int(row["view_index"]),
        ),
    )
    return [
        {
            **row,
            "rank": int(rank),
        }
        for rank, row in enumerate(report, start=1)
    ]


def select_views_by_feature_proxy(
    pipeline: Any,
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
) -> SelectionResult:
    start_time = perf_counter()
    tokens = _get_image_tokens(pipeline, candidate_images)
    with torch.no_grad():
        token_norm = tokens.float().norm(dim=-1).mean(dim=1)
        token_variance = tokens.float().var(dim=1, unbiased=False).mean(dim=1)
        energy = token_norm + token_variance

    scores = [
        ViewScore(view_index=view_index, trellis_feature_energy=float(energy[offset].detach().cpu()))
        for offset, view_index in enumerate(candidate_indices)
    ]
    selected = _select_topk(scores, config.topk, "trellis_feature_energy")
    return SelectionResult(
        selected_indices=selected,
        candidate_indices=list(candidate_indices),
        scores=scores,
        metadata={
            "method": "trellis_feature_energy_attention_proxy",
            "score_name": "trellis_feature_energy",
            "is_attention_proxy": True,
            "scorer_description": "Explicit fallback scorer using TRELLIS image-token norm plus variance.",
            "topk": config.topk,
            "tokens_shape": list(tokens.shape),
            "elapsed_seconds": perf_counter() - start_time,
        },
    )


def _validate_inputs(
    candidate_indices: list[int],
    candidate_images: list[Image.Image],
    config: SelectionConfig,
) -> None:
    if len(candidate_indices) != len(candidate_images):
        raise ValueError("candidate_indices and candidate_images must have the same length")
    if not candidate_indices:
        raise ValueError("candidate_indices must be non-empty")
    if config.topk <= 0:
        raise ValueError("topk must be positive")
    if config.warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")
    if config.q_chunk_size <= 0:
        raise ValueError("q_chunk_size must be positive")
    if config.attention_layer < 0:
        raise ValueError("attention_layer must be non-negative")
    if config.memory_depth <= 0:
        raise ValueError("memory_depth must be positive")


def _get_image_tokens(pipeline: Any, images: list[Image.Image]) -> torch.Tensor:
    cond = pipeline.get_cond(images, 512, include_neg_cond=False)
    tokens = cond["cond"]
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise TypeError(f"Expected TRELLIS image tokens with shape [V, T, C], got {type(tokens)}")
    return tokens


def _select_topk(scores: list[ViewScore], topk: int, score_name: str) -> list[int]:
    ranked = sorted(scores, key=lambda score: getattr(score, score_name), reverse=True)
    return sorted(score.view_index for score in ranked[: min(topk, len(ranked))])
