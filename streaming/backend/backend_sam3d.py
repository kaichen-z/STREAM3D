from __future__ import annotations

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils._pytree import tree_map_only

from streaming.backend.attention_metric import (
    AttentionWeightingConfig,
    ConditionMetricMode,
)


@dataclass
class Stage2CandidateAttentionBatch:
    attention_scores_by_view: Dict[int, torch.Tensor]
    initial_noise: Any
    latent_shape_spec: Any
    attention_count: int
    collector_step: int


def _sam3d_module(name: str) -> Any:
    return importlib.import_module(name)


def load_pipeline(model_config_path: Path, compile_model: bool = False) -> Any:
    inference_loader = _sam3d_module("sam3d_objects.pipeline.inference_loader")
    return inference_loader.load_pipeline_from_config(
        Path(model_config_path),
        compile=bool(compile_model),
    )


def new_run_profile(
    pipeline: Any | None = None,
    *,
    num_views: int | None = None,
    stage1_inference_steps: int | None = None,
    stage2_inference_steps: int | None = None,
) -> Any:
    stages = _sam3d_module("sam3d_objects.pipeline.run_multi_view_stages")
    if pipeline is None or num_views is None:
        return stages.RunProfile(payload={})
    return stages.initialize_run_profile(
        pipeline,
        num_views=int(num_views),
        stage1_inference_steps=stage1_inference_steps,
        stage2_inference_steps=stage2_inference_steps,
    )


def prepare_views(pipeline: Any, **kwargs: Any) -> Any:
    stages = _sam3d_module("sam3d_objects.pipeline.run_multi_view_stages")
    return stages.prepare_views(pipeline, **kwargs)


def run_stage1(pipeline: Any, **kwargs: Any) -> Any:
    stages = _sam3d_module("sam3d_objects.pipeline.run_multi_view_stages")
    return stages.run_stage1(pipeline, **kwargs)


def run_stage2(pipeline: Any, **kwargs: Any) -> Any:
    stages = _sam3d_module("sam3d_objects.pipeline.run_multi_view_stages")
    return stages.run_stage2(pipeline, **kwargs)


def assemble_result(pipeline: Any, **kwargs: Any) -> Any:
    stages = _sam3d_module("sam3d_objects.pipeline.run_multi_view_stages")
    return stages.assemble_result(pipeline, **kwargs)


def run_multi_view(pipeline: Any, **kwargs: Any) -> Any:
    return pipeline.run_multi_view(**kwargs)


def build_stage2_weighting_config(
    stage2_weighting: Dict[str, Any],
    force_disabled: bool = False,
) -> Any | None:
    stage2_weighting = dict(stage2_weighting)
    stage2_weighting_enabled = bool(stage2_weighting.pop("enabled", True))
    if force_disabled or not stage2_weighting_enabled:
        return None

    weighting_config = AttentionWeightingConfig(**stage2_weighting)
    return weighting_config if weighting_config.weight_metrics else None


def build_stage2_weighting_metadata(weighting_config: Any | None) -> Dict[str, Any]:
    if weighting_config is None:
        return {"stage2_weighting": {"enabled": False}}

    payload = {
        key: value
        for key, value in vars(weighting_config).items()
        if key != "weight_metrics"
    }
    weight_modes = [metric.metric for metric in weighting_config.weight_metrics]
    weight_sources = [mode.value for mode in weight_modes]
    payload["metric"] = (
        weight_sources[0] if len(weight_sources) == 1 else weight_sources
    )
    if len(weight_sources) > 1:
        payload["weight_normalization"] = "average_of_normalized_weight_sources"
    if ConditionMetricMode.JOINT_ATTENTION_MASS in weight_modes:
        payload["jam_weight_normalization"] = "evidence_sum"
    if ConditionMetricMode.ENTROPY in weight_modes:
        payload["entropy_weight_normalization"] = "confidence_sum"
    return _to_jsonable({"stage2_weighting": {"enabled": True, **payload}})


def collect_ss_warmup_attentions(
    *,
    pipeline: Any,
    prepared: Any,
    inference_steps: int,
    warmup_steps: int,
    attention_layer: int,
    fixed_initial_noise: Optional[Any],
) -> Dict[str, Any]:
    weighting = _sam3d_module("sam3d_objects.pipeline.multi_view_weighting")
    ss_generator = pipeline.models["ss_generator"]
    image = prepared.view_ss_input_dicts[0]["image"]
    batch_size = int(image.shape[0])
    num_views = int(prepared.num_views)

    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = pipeline.ss_cfg_strength
    ss_generator.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm

    previous_steps = ss_generator.inference_steps
    previous_generate_noise = ss_generator._generate_noise
    ss_generator.inference_steps = int(inference_steps)
    latent_shape_spec = build_latent_shape_spec(pipeline, batch_size)
    active_warmup_steps = min(int(warmup_steps), int(ss_generator.inference_steps))
    if active_warmup_steps <= 0:
        active_warmup_steps = 1

    if fixed_initial_noise is None:
        initial_noise = previous_generate_noise(latent_shape_spec, image.device)
        initial_noise = _extract_generator_state_like(initial_noise, latent_shape_spec)
    else:
        if tree_shape_spec(fixed_initial_noise) != tree_shape_spec(latent_shape_spec):
            raise ValueError(
                f"SS latent shape changed across chunks: expected "
                f"{tree_shape_spec(fixed_initial_noise)}, got {tree_shape_spec(latent_shape_spec)}."
            )
        initial_noise = _extract_generator_state_like(
            fixed_initial_noise, latent_shape_spec
        )

    condition_args, condition_kwargs = pipeline.get_multi_view_condition_input(
        pipeline.condition_embedders["ss_condition_embedder"],
        prepared.view_ss_input_dicts,
        pipeline.ss_condition_input_mapping,
    )
    collector = weighting.SSAttentionCollector(
        num_views=num_views,
        target_layer=int(attention_layer),
    )

    def fixed_noise_generator(shape: Any, device: torch.device) -> Any:
        if tree_shape_spec(shape) != tree_shape_spec(latent_shape_spec):
            raise ValueError(
                f"SS latent shape changed within warmup: expected {latent_shape_spec}, got {shape}"
            )
        return move_tree_to_device(initial_noise, device)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype)
        if torch.cuda.is_available()
        else nullcontext()
    )

    try:
        ss_generator.inference_steps = active_warmup_steps
        ss_generator._generate_noise = fixed_noise_generator
        with torch.no_grad():
            with autocast_context:
                with weighting.inject_ss_generator_with_collector(
                    ss_generator,
                    num_views=num_views,
                    num_steps=active_warmup_steps,
                    attention_collector=collector,
                ):
                    _ = ss_generator(
                        latent_shape_spec,
                        image.device,
                        *condition_args,
                        **condition_kwargs,
                    )
    finally:
        ss_generator._generate_noise = previous_generate_noise
        ss_generator.inference_steps = previous_steps

    attentions = collector.get_attentions()
    return {
        "attentions": attentions,
        "attention_scores": collector.get_attention_scores(),
        "initial_noise": _detach_cpu_tree(initial_noise),
        "latent_shape_spec": tree_shape_spec(latent_shape_spec),
        "collector_step": int(collector._current_step),
    }


def collect_stage2_candidate_attention_batch(
    *,
    pipeline: Any,
    prepared: Any,
    candidates: Sequence[Dict[str, Any]],
    coords: torch.Tensor,
    stage2_inference_steps: Optional[int],
    use_stage2_distillation: bool,
    config: Any,
    fixed_initial_noise: Optional[Any],
) -> Stage2CandidateAttentionBatch:
    weighting = _sam3d_module("sam3d_objects.pipeline.multi_view_weighting")
    image = prepared.view_slat_input_dicts[0]["image"]
    device = image.device
    slat_generator = pipeline.models["slat_generator"]
    num_views = int(prepared.num_views)
    latent_shape = (int(image.shape[0]), int(coords.shape[0]), 8)

    previous_steps = slat_generator.inference_steps
    previous_generate_noise = slat_generator._generate_noise
    if stage2_inference_steps:
        slat_generator.inference_steps = int(stage2_inference_steps)
    warmup_steps = max(1, int(config.attention_step) + 1)

    if use_stage2_distillation:
        slat_generator.no_shortcut = False
        slat_generator.reverse_fn.strength = 0
    else:
        slat_generator.no_shortcut = True
        slat_generator.reverse_fn.strength = pipeline.slat_cfg_strength

    if fixed_initial_noise is None:
        initial_noise = previous_generate_noise(latent_shape, device)
        initial_noise = _detach_cpu_tree(initial_noise)
    else:
        if tree_shape_spec(fixed_initial_noise) != tree_shape_spec(latent_shape):
            raise ValueError(
                f"Stage2 warmup latent shape changed: expected "
                f"{tree_shape_spec(fixed_initial_noise)}, got {tree_shape_spec(latent_shape)}."
            )
        initial_noise = fixed_initial_noise

    def fixed_noise_generator(shape: Any, fixed_device: torch.device) -> Any:
        if tree_shape_spec(shape) != tree_shape_spec(latent_shape):
            raise ValueError(
                f"Stage2 warmup latent shape changed within scoring: expected "
                f"{tree_shape_spec(latent_shape)}, got {tree_shape_spec(shape)}."
            )
        return move_tree_to_device(initial_noise, fixed_device)

    condition_args, condition_kwargs = pipeline.get_multi_view_condition_input(
        pipeline.condition_embedders["slat_condition_embedder"],
        prepared.view_slat_input_dicts,
        pipeline.slat_condition_input_mapping,
    )
    condition_args += (coords.cpu().numpy(),)

    collector = weighting.AttentionCollector(
        num_views=num_views,
        target_layer=int(config.attention_layer),
        target_step=int(config.attention_step),
        patch_start=int(config.patch_start),
        patch_end=int(config.patch_end),
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=pipeline.dtype)
        if torch.cuda.is_available()
        else nullcontext()
    )

    try:
        slat_generator.inference_steps = warmup_steps
        slat_generator._generate_noise = fixed_noise_generator
        with torch.no_grad():
            with autocast_context:
                with weighting.inject_generator_multi_view_with_collector(
                    slat_generator,
                    num_views=num_views,
                    num_steps=warmup_steps,
                    attention_collector=collector,
                ):
                    _ = slat_generator(
                        latent_shape,
                        device,
                        *condition_args,
                        **condition_kwargs,
                    )
    finally:
        slat_generator._generate_noise = previous_generate_noise
        slat_generator.inference_steps = previous_steps

    attention_scores = collector.get_attention_scores()
    return Stage2CandidateAttentionBatch(
        attention_scores_by_view={
            int(view_idx): scores
            for view_idx, scores in sorted(attention_scores.items())
        },
        initial_noise=_detach_cpu_tree(initial_noise),
        latent_shape_spec=tree_shape_spec(latent_shape),
        attention_count=int(len(attention_scores)),
        collector_step=int(collector._current_step),
    )


def tree_shape_spec(tree: Any) -> Any:
    if torch.is_tensor(tree):
        return tuple(int(dim) for dim in tree.shape)
    if isinstance(tree, int):
        return int(tree)
    if isinstance(tree, torch.Size):
        return tuple(int(dim) for dim in tree)
    if isinstance(tree, dict):
        return {str(key): tree_shape_spec(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_shape_spec(value) for value in tree)
    if isinstance(tree, list):
        return [tree_shape_spec(value) for value in tree]
    raise TypeError(f"Unsupported noise tree leaf: {type(tree)!r}")


def build_latent_shape_spec(pipeline: Any, batch_size: int) -> Any:
    ss_generator = pipeline.models["ss_generator"]
    if pipeline.is_mm_dit():
        return {
            key: (batch_size,) + (value.pos_emb.shape[0], value.input_layer.in_features)
            for key, value in ss_generator.reverse_fn.backbone.latent_mapping.items()
        }
    return (batch_size, 4096, 8)


def move_tree_to_device(tree: Any, device: torch.device) -> Any:
    return tree_map_only(torch.Tensor, lambda tensor: tensor.to(device), tree)


def build_canonical_to_reference_cv_affine(**kwargs: Any) -> Any:
    transforms = _sam3d_module("sam3d_objects.utils.coordinate_transforms")
    return transforms.build_canonical_to_reference_cv_affine(**kwargs)


def gaussian_class() -> Any:
    module = _sam3d_module(
        "sam3d_objects.model.backbone.tdfy_dit.representations.gaussian"
    )
    return module.Gaussian


def gaussian_renderer_class() -> Any:
    module = _sam3d_module(
        "sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render"
    )
    return module.GaussianRenderer


def _detach_cpu_tree(tree: Any) -> Any:
    return tree_map_only(
        torch.Tensor,
        lambda tensor: tensor.detach().cpu().to(torch.float32).contiguous(),
        tree,
    )


def _extract_generator_state_like(state: Any, shape_spec: Any) -> Any:
    if torch.is_tensor(state):
        return state.detach().cpu().to(torch.float32).contiguous()

    is_shape_leaf = isinstance(shape_spec, (list, tuple)) and all(
        isinstance(item, int) for item in shape_spec
    )
    if is_shape_leaf:
        if not torch.is_tensor(state):
            raise TypeError(
                f"Expected tensor state for generator leaf spec {tuple(shape_spec)}, got {type(state)!r}."
            )
        return state.detach().cpu().to(torch.float32).contiguous()

    if isinstance(shape_spec, dict):
        if not isinstance(state, dict):
            raise TypeError(
                f"Expected dict generator state matching keys {list(shape_spec.keys())}, got {type(state)!r}."
            )
        return {
            key: _extract_generator_state_like(state[key], child_spec)
            for key, child_spec in shape_spec.items()
        }

    if isinstance(shape_spec, tuple):
        if not isinstance(state, (list, tuple)) or len(state) != len(shape_spec):
            raise TypeError(
                f"Expected tuple/list generator state of length {len(shape_spec)}, got {type(state)!r}."
            )
        return tuple(
            _extract_generator_state_like(child_state, child_spec)
            for child_state, child_spec in zip(state, shape_spec)
        )

    if isinstance(shape_spec, list):
        if not isinstance(state, (list, tuple)) or len(state) != len(shape_spec):
            raise TypeError(
                f"Expected list/tuple generator state of length {len(shape_spec)}, got {type(state)!r}."
            )
        return [
            _extract_generator_state_like(child_state, child_spec)
            for child_state, child_spec in zip(state, shape_spec)
        ]

    raise TypeError(f"Unsupported generator shape spec type: {type(shape_spec)!r}")


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
