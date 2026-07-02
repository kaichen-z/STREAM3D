from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import backend_sam3d
from .attention_metric import (
    AttentionMetricFactory,
    ConditionMetricMode,
    stage1_score_by_view_from_warmup,
)
from .selector.cache import (
    build_seen_view_pool,
    build_selected_runtime_view_spec,
    make_view_condition_cache_state,
    to_jsonable,
    tree_shape_spec,
)
from .selector.selector_vote import (
    iter_stage2_candidate_batches,
    stage2_view_score_batch_from_attention_scores,
    view_score_batch_from_score_by_view,
)
from ..data.chunks import (
    build_chunk_info,
    build_prev_view_index_map,
    build_unique_warmup_chunk_spec,
    load_images_and_masks_for_chunk,
)
from streaming.utils.streaming_da3 import build_chunk_da3_result, load_da3
from streaming.utils.streaming_output import (
    build_chunk_crop_views,
    save_streaming_result_bundle,
)

from .config import (
    StreamingBackendConfig,
    SelectedChunkPlanContext,
    ChunkResultArtifacts,
    WarmupResult,
    SelectedRuntime,
)
from .selector import build_view_condition_selector


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
        return value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _runtime_seed(context: SelectedChunkPlanContext) -> int | None:
    return None if context.args.shared_chunk_rng_stream else int(context.args.seed)


def _build_source_chunk(chunk_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_index": int(chunk_spec["chunk_index"]),
        "chunk_name": str(chunk_spec["chunk_name"]),
        "image_paths": [
            str(Path(path).resolve()) for path in chunk_spec["frame_paths"]
        ],
        "image_names": [Path(path).name for path in chunk_spec["frame_paths"]],
        "global_frame_indices": [
            int(index) for index in chunk_spec["global_frame_indices"]
        ],
        "num_views": len(chunk_spec["stems"]),
    }


def save_stage1_sparse_outputs(
    output_dir: Path,
    *,
    stage1: Any,
    selected_views: Sequence[Dict[str, Any]],
    runtime_chunk_spec: Dict[str, Any],
    profile: Any,
) -> List[str]:
    sparse_dir = output_dir / "stage1_sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, np.ndarray] = {}
    for key in (
        "translation",
        "rotation",
        "scale",
        "downsample_factor",
        "pointmap_scale",
        "pointmap_shift",
        "coords",
    ):
        value = stage1.ss_return_dict.get(key)
        if value is not None:
            params[key] = _as_numpy(value)
    np.savez(sparse_dir / "params.npz", **params)

    pose_metadata = {}
    for key in ("scale", "rotation", "translation"):
        value = stage1.ss_return_dict.get(key)
        if value is not None:
            pose_metadata[key] = _as_numpy(value).tolist()

    metadata = to_jsonable(
        {
            "params_npz": "params.npz",
            "params_keys": sorted(params.keys()),
            "generated_coord_count": int(stage1.coords.shape[0]),
            "pose": pose_metadata,
            "selected_views": list(selected_views),
            "runtime_frame_keys": list(runtime_chunk_spec["stems"]),
            "global_frame_indices": [
                int(index) for index in runtime_chunk_spec["global_frame_indices"]
            ],
            "profile": profile.payload,
        }
    )
    (sparse_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [
        str(Path("stage1_sparse") / "params.npz"),
        str(Path("stage1_sparse") / "metadata.json"),
    ]


def _build_selected_chunk_plan_context(
    args: StreamingBackendConfig,
    example: StreamingExample,
    execution_plan: Any,
) -> SelectedChunkPlanContext:
    scene_da3 = load_da3(
        example.da3_root.resolve(),
        image_files=example.image_files,
        camera_pose_source=args.camera_pose_source,
        dataset_camera_path=args.dataset_camera_path,
    )
    pipeline = backend_sam3d.load_pipeline(args.model_config_path, compile_model=False)

    if args.shared_chunk_rng_stream:
        torch.manual_seed(int(args.seed))

    cache_config = args.cache
    stage2_selection_config = args.stage2_selection
    cache_state = make_view_condition_cache_state(cache_config)
    view_condition_selector = build_view_condition_selector(cache_config, cache_state)

    return SelectedChunkPlanContext(
        args=args,
        example=example,
        execution_plan=execution_plan,
        image_files=list(example.image_files),
        mask_root=example.mask_root.resolve(),
        scene_da3=scene_da3,
        pipeline=pipeline,
        cache_config=cache_config,
        stage2_selection_config=stage2_selection_config,
        cache_state=cache_state,
        view_condition_selector=view_condition_selector,
        reconstruction_indices={
            int(chunk_spec["chunk_index"])
            for chunk_spec in execution_plan.reconstruction_plan
        },
    )


def _update_cache_state_from_warmup(
    context: SelectedChunkPlanContext,
    warmup: Dict[str, Any],
) -> None:
    if context.cache_state.fixed_initial_noise is None:
        context.cache_state.fixed_initial_noise = warmup["initial_noise"]
        context.cache_state.fixed_latent_shape_spec = tree_shape_spec(
            warmup["latent_shape_spec"]
        )
        return

    current_latent_shape = tree_shape_spec(warmup["latent_shape_spec"])
    if current_latent_shape != context.cache_state.fixed_latent_shape_spec:
        raise ValueError(
            "SS warmup latent shape changed across chunks: "
            f"{context.cache_state.fixed_latent_shape_spec} vs {current_latent_shape}"
        )


def _prepare_warmup_result(
    context: SelectedChunkPlanContext,
    chunk_spec: Dict[str, Any],
) -> WarmupResult:
    chunk_name = str(chunk_spec["chunk_name"])
    chunk_index = int(chunk_spec["chunk_index"])
    source_frame_keys = list(chunk_spec["stems"])
    warmup_chunk_spec = chunk_spec
    warmup_frame_keys = source_frame_keys
    skipped_duplicate_global_indices: list[int] = []
    warmup_profile = backend_sam3d.new_run_profile()
    warmup: Dict[str, Any] | None = None
    warmup_images = None
    warmup_masks = None
    warmup_da3 = None
    prepared_warmup = None
    attention_count = 0
    cache_warnings: list[str] = []
    selection_warnings: list[str] = []
    selection_metadata: Dict[str, Any] = {}

    warmup_chunk_spec, skipped_duplicate_global_indices = (
        build_unique_warmup_chunk_spec(
            chunk_spec,
            seen_global_frame_indices=context.evaluated_warmup_global_indices,
        )
    )
    warmup_frame_keys = list(warmup_chunk_spec["stems"])
    if not warmup_frame_keys:
        raise ValueError(
            f"{chunk_name}: no new warmup frames remain after skipping "
            "duplicate overlap frames."
        )
    warmup_images, warmup_masks = load_images_and_masks_for_chunk(
        warmup_chunk_spec["frame_paths"],
        context.mask_root,
        warmup_frame_keys,
    )
    warmup_da3 = build_chunk_da3_result(
        context.scene_da3,
        global_frame_indices=warmup_chunk_spec["global_frame_indices"],
    )
    prepared_warmup = backend_sam3d.prepare_views(
        context.pipeline,
        view_images=warmup_images,
        view_masks=warmup_masks,
        view_pointmaps=warmup_da3["pointmaps_sam3d"],
        runtime_frame_keys=warmup_frame_keys,
        seed=_runtime_seed(context),
        mode=str(context.args.pipeline["mode"]),
        profile=warmup_profile,
    )
    warmup = backend_sam3d.collect_ss_warmup_attentions(
        pipeline=context.pipeline,
        prepared=prepared_warmup,
        inference_steps=int(context.args.pipeline["stage1_inference_steps"]),
        warmup_steps=context.cache_config.warmup_steps,
        attention_layer=context.cache_config.attention_layer,
        fixed_initial_noise=context.cache_state.fixed_initial_noise,
    )
    _update_cache_state_from_warmup(context, warmup)

    attention_count = len(warmup["attention_scores"])
    if attention_count != len(warmup_frame_keys):
        cache_warnings.append(
            f"{chunk_name}: collected {attention_count}/{len(warmup_frame_keys)} "
            "view attention tensors."
        )
    score_by_view = stage1_score_by_view_from_warmup(
        warmup=warmup,
        metric=context.cache_config.metric,
        patch_start=context.cache_config.patch_start,
        patch_end=context.cache_config.patch_end,
        kappa=context.cache_config.jam_kappa,
    )
    selection = context.view_condition_selector.stage1_select(
        chunk_spec=chunk_spec,
        warmup_chunk_spec=warmup_chunk_spec,
        warmup=warmup,
        score_by_view=score_by_view,
        frame_names=warmup_frame_keys,
        chunk_index=chunk_index,
        chunk_name=chunk_name,
    )
    selected_views = [dict(view) for view in selection.selected_views]
    selection_warnings = list(selection.warnings)
    selection_metadata = dict(selection.metadata)

    context.evaluated_warmup_global_indices.update(
        int(index) for index in warmup_chunk_spec["global_frame_indices"]
    )

    cache_warnings.extend(selection_warnings)
    return WarmupResult(
        chunk_name=chunk_name,
        chunk_index=chunk_index,
        warmup_chunk_spec=warmup_chunk_spec,
        warmup_frame_keys=warmup_frame_keys,
        selected_views=selected_views,
        selection_warnings=selection_warnings,
        selection_metadata=selection_metadata,
        warmup=warmup,
        warmup_profile=warmup_profile,
        attention_count=attention_count,
        cache_warnings=cache_warnings,
        skipped_duplicate_global_indices=skipped_duplicate_global_indices,
        source_chunk=_build_source_chunk(chunk_spec),
        prepared_warmup=prepared_warmup,
        warmup_images=warmup_images,
        warmup_masks=warmup_masks,
        warmup_da3=warmup_da3,
    )


def _build_selected_runtime(
    context: SelectedChunkPlanContext,
    *,
    selected_views: Sequence[Dict[str, Any]],
    chunk_name: str,
    chunk_index: int,
) -> SelectedRuntime:
    selected_views = [dict(view) for view in selected_views]
    runtime_chunk_spec = build_selected_runtime_view_spec(
        image_files=context.image_files,
        selected_views=selected_views,
        chunk_index=chunk_index,
        chunk_name=chunk_name,
    )
    loaded_image_names = list(runtime_chunk_spec["stems"])
    runtime_frame_keys = list(runtime_chunk_spec["stems"])
    view_images, view_masks = load_images_and_masks_for_chunk(
        runtime_chunk_spec["frame_paths"],
        context.mask_root,
        runtime_frame_keys,
    )
    chunk_da3 = build_chunk_da3_result(
        context.scene_da3,
        global_frame_indices=runtime_chunk_spec["global_frame_indices"],
    )
    prev_view_index_map = None
    if context.prev_loaded_image_names is not None:
        prev_view_index_map = build_prev_view_index_map(
            prev_loaded_image_names=context.prev_loaded_image_names,
            current_loaded_image_names=loaded_image_names,
        )
    output_dir = context.args.output_root / context.example.object_name / chunk_name
    crop_views = build_chunk_crop_views(
        chunk_input={
            "view_images": view_images,
            "view_masks": view_masks,
        },
        chunk_info={
            "loaded_image_names": loaded_image_names,
        },
    )
    pipeline_kwargs = dict(context.args.pipeline)
    stage2_weighting = dict(pipeline_kwargs.pop("stage2_weighting"))

    return SelectedRuntime(
        chunk_name=chunk_name,
        chunk_index=int(chunk_index),
        selected_views=selected_views,
        runtime_chunk_spec=runtime_chunk_spec,
        loaded_image_names=loaded_image_names,
        runtime_frame_keys=runtime_frame_keys,
        view_images=view_images,
        view_masks=view_masks,
        chunk_da3=chunk_da3,
        crop_views=crop_views,
        prev_view_index_map=prev_view_index_map,
        output_dir=output_dir,
        pipeline_kwargs=pipeline_kwargs,
        stage2_weighting=stage2_weighting,
    )


def _build_warmup_prefix_record(
    context: SelectedChunkPlanContext,
    warmup_result: WarmupResult,
) -> Dict[str, Any]:
    if warmup_result.warmup is None:
        raise RuntimeError("TOKEN_VOTE warmup metadata is missing.")
    return {
        "chunk_index": int(warmup_result.chunk_index),
        "chunk_name": str(warmup_result.chunk_name),
        "global_frame_indices": [
            int(index)
            for index in warmup_result.warmup_chunk_spec["global_frame_indices"]
        ],
        "num_views": len(warmup_result.warmup_frame_keys),
        "attention_view_count": int(warmup_result.attention_count),
        "collector_step": int(warmup_result.warmup["collector_step"]),
        "selected_views": [dict(view) for view in warmup_result.selected_views],
        "source_global_frame_indices": [
            int(index) for index in warmup_result.source_chunk["global_frame_indices"]
        ],
        "skipped_duplicate_global_frame_indices": [
            int(index) for index in warmup_result.skipped_duplicate_global_indices
        ],
    }


def _build_cache_metadata(
    context: SelectedChunkPlanContext,
    warmup_result: WarmupResult,
    *,
    stage1_selected_views: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    cache_metadata: Dict[str, Any] = {
        "enabled": True,
        "selection_mode": context.cache_config.selection_mode.value,
        "camera_pose_source": context.scene_da3.get("camera_pose_source"),
        "dataset_camera_path": context.scene_da3.get("dataset_camera_path"),
        "depth_metric_scale": context.scene_da3.get("depth_metric_scale"),
        "depth_metric_scale_source": context.scene_da3.get("depth_metric_scale_source"),
        "depth_metric_scale_audit": context.scene_da3.get("depth_metric_scale_audit"),
        "pointmap_metric_scale": context.scene_da3.get("pointmap_metric_scale"),
        "mode": (
            "single_chunk_prefix_warmup"
            if context.execution_plan.fast_single_chunk
            else "selected_chunk_warmup"
        ),
        "config": context.cache_config.to_metadata(),
        "requested_chunk_indices": context.execution_plan.requested_chunk_indices,
        "resolved_chunk_indices": context.execution_plan.resolved_chunk_indices,
        "warmup_chunk": warmup_result.source_chunk,
        "selected_views": [dict(view) for view in stage1_selected_views],
        "stage1_selected_views": [dict(view) for view in stage1_selected_views],
        "warnings": list(warmup_result.cache_warnings),
    }

    if warmup_result.warmup is None:
        raise RuntimeError("TOKEN_VOTE warmup metadata is missing.")
    cache_metadata["warmup_profile"] = warmup_result.warmup_profile.payload
    cache_metadata["attention_view_count"] = int(warmup_result.attention_count)
    cache_metadata["collector_step"] = int(warmup_result.warmup["collector_step"])
    cache_metadata["evaluated_warmup_chunk"] = {
        "chunk_index": int(warmup_result.chunk_index),
        "chunk_name": str(warmup_result.chunk_name),
        "global_frame_indices": [
            int(index)
            for index in warmup_result.warmup_chunk_spec["global_frame_indices"]
        ],
        "num_views": len(warmup_result.warmup_frame_keys),
        "skipped_duplicate_global_frame_indices": [
            int(index) for index in warmup_result.skipped_duplicate_global_indices
        ],
    }

    if context.execution_plan.fast_single_chunk:
        cache_metadata["warmup_prefix"] = {
            "target_chunk": {
                "chunk_index": int(warmup_result.chunk_index),
                "chunk_name": str(warmup_result.chunk_name),
            },
            "chunk_indices": [
                int(record["chunk_index"]) for record in context.warmup_prefix
            ],
            "chunk_names": [
                str(record["chunk_name"]) for record in context.warmup_prefix
            ],
            "num_chunks": len(context.warmup_prefix),
            "chunks": context.warmup_prefix,
        }
    return cache_metadata


def _save_selected_chunk_bundle(
    context: SelectedChunkPlanContext,
    *,
    warmup_result: WarmupResult,
    stage1_runtime: SelectedRuntime,
    final_runtime: SelectedRuntime,
    artifacts: ChunkResultArtifacts,
    is_final_chunk: bool,
) -> Path:
    chunk_info = build_chunk_info(
        chunk_spec=final_runtime.runtime_chunk_spec,
        loaded_image_names=final_runtime.loaded_image_names,
        runtime_frame_keys=final_runtime.runtime_frame_keys,
        prev_view_index_map=final_runtime.prev_view_index_map,
    )
    chunk_info["source_chunk"] = warmup_result.source_chunk
    chunk_info["view_condition_cache"] = to_jsonable(
        _build_cache_metadata(
            context,
            warmup_result,
            stage1_selected_views=stage1_runtime.selected_views,
        )
    )
    if artifacts.stage2_selection_metadata is not None:
        chunk_info["stage2_view_selection"] = artifacts.stage2_selection_metadata
    if artifacts.stage1_sparse_files:
        chunk_info["stage1_sparse"] = {
            "saved_outputs": artifacts.stage1_sparse_files,
        }

    stage2_crop_views = build_chunk_crop_views(
        chunk_input={
            "view_images": final_runtime.view_images,
            "view_masks": final_runtime.view_masks,
        },
        chunk_info={
            "loaded_image_names": final_runtime.loaded_image_names,
        },
    )

    save_streaming_result_bundle(
        output_dir=final_runtime.output_dir,
        chunk_info=chunk_info,
        crop_views=stage2_crop_views,
        stage1_crop_views=stage1_runtime.crop_views,
        stage2_crop_views=stage2_crop_views,
        mask_prompt=context.example.object_name,
        result=artifacts.outputs,
        seed=_runtime_seed(context),
        stage1_steps=final_runtime.pipeline_kwargs["stage1_inference_steps"],
        stage2_steps=final_runtime.pipeline_kwargs["stage2_inference_steps"],
        decode_formats=final_runtime.pipeline_kwargs["decode_formats"],
        weighting_metadata=artifacts.stage2_weighting_metadata,
        prev_chunk_id=context.prev_reconstructed_chunk_index,
        status_label="definitive" if is_final_chunk else "intermediate",
    )

    context.prev_loaded_image_names = list(final_runtime.loaded_image_names)
    context.prev_reconstructed_chunk_index = int(warmup_result.chunk_index)
    if is_final_chunk:
        context.final_result_dir = final_runtime.output_dir
    return final_runtime.output_dir


def _run_stage2_disabled_chunk(
    context: SelectedChunkPlanContext,
    *,
    warmup_result: WarmupResult,
    stage1_runtime: SelectedRuntime,
) -> tuple[SelectedRuntime, ChunkResultArtifacts]:
    weighting_config = backend_sam3d.build_stage2_weighting_config(
        stage1_runtime.stage2_weighting
    )
    stage1_runtime.pipeline_kwargs.update(
        view_images=stage1_runtime.view_images,
        view_masks=stage1_runtime.view_masks,
        view_pointmaps=stage1_runtime.chunk_da3["pointmaps_sam3d"],
        runtime_frame_keys=stage1_runtime.runtime_frame_keys,
        seed=_runtime_seed(context),
        weighting_config=weighting_config,
    )
    outputs = backend_sam3d.run_multi_view(
        context.pipeline, **stage1_runtime.pipeline_kwargs
    )
    return (
        stage1_runtime,
        ChunkResultArtifacts(
            outputs=outputs,
            stage2_weighting_metadata=backend_sam3d.build_stage2_weighting_metadata(
                weighting_config
            ),
            stage2_selection_metadata=None,
            stage1_sparse_files=[],
        ),
    )


def _run_stage2_enabled_chunk(
    context: SelectedChunkPlanContext,
    *,
    warmup_result: WarmupResult,
    stage1_runtime: SelectedRuntime,
) -> tuple[SelectedRuntime, ChunkResultArtifacts]:
    total_start = time.perf_counter()
    stage1_profile = backend_sam3d.new_run_profile(
        context.pipeline,
        num_views=len(stage1_runtime.view_images),
        stage1_inference_steps=stage1_runtime.pipeline_kwargs["stage1_inference_steps"],
        stage2_inference_steps=stage1_runtime.pipeline_kwargs["stage2_inference_steps"],
    )
    stage1_profile.payload["stage1_selection_num_views"] = len(
        stage1_runtime.selected_views
    )
    stage1_profile.payload["stage1_preprocess"] = stage1_profile.payload.get(
        "preprocess"
    )
    stage1_profile.payload["stage1"] = stage1_profile.payload.get("stage1")

    stage1_prepared = backend_sam3d.prepare_views(
        context.pipeline,
        view_images=stage1_runtime.view_images,
        view_masks=stage1_runtime.view_masks,
        view_pointmaps=stage1_runtime.chunk_da3["pointmaps_sam3d"],
        runtime_frame_keys=stage1_runtime.runtime_frame_keys,
        seed=_runtime_seed(context),
        mode=str(stage1_runtime.pipeline_kwargs["mode"]),
        profile=stage1_profile,
    )
    stage1_outputs = backend_sam3d.run_stage1(
        context.pipeline,
        prepared=stage1_prepared,
        stage1_inference_steps=stage1_runtime.pipeline_kwargs["stage1_inference_steps"],
        use_stage1_distillation=bool(
            stage1_runtime.pipeline_kwargs["use_stage1_distillation"]
        ),
        mode=str(stage1_runtime.pipeline_kwargs["mode"]),
        ss_weighting=bool(stage1_runtime.pipeline_kwargs["ss_weighting"]),
        ss_attention_layer=int(stage1_runtime.pipeline_kwargs["ss_attention_layer"]),
        ss_weight_source=str(stage1_runtime.pipeline_kwargs["ss_weight_source"]),
        ss_jam_alpha=float(stage1_runtime.pipeline_kwargs["ss_jam_alpha"]),
        ss_jam_kappa=float(stage1_runtime.pipeline_kwargs["ss_jam_kappa"]),
        ss_uniform_blend=float(stage1_runtime.pipeline_kwargs["ss_uniform_blend"]),
        ss_min_weight=float(stage1_runtime.pipeline_kwargs["ss_min_weight"]),
        ss_warmup_steps=int(stage1_runtime.pipeline_kwargs["ss_warmup_steps"]),
        ss_patch_start=int(stage1_runtime.pipeline_kwargs["ss_patch_start"]),
        ss_patch_end=int(stage1_runtime.pipeline_kwargs["ss_patch_end"]),
        profile=stage1_profile,
    )
    stage1_sparse_files = save_stage1_sparse_outputs(
        stage1_runtime.output_dir,
        stage1=stage1_outputs,
        selected_views=stage1_runtime.selected_views,
        runtime_chunk_spec=stage1_runtime.runtime_chunk_spec,
        profile=stage1_profile,
    )

    stage2_candidate_pool = build_seen_view_pool(context.seen_chunk_specs)
    stage2_score_batches = []
    stage2_patch_mass_by_global_index: dict[int, torch.Tensor] = {}
    stage2_entropy_confidence_by_global_index: dict[int, torch.Tensor] = {}
    fixed_stage2_initial_noise = None
    stage2_attention_count = 0
    stage2_selection_warnings: list[str] = []
    stage2_collector_steps: list[int] = []
    stage2_selection_metric = ConditionMetricMode(
        context.stage2_selection_config.metric
    )
    stage2_metric = AttentionMetricFactory.build(stage2_selection_metric)
    for candidate_batch in iter_stage2_candidate_batches(
        stage2_candidate_pool,
        batch_size=context.stage2_selection_config.candidate_batch_size,
    ):
        candidate_runtime = _build_selected_runtime(
            context,
            selected_views=candidate_batch,
            chunk_name=warmup_result.chunk_name,
            chunk_index=warmup_result.chunk_index,
        )
        candidate_profile = backend_sam3d.new_run_profile()
        candidate_prepared = backend_sam3d.prepare_views(
            context.pipeline,
            view_images=candidate_runtime.view_images,
            view_masks=candidate_runtime.view_masks,
            view_pointmaps=candidate_runtime.chunk_da3["pointmaps_sam3d"],
            runtime_frame_keys=candidate_runtime.runtime_frame_keys,
            seed=_runtime_seed(context),
            mode=str(stage1_runtime.pipeline_kwargs["mode"]),
            profile=candidate_profile,
        )
        candidate_attention = backend_sam3d.collect_stage2_candidate_attention_batch(
            pipeline=context.pipeline,
            prepared=candidate_prepared,
            candidates=candidate_batch,
            coords=stage1_outputs.coords,
            stage2_inference_steps=stage1_runtime.pipeline_kwargs[
                "stage2_inference_steps"
            ],
            use_stage2_distillation=bool(
                stage1_runtime.pipeline_kwargs["use_stage2_distillation"]
            ),
            config=context.stage2_selection_config,
            fixed_initial_noise=fixed_stage2_initial_noise,
        )
        if fixed_stage2_initial_noise is None:
            fixed_stage2_initial_noise = candidate_attention.initial_noise
        if stage2_selection_metric is ConditionMetricMode.MASS_RELATIVE:
            for view_idx, scores in sorted(
                candidate_attention.attention_scores_by_view.items()
            ):
                global_frame_index = int(
                    candidate_batch[int(view_idx)]["global_frame_index"]
                )
                patch_mass, entropy_confidence = stage2_metric.components(
                    scores,
                    patch_start=context.stage2_selection_config.patch_start,
                    patch_end=context.stage2_selection_config.patch_end,
                )
                stage2_patch_mass_by_global_index[global_frame_index] = patch_mass
                stage2_entropy_confidence_by_global_index[global_frame_index] = (
                    entropy_confidence
                )
        else:
            stage2_score_batches.append(
                stage2_view_score_batch_from_attention_scores(
                    attention_scores_by_view=candidate_attention.attention_scores_by_view,
                    candidates=candidate_batch,
                    metric=context.stage2_selection_config.metric,
                    patch_start=context.stage2_selection_config.patch_start,
                    patch_end=context.stage2_selection_config.patch_end,
                    kappa=context.stage2_selection_config.jam_kappa,
                )
            )
        stage2_attention_count += int(candidate_attention.attention_count)
        stage2_collector_steps.append(int(candidate_attention.collector_step))
        if int(candidate_attention.attention_count) != len(candidate_batch):
            stage2_selection_warnings.append(
                f"{warmup_result.chunk_name}: collected Stage2 attention for "
                f"{int(candidate_attention.attention_count)}/{len(candidate_batch)} "
                "candidate views."
            )
        del candidate_runtime
        del candidate_prepared

    if stage2_selection_metric is ConditionMetricMode.MASS_RELATIVE:
        evidence_by_global_index = stage2_metric.from_components(
            stage2_patch_mass_by_global_index,
            stage2_entropy_confidence_by_global_index,
            kappa=float(context.stage2_selection_config.jam_kappa),
        )
        global_frame_indices = sorted(int(index) for index in evidence_by_global_index)
        record_by_global_index = {
            int(view["global_frame_index"]): dict(view)
            for view in stage2_candidate_pool
        }
        stage2_score_batches.append(
            view_score_batch_from_score_by_view(
                score_by_view={
                    position: evidence_by_global_index[global_frame_index]
                    for position, global_frame_index in enumerate(global_frame_indices)
                },
                global_frame_indices=global_frame_indices,
                view_records=[
                    record_by_global_index[global_frame_index]
                    for global_frame_index in global_frame_indices
                ],
            )
        )

    stage2_selection = context.view_condition_selector.stage2_select(
        seen_chunk_specs=context.seen_chunk_specs,
        chunk_index=warmup_result.chunk_index,
        chunk_name=warmup_result.chunk_name,
        score_batches=stage2_score_batches,
        attention_view_count=stage2_attention_count,
        collector_steps=stage2_collector_steps,
        warnings=stage2_selection_warnings,
        config=context.stage2_selection_config,
    )

    final_runtime = _build_selected_runtime(
        context,
        selected_views=stage2_selection.selected_views,
        chunk_name=warmup_result.chunk_name,
        chunk_index=warmup_result.chunk_index,
    )
    final_profile = backend_sam3d.new_run_profile(
        context.pipeline,
        num_views=len(final_runtime.view_images),
        stage1_inference_steps=stage1_runtime.pipeline_kwargs["stage1_inference_steps"],
        stage2_inference_steps=stage1_runtime.pipeline_kwargs["stage2_inference_steps"],
    )
    final_profile.payload["stage1_selection_num_views"] = len(
        stage1_runtime.selected_views
    )
    final_profile.payload["stage1_preprocess"] = stage1_profile.payload.get(
        "preprocess"
    )
    final_profile.payload["stage1"] = stage1_profile.payload.get("stage1")
    final_profile.payload["stage2_selection"] = {
        "candidate_pool_size": int(stage2_selection.candidate_pool_size),
        "attention_view_count": int(stage2_selection.attention_view_count),
        "collector_steps": stage2_selection.collector_steps,
    }
    final_runtime.pipeline_kwargs.update(
        view_images=final_runtime.view_images,
        view_masks=final_runtime.view_masks,
        view_pointmaps=final_runtime.chunk_da3["pointmaps_sam3d"],
        runtime_frame_keys=final_runtime.runtime_frame_keys,
        seed=_runtime_seed(context),
        mode=str(stage1_runtime.pipeline_kwargs["mode"]),
        profile=final_profile,
    )
    weighting_config = backend_sam3d.build_stage2_weighting_config(
        stage1_runtime.stage2_weighting,
        force_disabled=not bool(context.stage2_selection_config.final_stage2_weighting),
    )
    final_prepared = backend_sam3d.prepare_views(
        context.pipeline,
        view_images=final_runtime.view_images,
        view_masks=final_runtime.view_masks,
        view_pointmaps=final_runtime.chunk_da3["pointmaps_sam3d"],
        runtime_frame_keys=final_runtime.runtime_frame_keys,
        seed=_runtime_seed(context),
        mode=str(stage1_runtime.pipeline_kwargs["mode"]),
        profile=final_profile,
    )
    stage2_outputs = backend_sam3d.run_stage2(
        context.pipeline,
        prepared=final_prepared,
        stage1=stage1_outputs,
        stage2_inference_steps=stage1_runtime.pipeline_kwargs["stage2_inference_steps"],
        use_stage2_distillation=bool(
            stage1_runtime.pipeline_kwargs["use_stage2_distillation"]
        ),
        mode=str(stage1_runtime.pipeline_kwargs["mode"]),
        weighting_config=weighting_config,
        save_stage2_init=bool(stage1_runtime.pipeline_kwargs["save_stage2_init"]),
        save_stage2_init_path=stage1_runtime.pipeline_kwargs["save_stage2_init_path"],
        profile=final_profile,
    )
    outputs = backend_sam3d.assemble_result(
        context.pipeline,
        prepared=final_prepared,
        stage1=stage1_outputs,
        stage2=stage2_outputs,
        decode_formats=stage1_runtime.pipeline_kwargs["decode_formats"],
        with_mesh_postprocess=bool(
            stage1_runtime.pipeline_kwargs["with_mesh_postprocess"]
        ),
        with_texture_baking=bool(stage1_runtime.pipeline_kwargs["with_texture_baking"]),
        use_vertex_color=bool(stage1_runtime.pipeline_kwargs["use_vertex_color"]),
        profile=final_profile,
        total_start=total_start,
    )
    stage2_selection_metadata = to_jsonable(
        {
            "enabled": bool(context.stage2_selection_config.enabled),
            "config": context.stage2_selection_config.to_metadata(),
            "selected_views": [dict(view) for view in stage2_selection.selected_views],
            "view_vote_report": [
                dict(view) for view in stage2_selection.view_vote_report
            ],
            "warnings": list(stage2_selection.warnings),
        }
    )
    stage2_weighting_metadata = backend_sam3d.build_stage2_weighting_metadata(
        weighting_config
    )
    return (
        final_runtime,
        ChunkResultArtifacts(
            outputs=outputs,
            stage2_weighting_metadata=stage2_weighting_metadata,
            stage2_selection_metadata=stage2_selection_metadata,
            stage1_sparse_files=stage1_sparse_files,
        ),
    )
