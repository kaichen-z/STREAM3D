"""Staged helpers for :meth:`InferencePipeline.run_multi_view`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from loguru import logger


@dataclass
class RunProfile:
    payload: Dict[str, Any]


@dataclass
class PreparedViews:
    num_views: int
    view_ss_input_dicts: List[dict]
    view_slat_input_dicts: List[dict]
    raw_view_pointmaps: List[np.ndarray]
    runtime_frame_keys: Optional[List[str]]


@dataclass
class Stage1Outputs:
    ss_return_dict: Dict[str, Any]
    coords: torch.Tensor


@dataclass
class Stage2Outputs:
    slat: Any
    weight_manager: Optional[Any]


def initialize_run_profile(
    pipeline,
    *,
    num_views: int,
    stage1_inference_steps: Optional[int],
    stage2_inference_steps: Optional[int],
) -> RunProfile:
    return RunProfile(
        payload={
            "num_views": int(num_views),
            "stage1_inference_steps": None if stage1_inference_steps is None else int(stage1_inference_steps),
            "stage2_inference_steps": None if stage2_inference_steps is None else int(stage2_inference_steps),
            "streaming_reconstruction_mode": "view_condition_cache",
        }
    )


def prepare_views(
    pipeline,
    *,
    view_images: List[Union[np.ndarray, Image.Image]],
    view_masks: Optional[List[Optional[Union[None, np.ndarray, Image.Image]]]],
    view_pointmaps: Optional[List[Optional[np.ndarray]]],
    runtime_frame_keys: Optional[List[str]],
    seed: Optional[int],
    mode: Literal["stochastic", "multidiffusion"],
    profile: RunProfile,
) -> PreparedViews:
    num_views = len(view_images)
    preprocess_start = time.perf_counter()
    if view_masks is None:
        view_masks = [None] * num_views
    elif len(view_masks) != num_views:
        raise ValueError(f"Number of masks must match number of images: expected {num_views}, got {len(view_masks)}")

    if view_pointmaps is None:
        view_pointmaps = [None] * num_views
    elif len(view_pointmaps) != num_views:
        raise ValueError(f"Number of pointmaps must match number of images: expected {num_views}, got {len(view_pointmaps)}")
    else:
        logger.info(f"Using external pointmaps for {sum(1 for p in view_pointmaps if p is not None)}/{num_views} views")

    if seed is not None:
        torch.manual_seed(seed)

    logger.info(f"Running multi-view inference with {num_views} views, mode={mode}")
    if runtime_frame_keys is not None and len(runtime_frame_keys) != num_views:
        raise ValueError(
            f"runtime_frame_keys length mismatch: expected {num_views}, got {len(runtime_frame_keys)}"
        )

    view_ss_input_dicts: List[dict] = []
    view_slat_input_dicts: List[dict] = []
    raw_view_pointmaps: List[np.ndarray] = []
    for i, (image, mask, ext_pointmap) in enumerate(zip(view_images, view_masks, view_pointmaps)):
        logger.info(f"Preprocessing view {i+1}/{num_views}")
        if mask is not None:
            image = np.array(image) if isinstance(image, Image.Image) else np.array(image)
            mask = np.array(mask)
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            elif mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
            if mask.ndim == 2:
                mask = mask[..., None]
            if image.shape[-1] == 3:
                rgba_image = np.concatenate([image, mask], axis=-1).astype(np.uint8)
            elif image.shape[-1] == 4:
                rgba_image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            rgba_image = np.array(image) if isinstance(image, Image.Image) else np.array(image)

        rgba_image_pil = Image.fromarray(rgba_image)
        if hasattr(pipeline, "compute_pointmap"):
            if ext_pointmap is not None:
                logger.info(f"  View {i+1}: Using external pointmap, shape={ext_pointmap.shape}")
                ext_pointmap_tensor = torch.from_numpy(ext_pointmap).float()
                pointmap_dict = pipeline.compute_pointmap(rgba_image_pil, pointmap=ext_pointmap_tensor)
                pointmap = pointmap_dict["pointmap"]
            else:
                pointmap_dict = pipeline.compute_pointmap(rgba_image_pil, pointmap=None)
                pointmap = pointmap_dict["pointmap"]

            if pointmap is not None:
                pointmap_metric = pointmap.detach()
                if hasattr(type(pipeline), "_down_sample_img"):
                    pointmap_metric = type(pipeline)._down_sample_img(pointmap_metric)
                pointmap_metric = pointmap_metric.cpu().permute(1, 2, 0)
                raw_view_pointmaps.append(pointmap_metric.numpy())

            ss_input_dict = pipeline.preprocess_image(
                rgba_image_pil, pipeline.ss_preprocessor, pointmap=pointmap
            )
            slat_input_dict = pipeline.preprocess_image(
                rgba_image_pil, pipeline.slat_preprocessor
            )
        else:
            if ext_pointmap is not None:
                logger.warning(
                    f"  View {i+1}: External pointmap provided but pipeline doesn't support it (not InferencePipelinePointMap)"
                )
            ss_input_dict = pipeline.preprocess_image(rgba_image_pil, pipeline.ss_preprocessor)
            slat_input_dict = pipeline.preprocess_image(rgba_image_pil, pipeline.slat_preprocessor)

        view_ss_input_dicts.append(ss_input_dict)
        view_slat_input_dicts.append(slat_input_dict)

    profile.payload["preprocess"] = {
        "seconds": time.perf_counter() - preprocess_start,
    }
    return PreparedViews(
        num_views=num_views,
        view_ss_input_dicts=view_ss_input_dicts,
        view_slat_input_dicts=view_slat_input_dicts,
        raw_view_pointmaps=raw_view_pointmaps,
        runtime_frame_keys=runtime_frame_keys,
    )


def run_stage1(
    pipeline,
    *,
    prepared: PreparedViews,
    stage1_inference_steps: Optional[int],
    use_stage1_distillation: bool,
    mode: Literal["stochastic", "multidiffusion"],
    ss_weighting: bool,
    ss_attention_layer: int,
    ss_weight_source: str,
    ss_jam_alpha: float,
    ss_jam_kappa: float,
    ss_uniform_blend: float,
    ss_min_weight: float,
    ss_warmup_steps: int,
    ss_patch_start: int,
    ss_patch_end: int,
    profile: RunProfile,
) -> Stage1Outputs:
    logger.info("Stage 1: Sampling sparse structure...")
    stage1_start = time.perf_counter()
    ss_return_dict = pipeline.sample_sparse_structure_multi_view(
        prepared.view_ss_input_dicts,
        inference_steps=stage1_inference_steps,
        use_distillation=use_stage1_distillation,
        mode=mode,
        ss_weighting=ss_weighting,
        ss_attention_layer=ss_attention_layer,
        ss_weight_source=ss_weight_source,
        ss_jam_alpha=ss_jam_alpha,
        ss_jam_kappa=ss_jam_kappa,
        ss_uniform_blend=ss_uniform_blend,
        ss_min_weight=ss_min_weight,
        ss_warmup_steps=ss_warmup_steps,
        ss_patch_start=ss_patch_start,
        ss_patch_end=ss_patch_end,
    )
    profile.payload["stage1"] = {
        "seconds": time.perf_counter() - stage1_start,
    }

    pointmap_scale = prepared.view_ss_input_dicts[0].get("pointmap_scale", None)
    pointmap_shift = prepared.view_ss_input_dicts[0].get("pointmap_shift", None)
    ss_return_dict.update(
        pipeline.pose_decoder(
            ss_return_dict,
            scene_scale=pointmap_scale,
            scene_shift=pointmap_shift,
        )
    )
    ss_return_dict["pointmap_scale"] = pointmap_scale
    ss_return_dict["pointmap_shift"] = pointmap_shift

    if "all_view_poses_raw" in ss_return_dict:
        all_view_poses_decoded = pipeline._decode_all_view_poses(
            ss_return_dict["all_view_poses_raw"],
            prepared.view_ss_input_dicts,
        )
        ss_return_dict["all_view_poses_decoded"] = all_view_poses_decoded
        logger.info(f"[Multi-view] Decoded poses for {len(all_view_poses_decoded)} views")

    if "scale" in ss_return_dict:
        logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']}")
        ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

    return Stage1Outputs(
        ss_return_dict=ss_return_dict,
        coords=ss_return_dict["coords"],
    )


def finalize_stage1_only(
    pipeline,
    *,
    prepared: PreparedViews,
    stage1: Stage1Outputs,
    profile: RunProfile,
    total_start: float,
) -> dict:
    profile.payload["total_seconds"] = time.perf_counter() - total_start
    pipeline._last_run_multi_view_profile = profile.payload
    logger.info("Finished!")
    result = dict(stage1.ss_return_dict)
    result["voxel"] = result["coords"][:, 1:] / 64 - 0.5
    return result


def run_stage2(
    pipeline,
    *,
    prepared: PreparedViews,
    stage1: Stage1Outputs,
    stage2_inference_steps: Optional[int],
    use_stage2_distillation: bool,
    mode: Literal["stochastic", "multidiffusion"],
    weighting_config: Optional[Any],
    save_stage2_init: bool,
    save_stage2_init_path: Optional[Any],
    profile: RunProfile,
) -> Stage2Outputs:
    logger.info("Stage 2: Sampling structured latent...")
    stage2_start = time.perf_counter()
    weight_manager = None
    use_weighted_stage2 = (
        weighting_config is not None and bool(weighting_config.weight_metrics)
    )
    if use_weighted_stage2:
        logger.info("Using weighted multi-view fusion")

        slat, weight_manager = pipeline.sample_slat_multi_view_weighted(
            prepared.view_slat_input_dicts,
            stage1.coords,
            inference_steps=stage2_inference_steps,
            use_distillation=use_stage2_distillation,
            weighting_config=weighting_config,
            save_stage2_init=save_stage2_init,
            save_stage2_init_path=save_stage2_init_path,
        )
    else:
        slat = pipeline.sample_slat_multi_view(
            prepared.view_slat_input_dicts,
            stage1.coords,
            inference_steps=stage2_inference_steps,
            use_distillation=use_stage2_distillation,
            mode=mode,
        )

    profile.payload["stage2"] = {
        "seconds": time.perf_counter() - stage2_start,
    }
    return Stage2Outputs(slat=slat, weight_manager=weight_manager)


def assemble_result(
    pipeline,
    *,
    prepared: PreparedViews,
    stage1: Stage1Outputs,
    stage2: Stage2Outputs,
    decode_formats: Optional[List[str]],
    with_mesh_postprocess: bool,
    with_texture_baking: bool,
    use_vertex_color: bool,
    profile: RunProfile,
    total_start: float,
) -> dict:
    outputs = pipeline.decode_slat(
        stage2.slat,
        pipeline.decode_formats if decode_formats is None else decode_formats,
    )
    outputs = pipeline.postprocess_slat_output(
        outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
    )
    logger.info("Finished!")

    result = {
        **stage1.ss_return_dict,
        **outputs,
        "view_ss_input_dicts": prepared.view_ss_input_dicts,
    }
    if prepared.raw_view_pointmaps:
        result["raw_view_pointmaps"] = prepared.raw_view_pointmaps
    if stage2.weight_manager is not None:
        result["weight_manager"] = stage2.weight_manager

    profile.payload["total_seconds"] = time.perf_counter() - total_start
    pipeline._last_run_multi_view_profile = profile.payload
    return result
