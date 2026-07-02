from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image


def build_chunk_crop_targets(
    *,
    loaded_image_names: Sequence[str],
) -> List[Dict[str, Any]]:
    loaded_stems = [str(name) for name in loaded_image_names]

    targets: List[Dict[str, Any]] = []

    def build_crop_filename(stem: str, order_index: int) -> str:
        return f"{Path(stem).stem}_view{int(order_index):02d}_crop.png"

    def append_target(
        *,
        label: str,
        stem: str,
        order_index: int,
    ) -> None:
        if stem not in loaded_stems:
            raise KeyError(
                f"Crop view target '{stem}' is not present in the runtime chunk views: "
                f"{loaded_stems}"
            )
        targets.append(
            {
                "label": label,
                "stem": stem,
                "order_index": int(order_index),
                "crop_filename": build_crop_filename(stem, order_index),
            }
        )

    for order_index, stem in enumerate(loaded_stems):
        append_target(
            label=f"view_{int(order_index):02d}",
            stem=stem,
            order_index=order_index,
        )

    return targets


def build_chunk_crop_views(
    *,
    chunk_input: Dict[str, Any],
    chunk_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    loaded_names = list(chunk_info["loaded_image_names"])
    if not loaded_names:
        raise ValueError("Cannot build chunk crop views for an empty chunk.")

    target_specs = build_chunk_crop_targets(
        loaded_image_names=loaded_names,
    )
    loaded_name_to_index = {str(stem): idx for idx, stem in enumerate(loaded_names)}

    crop_views: List[Dict[str, Any]] = []
    for target_spec in target_specs:
        stem = str(target_spec["stem"])
        view_idx = loaded_name_to_index[stem]
        crop_views.append(
            {
                "label": str(target_spec["label"]),
                "image_name": stem,
                "view_index": int(target_spec["order_index"]),
                "image": np.asarray(
                    chunk_input["view_images"][view_idx], dtype=np.uint8
                ),
                "mask": np.asarray(chunk_input["view_masks"][view_idx], dtype=bool),
                "crop_filename": str(target_spec["crop_filename"]),
            }
        )
    return crop_views


def compute_mask_bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    height, width = mask_bool.shape[:2]
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, width, height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def save_mask_crop_rgba(
    output_dir: Path,
    *,
    image: np.ndarray,
    mask: np.ndarray,
    filename: str,
) -> Dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    x0, y0, x1, y1 = compute_mask_bbox_xyxy(mask_bool)
    crop_image = np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1, :3].copy()
    crop_mask = mask_bool[y0:y1, x0:x1]

    rgba = np.zeros((crop_image.shape[0], crop_image.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_image
    rgba[..., :3][~crop_mask] = 0
    rgba[..., 3] = crop_mask.astype(np.uint8) * 255

    output_path = output_dir / filename
    Image.fromarray(rgba, mode="RGBA").save(output_path)
    return {
        "filename": filename,
        "bbox_xyxy": [x0, y0, x1, y1],
        "crop_size_hw": [int(rgba.shape[0]), int(rgba.shape[1])],
    }


def get_result_gaussian(result: Dict[str, Any]) -> Optional[Any]:
    gaussian = result.get("gs")
    if gaussian is not None:
        return gaussian
    gaussian_list = result.get("gaussian")
    if isinstance(gaussian_list, list) and gaussian_list:
        return gaussian_list[0]
    return None


def requested_decode_formats(decode_formats: Optional[Sequence[str]]) -> set[str]:
    if decode_formats is None:
        return set()
    if isinstance(decode_formats, str):
        return {
            item.strip().lower() for item in decode_formats.split(",") if item.strip()
        }
    return {str(item).strip().lower() for item in decode_formats if str(item).strip()}


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
        return value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def save_chunk_crop_images(
    output_dir: Path,
    *,
    crop_views: Sequence[Dict[str, Any]],
    directory_name: str,
) -> List[str]:
    crop_dir = output_dir / directory_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in crop_dir.glob("*.png"):
        stale_path.unlink()
    if len(crop_views) == 0:
        return []

    saved_files: List[str] = []
    for view in crop_views:
        crop_meta = save_mask_crop_rgba(
            crop_dir,
            image=view["image"],
            mask=view["mask"],
            filename=str(view["crop_filename"]),
        )
        saved_files.append(str(Path(directory_name) / crop_meta["filename"]))

    # TODO: if a trustworthy reconstruction-side visualization returns, keep it as a
    # separate output path instead of reviving the removed debug_render files.
    return saved_files


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _merge_unique_strings(
    existing: Sequence[Any], new_items: Sequence[Any]
) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for item in list(existing) + list(new_items):
        if item is None:
            continue
        token = str(item)
        if token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def update_streaming_result_metadata(
    output_dir: Path,
    *,
    metadata_updates: Dict[str, Any],
    additional_saved_outputs: Sequence[str] = (),
) -> Dict[str, Any]:
    metadata_path = output_dir / "result_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Streaming result metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if additional_saved_outputs:
        metadata["saved_outputs"] = _merge_unique_strings(
            metadata.get("saved_outputs", []),
            additional_saved_outputs,
        )
    for key, value in metadata_updates.items():
        if (
            key in {"saved_outputs", "render_frame_refs", "rendered_view_files"}
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            metadata[key] = _merge_unique_strings(metadata.get(key, []), value)
            continue
        metadata[key] = value
    _write_json(metadata_path, metadata)
    return metadata


def save_streaming_result_bundle(
    output_dir: Path,
    *,
    chunk_info: Dict[str, Any],
    crop_views: Sequence[Dict[str, Any]],
    stage1_crop_views: Optional[Sequence[Dict[str, Any]]] = None,
    stage2_crop_views: Optional[Sequence[Dict[str, Any]]] = None,
    mask_prompt: str,
    result: Dict[str, Any],
    seed: Optional[int] = None,
    stage1_steps: Optional[int] = None,
    stage2_steps: Optional[int] = None,
    decode_formats: Optional[Sequence[str]] = None,
    weighting_metadata: Optional[Dict[str, Any]] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    prev_chunk_id: Optional[int] = None,
    status_label: str = "intermediate",
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []
    if stage2_crop_views is None:
        stage2_crop_views = crop_views
    if stage1_crop_views is None:
        stage1_crop_views = []
    stage1_crop_views = list(stage1_crop_views)
    stage2_crop_views = list(stage2_crop_views)

    if result.get("glb") is not None:
        glb_path = output_dir / "result.glb"
        result["glb"].export(str(glb_path))
        saved_files.append(glb_path.name)
        logger.info(f"Saved GLB to {glb_path}")

    gaussian = get_result_gaussian(result)
    if gaussian is not None:
        ply_path = output_dir / "result.ply"
        gaussian.save_ply(str(ply_path))
        saved_files.append(ply_path.name)
        logger.info(f"Saved Gaussian PLY to {ply_path}")

    requested_formats = requested_decode_formats(decode_formats)
    if "mesh" in requested_formats and "result.glb" not in saved_files:
        raise RuntimeError(
            "decode_formats requested mesh output, but result.glb was not produced."
        )
    if "gaussian" in requested_formats and "result.ply" not in saved_files:
        raise RuntimeError(
            "decode_formats requested Gaussian output, but result.ply was not produced."
        )

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
        value = result.get(key)
        if value is None:
            continue
        params[key] = _as_numpy(value)
    if params:
        params_path = output_dir / "params.npz"
        np.savez(params_path, **params)
        saved_files.append(params_path.name)

    pose_payload: Dict[str, np.ndarray] = {}
    pose_metadata: Dict[str, Any] = {}
    for key in ("scale", "rotation", "translation"):
        value = result.get(key)
        if value is None:
            continue
        value_np = _as_numpy(value)
        pose_payload[key] = value_np
        pose_metadata[key] = value_np.tolist()
    if pose_payload:
        pose_path = output_dir / "result_pose.npz"
        np.savez(pose_path, **pose_payload)
        saved_files.append(pose_path.name)
        pose_metadata["pose_npz"] = pose_path.name

    stage1_crop_files = save_chunk_crop_images(
        output_dir,
        crop_views=stage1_crop_views,
        directory_name="stage1_selected_crops",
    )
    stage2_crop_files = save_chunk_crop_images(
        output_dir,
        crop_views=stage2_crop_views,
        directory_name="stage2_selected_crops",
    )
    saved_files.extend(stage1_crop_files)
    saved_files.extend(stage2_crop_files)
    if "stage1_sparse" in chunk_info:
        saved_files = _merge_unique_strings(
            saved_files,
            chunk_info["stage1_sparse"].get("saved_outputs", []),
        )

    metadata = {
        "image_paths": chunk_info["image_paths"],
        "image_names": chunk_info["image_names"],
        "loaded_image_names": chunk_info["loaded_image_names"],
        "runtime_frame_keys": chunk_info["runtime_frame_keys"],
        "global_frame_indices": chunk_info["global_frame_indices"],
        "num_views": chunk_info["num_views"],
        "mask_prompt": mask_prompt,
        "generated_coord_count": (
            int(result["coords"].shape[0]) if "coords" in result else None
        ),
        "saved_outputs": saved_files + ["result_metadata.json"],
        "prev_chunk_id": prev_chunk_id,
        "status_label": status_label,
        "stage1_selected_crop_view_names": [
            str(item["image_name"]) for item in stage1_crop_views
        ],
        "stage1_selected_crop_labels": [
            str(item["label"]) for item in stage1_crop_views
        ],
        "stage1_selected_crop_files": [
            str(Path("stage1_selected_crops") / str(item["crop_filename"]))
            for item in stage1_crop_views
        ],
        "stage2_selected_crop_view_names": [
            str(item["image_name"]) for item in stage2_crop_views
        ],
        "stage2_selected_crop_labels": [
            str(item["label"]) for item in stage2_crop_views
        ],
        "stage2_selected_crop_files": [
            str(Path("stage2_selected_crops") / str(item["crop_filename"]))
            for item in stage2_crop_views
        ],
        # Backward-compatible aliases: keep pointing to the final reconstruction inputs.
        "input_crop_view_names": [
            str(item["image_name"]) for item in stage2_crop_views
        ],
        "input_crop_labels": [str(item["label"]) for item in stage2_crop_views],
        "input_crop_files": [
            str(Path("stage2_selected_crops") / str(item["crop_filename"]))
            for item in stage2_crop_views
        ],
        **pose_metadata,
    }
    if "prev_view_index_map" in chunk_info:
        metadata["prev_view_index_map"] = chunk_info["prev_view_index_map"]
    if "sliding_window" in chunk_info:
        metadata["sliding_window"] = chunk_info["sliding_window"]
    if "source_chunk" in chunk_info:
        metadata["source_chunk"] = chunk_info["source_chunk"]
    if "view_condition_cache" in chunk_info:
        metadata["view_condition_cache"] = chunk_info["view_condition_cache"]
    if "stage2_view_selection" in chunk_info:
        metadata["stage2_view_selection"] = chunk_info["stage2_view_selection"]
    if "stage1_sparse" in chunk_info:
        metadata["stage1_sparse"] = chunk_info["stage1_sparse"]
    if seed is not None:
        metadata["seed"] = int(seed)
    if stage1_steps is not None:
        metadata["stage1_steps"] = int(stage1_steps)
    if stage2_steps is not None:
        metadata["stage2_steps"] = int(stage2_steps)
    if decode_formats is not None:
        metadata["decode_formats"] = list(decode_formats)
    if alpha is not None:
        metadata["alpha"] = float(alpha)
    if beta is not None:
        metadata["beta"] = float(beta)
    if weighting_metadata is not None:
        metadata.update(weighting_metadata)
    _write_json(output_dir / "result_metadata.json", metadata)
    saved_files.append("result_metadata.json")
    return saved_files
