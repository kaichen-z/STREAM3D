from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, List, Dict

from PIL import Image
import numpy as np
from loguru import logger


def load_images_and_masks_for_chunk(
    image_paths: Sequence[Path],
    mask_root: Path,
    image_stems: Sequence[str],
):
    def load_image(path: Path) -> np.ndarray:
        image = Image.open(path)
        return np.asarray(image).astype(np.uint8)

    def load_mask_from_rgba(path: Path) -> np.ndarray:
        image = Image.open(path)
        image_array = np.asarray(image)
        if image.mode == "RGBA" and image_array.ndim == 3 and image_array.shape[2] >= 4:
            return image_array[..., 3] > 0
        if image.mode == "RGB":
            logger.warning(
                f"Mask file {path} is RGB, not RGBA. Using all pixels as mask."
            )
            return np.ones((image_array.shape[0], image_array.shape[1]), dtype=bool)
        logger.warning(
            f"Unexpected mask mode {image.mode} for {path}. Using all pixels as mask."
        )
        return np.ones((image_array.shape[0], image_array.shape[1]), dtype=bool)

    images = [load_image(image_path) for image_path in image_paths]
    masks = [load_mask_from_rgba(mask_root / f"{stem}.png") for stem in image_stems]
    return images, masks


def build_unique_warmup_chunk_spec(
    chunk_spec: dict[str, Any],
    *,
    seen_global_frame_indices: Sequence[int] | set[int],
) -> tuple[dict[str, Any], list[int]]:
    seen_indices = {int(index) for index in seen_global_frame_indices}
    selected_positions: list[int] = []
    skipped_indices: list[int] = []
    for position, global_frame_index in enumerate(chunk_spec["global_frame_indices"]):
        index = int(global_frame_index)
        if index in seen_indices:
            skipped_indices.append(index)
        else:
            selected_positions.append(int(position))

    unique_spec = dict(chunk_spec)
    unique_spec["frame_paths"] = [
        chunk_spec["frame_paths"][position] for position in selected_positions
    ]
    unique_spec["stems"] = [
        chunk_spec["stems"][position] for position in selected_positions
    ]
    unique_spec["names"] = [
        chunk_spec["names"][position] for position in selected_positions
    ]
    unique_spec["global_frame_indices"] = [
        int(chunk_spec["global_frame_indices"][position])
        for position in selected_positions
    ]
    unique_spec["num_views"] = len(selected_positions)
    return unique_spec, skipped_indices


def build_chunk_info(
    *,
    chunk_spec: dict[str, Any],
    loaded_image_names: Sequence[str],
    runtime_frame_keys: Sequence[str],
    prev_view_index_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    chunk_info = {
        "chunk_name": str(chunk_spec["chunk_name"]),
        "chunk_index": int(chunk_spec["chunk_index"]),
        "image_paths": [str(path.resolve()) for path in chunk_spec["frame_paths"]],
        "image_names": [path.name for path in chunk_spec["frame_paths"]],
        "loaded_image_names": [str(name) for name in loaded_image_names],
        "runtime_frame_keys": [str(key) for key in runtime_frame_keys],
        "global_frame_indices": [
            int(index) for index in chunk_spec["global_frame_indices"]
        ],
        "num_views": len(loaded_image_names),
    }
    if prev_view_index_map is not None:
        chunk_info["prev_view_index_map"] = {
            int(current_idx): int(prev_idx)
            for current_idx, prev_idx in prev_view_index_map.items()
        }
    return chunk_info


def build_prev_view_index_map(
    *,
    prev_loaded_image_names: Sequence[str],
    current_loaded_image_names: Sequence[str],
) -> Dict[int, int]:
    prev_index_by_stem: Dict[str, int] = {}
    for prev_idx, image_name in enumerate(prev_loaded_image_names):
        stem = Path(str(image_name)).stem
        if stem in prev_index_by_stem:
            raise ValueError(
                f"Duplicate previous-chunk image stem '{stem}' cannot be mapped unambiguously."
            )
        prev_index_by_stem[stem] = int(prev_idx)

    current_view_to_prev_view: Dict[int, int] = {}
    seen_current_stems: set[str] = set()
    for current_idx, image_name in enumerate(current_loaded_image_names):
        stem = Path(str(image_name)).stem
        if stem in seen_current_stems:
            raise ValueError(
                f"Duplicate current-chunk image stem '{stem}' cannot be mapped unambiguously."
            )
        seen_current_stems.add(stem)
        prev_idx = prev_index_by_stem.get(stem)
        if prev_idx is not None:
            current_view_to_prev_view[int(current_idx)] = int(prev_idx)

    return current_view_to_prev_view
