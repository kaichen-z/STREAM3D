from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from streaming.backend.selector.selector import (
    ViewConditionCacheConfig,
    ViewConditionCacheState,
)


def make_view_condition_cache_state(
    config: ViewConditionCacheConfig,
) -> ViewConditionCacheState:
    return ViewConditionCacheState()


def build_seen_view_pool(chunk_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    view_by_global_index: Dict[int, Dict[str, Any]] = {}
    for chunk_spec in chunk_specs:
        stems = list(chunk_spec["stems"])
        global_frame_indices = [
            int(index) for index in chunk_spec["global_frame_indices"]
        ]
        for local_view_index, (stem, global_frame_index) in enumerate(
            zip(stems, global_frame_indices)
        ):
            if global_frame_index in view_by_global_index:
                continue
            record = {
                "frame_name": str(stem),
                "chunk_index": int(chunk_spec["chunk_index"]),
                "chunk_name": str(chunk_spec["chunk_name"]),
                "local_view_index": int(local_view_index),
                "global_frame_index": int(global_frame_index),
            }
            if "frame_paths" in chunk_spec:
                frame_path = Path(chunk_spec["frame_paths"][local_view_index])
                record["image_name"] = frame_path.name
            view_by_global_index[global_frame_index] = record

    return [
        view_by_global_index[global_frame_index]
        for global_frame_index in sorted(view_by_global_index)
    ]


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


def build_selected_runtime_view_spec(
    *,
    image_files: Sequence[Path],
    selected_views: Sequence[Dict[str, Any]],
    chunk_index: int,
    chunk_name: str,
) -> Dict[str, Any]:
    global_indices = [int(view["global_frame_index"]) for view in selected_views]
    frame_paths = [Path(image_files[index]) for index in global_indices]
    return {
        "chunk_index": int(chunk_index),
        "chunk_name": str(chunk_name),
        "frame_paths": frame_paths,
        "stems": [path.stem for path in frame_paths],
        "names": [path.name for path in frame_paths],
        "global_frame_indices": global_indices,
        "num_views": len(frame_paths),
    }


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
