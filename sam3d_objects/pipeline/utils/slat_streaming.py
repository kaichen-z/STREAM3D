from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_map_only, tree_unflatten


def _detach_cpu_tree(tree):
    return tree_map_only(
        torch.Tensor,
        lambda x: x.detach().cpu().to(torch.float32).contiguous(),
        tree,
    )


def _stack_view_weight_dict(weights: Dict[int, torch.Tensor]) -> torch.Tensor:
    view_indices = sorted(int(view_idx) for view_idx in weights.keys())
    if not view_indices:
        raise ValueError("Cannot stack an empty view-weight dict.")
    return torch.stack(
        [weights[int(view_idx)].detach().cpu().to(torch.float32).contiguous() for view_idx in view_indices],
        dim=0,
    )


def _unstack_view_weight_tensor(weights: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {
        int(view_idx): weights[int(view_idx)].detach().cpu().to(torch.float32).contiguous()
        for view_idx in range(int(weights.shape[0]))
    }


def _align_prev_slat_view_weights_to_current_coords(
    *,
    prev_weights: Optional[Dict[int, torch.Tensor]],
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> Tuple[Optional[Dict[int, torch.Tensor]], Optional[Dict[int, torch.Tensor]], Optional[torch.Tensor]]:
    if prev_weights is None or prev_coords is None:
        return None, None, None
    stacked = _stack_view_weight_dict(prev_weights)
    aligned, prev_only, prev_only_coords, _ = _align_sparse_row_domain_to_current_coords(
        prev_rows=stacked,
        prev_coords=prev_coords,
        current_coords=current_coords,
    )
    aligned_dict = _unstack_view_weight_tensor(aligned)
    prev_only_dict = None if prev_only is None else _unstack_view_weight_tensor(prev_only)
    return aligned_dict, prev_only_dict, prev_only_coords


def _build_sparse_coord_overlap_mask(
    *,
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> torch.Tensor:
    return _build_sparse_coord_overlap_index_map(
        prev_coords=prev_coords,
        current_coords=current_coords,
    )["current_overlap_mask"]


def _build_sparse_coord_overlap_index_map(
    *,
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> Dict[str, Any]:
    current_coords_cpu = current_coords.detach().cpu().to(torch.int64).contiguous()
    current_row_count = int(current_coords_cpu.shape[0])
    empty_long = torch.empty((0,), dtype=torch.int64)
    if prev_coords is None:
        return {
            "current_overlap_mask": torch.zeros((current_row_count,), dtype=torch.bool),
            "current_overlap_indices": empty_long.clone(),
            "prev_overlap_indices": empty_long.clone(),
            "prev_only_indices": empty_long.clone(),
            "stats": {
                "current_row_count": current_row_count,
                "overlap_row_count": 0,
                "prev_only_row_count": 0,
            },
        }

    prev_coords_cpu = prev_coords.detach().cpu().to(torch.int64).contiguous()
    current_lookup = {
        tuple(int(v) for v in row.tolist()): idx
        for idx, row in enumerate(current_coords_cpu)
    }
    current_overlap_mask = torch.zeros((current_row_count,), dtype=torch.bool)
    current_overlap_indices: List[int] = []
    prev_overlap_indices: List[int] = []
    matched_prev_mask = torch.zeros((int(prev_coords_cpu.shape[0]),), dtype=torch.bool)
    for prev_idx, row in enumerate(prev_coords_cpu):
        current_idx = current_lookup.get(tuple(int(v) for v in row.tolist()))
        if current_idx is None:
            continue
        current_overlap_mask[int(current_idx)] = True
        current_overlap_indices.append(int(current_idx))
        prev_overlap_indices.append(int(prev_idx))
        matched_prev_mask[int(prev_idx)] = True

    prev_only_indices = (~matched_prev_mask).nonzero(as_tuple=False).view(-1).to(torch.int64)
    return {
        "current_overlap_mask": current_overlap_mask,
        "current_overlap_indices": (
            torch.tensor(current_overlap_indices, dtype=torch.int64)
            if current_overlap_indices
            else empty_long.clone()
        ),
        "prev_overlap_indices": (
            torch.tensor(prev_overlap_indices, dtype=torch.int64)
            if prev_overlap_indices
            else empty_long.clone()
        ),
        "prev_only_indices": prev_only_indices,
        "stats": {
            "current_row_count": current_row_count,
            "overlap_row_count": int(len(current_overlap_indices)),
            "prev_only_row_count": int(prev_only_indices.numel()),
        },
    }


def _coerce_sparse_prediction_rows(prediction: torch.Tensor, expected_row_count: int) -> Tuple[torch.Tensor, bool]:
    prediction_cpu = prediction.detach().cpu().to(torch.float32).contiguous()
    if prediction_cpu.dim() >= 2 and int(prediction_cpu.shape[1]) == int(expected_row_count):
        return prediction_cpu, False
    if prediction_cpu.dim() >= 1 and int(prediction_cpu.shape[0]) == int(expected_row_count):
        return prediction_cpu.unsqueeze(0).contiguous(), True
    raise ValueError(
        f"Cannot infer sparse-row axis for cached prediction with shape {tuple(prediction_cpu.shape)} "
        f"and expected row count {int(expected_row_count)}."
    )


def _restore_sparse_prediction_rows(prediction_rows: torch.Tensor, squeezed_batch_dim: bool) -> torch.Tensor:
    if squeezed_batch_dim:
        return prediction_rows.squeeze(0).contiguous()
    return prediction_rows.contiguous()


def _align_cached_velocity_entry_to_current_coords(
    *,
    cache_entry: Any,
    prev_coords: torch.Tensor,
    current_coords: torch.Tensor,
) -> Tuple[Any, Optional[Any], Optional[torch.Tensor]]:
    if isinstance(cache_entry, (list, tuple)):
        stacked = torch.stack(
            [
                pred.detach().cpu().to(torch.float32).contiguous().squeeze(0)
                for pred in cache_entry
            ],
            dim=0,
        )
        aligned, prev_only, prev_only_coords, _ = _align_sparse_row_domain_to_current_coords(
            prev_rows=stacked,
            prev_coords=prev_coords,
            current_coords=current_coords,
        )
        aligned_entry = [
            aligned[int(view_idx)].unsqueeze(0).contiguous()
            for view_idx in range(int(aligned.shape[0]))
        ]
        prev_only_entry = None
        if prev_only is not None:
            prev_only_entry = [
                prev_only[int(view_idx)].unsqueeze(0).contiguous()
                for view_idx in range(int(prev_only.shape[0]))
            ]
        return aligned_entry, prev_only_entry, prev_only_coords

    leaves, spec = tree_flatten(cache_entry)
    aligned_leaves: List[Any] = []
    prev_only_leaves: List[Any] = []
    prev_only_coords_out: Optional[torch.Tensor] = None
    for leaf in leaves:
        if not torch.is_tensor(leaf):
            aligned_leaves.append(leaf)
            prev_only_leaves.append(leaf)
            continue
        leaf_rows, squeezed_batch_dim = _coerce_sparse_prediction_rows(leaf, int(prev_coords.shape[0]))
        aligned_rows, prev_only_rows, prev_only_coords, _ = _align_sparse_row_domain_to_current_coords(
            prev_rows=leaf_rows,
            prev_coords=prev_coords,
            current_coords=current_coords,
        )
        aligned_leaves.append(_restore_sparse_prediction_rows(aligned_rows, squeezed_batch_dim))
        prev_only_leaves.append(
            None if prev_only_rows is None else _restore_sparse_prediction_rows(prev_only_rows, squeezed_batch_dim)
        )
        if prev_only_rows is not None:
            prev_only_coords_out = prev_only_coords
    aligned_entry = tree_unflatten(aligned_leaves, spec)
    prev_only_entry = None if prev_only_coords_out is None else tree_unflatten(prev_only_leaves, spec)
    return aligned_entry, prev_only_entry, prev_only_coords_out


def _align_prev_slat_velocity_cache_to_current_coords(
    *,
    prev_cache: Dict[int, Any],
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> Tuple[Dict[int, Any], Dict[int, Any], Optional[torch.Tensor]]:
    aligned_cache: Dict[int, Any] = {}
    prev_only_cache: Dict[int, Any] = {}
    prev_only_coords_out: Optional[torch.Tensor] = None
    if prev_coords is None:
        return aligned_cache, prev_only_cache, prev_only_coords_out

    for step_idx, cache_entry in prev_cache.items():
        aligned_entry, prev_only_entry, prev_only_coords = _align_cached_velocity_entry_to_current_coords(
            cache_entry=cache_entry,
            prev_coords=prev_coords,
            current_coords=current_coords,
        )
        aligned_cache[int(step_idx)] = aligned_entry
        if prev_only_entry is not None:
            prev_only_cache[int(step_idx)] = prev_only_entry
            prev_only_coords_out = prev_only_coords
    return aligned_cache, prev_only_cache, prev_only_coords_out


def _align_prev_slat_latent_state_cache_to_current_coords(
    *,
    prev_cache: Dict[int, Any],
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> Tuple[Dict[int, Any], Dict[int, Any], Optional[torch.Tensor]]:
    aligned_cache: Dict[int, Any] = {}
    prev_only_cache: Dict[int, Any] = {}
    prev_only_coords_out: Optional[torch.Tensor] = None
    if prev_coords is None:
        return aligned_cache, prev_only_cache, prev_only_coords_out

    for step_idx, cache_entry in prev_cache.items():
        aligned_entry, prev_only_entry, prev_only_coords = _align_cached_velocity_entry_to_current_coords(
            cache_entry=cache_entry,
            prev_coords=prev_coords,
            current_coords=current_coords,
        )
        aligned_cache[int(step_idx)] = aligned_entry
        if prev_only_entry is not None:
            prev_only_cache[int(step_idx)] = prev_only_entry
            prev_only_coords_out = prev_only_coords
    return aligned_cache, prev_only_cache, prev_only_coords_out


def _merge_slat_view_weight_union(
    *,
    current_weights: Dict[int, torch.Tensor],
    current_coords: torch.Tensor,
    prev_only_weights: Optional[Dict[int, torch.Tensor]],
    prev_only_coords: Optional[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    if prev_only_weights is None or prev_only_coords is None or prev_only_coords.numel() == 0:
        return _detach_cpu_tree(current_weights)
    stacked_current = _stack_view_weight_dict(current_weights)
    stacked_prev_only = _stack_view_weight_dict(prev_only_weights)
    merged_weights, _, _ = _merge_sparse_row_domain_union(
        current_rows=stacked_current,
        current_coords=current_coords,
        prev_only_rows=stacked_prev_only,
        prev_only_coords=prev_only_coords,
    )
    return _unstack_view_weight_tensor(merged_weights)


def _merge_cached_velocity_entry_union(
    *,
    current_entry: Any,
    current_coords: torch.Tensor,
    prev_only_entry: Any,
    prev_only_coords: torch.Tensor,
) -> Any:
    if isinstance(current_entry, (list, tuple)):
        current_stacked = torch.stack(
            [
                pred.detach().cpu().to(torch.float32).contiguous().squeeze(0)
                for pred in current_entry
            ],
            dim=0,
        )
        prev_only_stacked = torch.stack(
            [
                pred.detach().cpu().to(torch.float32).contiguous().squeeze(0)
                for pred in prev_only_entry
            ],
            dim=0,
        )
        merged_rows, _, _ = _merge_sparse_row_domain_union(
            current_rows=current_stacked,
            current_coords=current_coords,
            prev_only_rows=prev_only_stacked,
            prev_only_coords=prev_only_coords,
        )
        return [
            merged_rows[int(view_idx)].unsqueeze(0).contiguous()
            for view_idx in range(int(merged_rows.shape[0]))
        ]

    current_leaves, spec = tree_flatten(current_entry)
    prev_only_leaves, prev_spec = tree_flatten(prev_only_entry)
    if spec != prev_spec:
        raise ValueError("Cached velocity tree structure mismatch while merging SLAT union cache.")

    merged_leaves: List[Any] = []
    for current_leaf, prev_only_leaf in zip(current_leaves, prev_only_leaves):
        if not torch.is_tensor(current_leaf):
            merged_leaves.append(current_leaf)
            continue
        current_rows, squeezed_batch_dim = _coerce_sparse_prediction_rows(current_leaf, int(current_coords.shape[0]))
        prev_only_rows, _ = _coerce_sparse_prediction_rows(prev_only_leaf, int(prev_only_coords.shape[0]))
        merged_rows, _, _ = _merge_sparse_row_domain_union(
            current_rows=current_rows,
            current_coords=current_coords,
            prev_only_rows=prev_only_rows,
            prev_only_coords=prev_only_coords,
        )
        merged_leaves.append(_restore_sparse_prediction_rows(merged_rows, squeezed_batch_dim))
    return tree_unflatten(merged_leaves, spec)


def _merge_slat_velocity_cache_union(
    *,
    current_cache: Dict[int, Any],
    current_coords: torch.Tensor,
    prev_only_cache: Dict[int, Any],
    prev_only_coords: Optional[torch.Tensor],
) -> Dict[int, Any]:
    if prev_only_coords is None or prev_only_coords.numel() == 0:
        return {
            int(step_idx): _detach_cpu_tree(cache_entry)
            for step_idx, cache_entry in current_cache.items()
        }
    merged_cache: Dict[int, Any] = {}
    for step_idx, cache_entry in current_cache.items():
        prev_only_entry = prev_only_cache.get(int(step_idx))
        if prev_only_entry is None:
            merged_cache[int(step_idx)] = _detach_cpu_tree(cache_entry)
            continue
        merged_cache[int(step_idx)] = _merge_cached_velocity_entry_union(
            current_entry=cache_entry,
            current_coords=current_coords,
            prev_only_entry=prev_only_entry,
            prev_only_coords=prev_only_coords,
        )
    return merged_cache


def _merge_slat_latent_state_cache_union(
    *,
    current_cache: Dict[int, Any],
    current_coords: torch.Tensor,
    prev_only_cache: Dict[int, Any],
    prev_only_coords: Optional[torch.Tensor],
) -> Dict[int, Any]:
    if prev_only_coords is None or prev_only_coords.numel() == 0:
        return {
            int(step_idx): _detach_cpu_tree(cache_entry)
            for step_idx, cache_entry in current_cache.items()
        }
    merged_cache: Dict[int, Any] = {}
    for step_idx, cache_entry in current_cache.items():
        prev_only_entry = prev_only_cache.get(int(step_idx))
        if prev_only_entry is None:
            merged_cache[int(step_idx)] = _detach_cpu_tree(cache_entry)
            continue
        merged_cache[int(step_idx)] = _merge_cached_velocity_entry_union(
            current_entry=cache_entry,
            current_coords=current_coords,
            prev_only_entry=prev_only_entry,
            prev_only_coords=prev_only_coords,
        )
    return merged_cache


def _align_sparse_row_domain_to_current_coords(
    *,
    prev_rows: Optional[torch.Tensor],
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, int]]:
    if prev_rows is None or prev_coords is None:
        raise ValueError("prev_rows and prev_coords must both be provided.")

    prev_coords_cpu = prev_coords.detach().cpu().to(torch.int64).contiguous()
    current_coords_cpu = current_coords.detach().cpu().to(torch.int64).contiguous()
    prev_rows_cpu = prev_rows.detach().cpu().to(torch.float32).contiguous()
    if prev_rows_cpu.dim() < 2:
        raise ValueError(f"Expected sparse row tensor with dim >= 2, got shape {tuple(prev_rows_cpu.shape)}")

    current_lookup = {
        tuple(int(v) for v in row.tolist()): idx
        for idx, row in enumerate(current_coords_cpu)
    }
    aligned_shape = list(prev_rows_cpu.shape)
    aligned_shape[1] = int(current_coords_cpu.shape[0])
    aligned = torch.zeros(aligned_shape, dtype=prev_rows_cpu.dtype)
    matched_prev_indices: List[int] = []
    overlap_count = 0
    for prev_idx, row in enumerate(prev_coords_cpu):
        current_idx = current_lookup.get(tuple(int(v) for v in row.tolist()))
        if current_idx is None:
            continue
        aligned[:, current_idx, ...] = prev_rows_cpu[:, prev_idx, ...]
        matched_prev_indices.append(int(prev_idx))
        overlap_count += 1

    if overlap_count == int(prev_coords_cpu.shape[0]):
        prev_only_coords = None
        prev_only_rows = None
    else:
        matched_mask = torch.zeros(prev_coords_cpu.shape[0], dtype=torch.bool)
        if matched_prev_indices:
            matched_mask[torch.tensor(matched_prev_indices, dtype=torch.long)] = True
        prev_only_coords = prev_coords_cpu[~matched_mask].clone()
        prev_only_rows = prev_rows_cpu[:, ~matched_mask, ...].clone()

    return aligned, prev_only_rows, prev_only_coords, {
        "current_row_count": int(current_coords_cpu.shape[0]),
        "overlap_row_count": int(overlap_count),
        "prev_only_row_count": 0 if prev_only_coords is None else int(prev_only_coords.shape[0]),
    }


def _compute_sparse_coord_canonical_order(coords: torch.Tensor) -> torch.Tensor:
    coords_cpu = coords.detach().cpu().to(torch.int64).contiguous()
    if coords_cpu.dim() != 2:
        raise ValueError(f"Expected sparse coords with shape [N, D], got {tuple(coords_cpu.shape)}.")
    row_count = int(coords_cpu.shape[0])
    if row_count <= 1:
        return torch.arange(row_count, dtype=torch.long)
    order = np.lexsort(
        tuple(coords_cpu[:, dim].numpy() for dim in range(int(coords_cpu.shape[1]) - 1, -1, -1))
    )
    return torch.from_numpy(order.astype(np.int64, copy=False))


def _canonicalize_sparse_row_domain(
    *,
    rows: torch.Tensor,
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows_cpu = rows.detach().cpu().contiguous()
    coords_cpu = coords.detach().cpu().contiguous()
    if rows_cpu.dim() < 2:
        raise ValueError(f"Expected sparse row tensor with dim >= 2, got shape {tuple(rows_cpu.shape)}.")
    if int(rows_cpu.shape[1]) != int(coords_cpu.shape[0]):
        raise ValueError(
            f"Sparse row/domain mismatch: rows shape {tuple(rows_cpu.shape)} vs coords shape {tuple(coords_cpu.shape)}."
        )
    order = _compute_sparse_coord_canonical_order(coords_cpu)
    if order.numel() <= 1:
        return rows_cpu, coords_cpu
    identity = torch.arange(int(order.numel()), dtype=torch.long)
    if torch.equal(order, identity):
        return rows_cpu, coords_cpu
    return (
        rows_cpu.index_select(1, order).contiguous(),
        coords_cpu.index_select(0, order).contiguous(),
    )


def _merge_sparse_row_domain_union(
    *,
    current_rows: torch.Tensor,
    current_coords: torch.Tensor,
    prev_only_rows: Optional[torch.Tensor],
    prev_only_coords: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    current_rows_cpu = current_rows.detach().cpu().to(torch.float32).contiguous()
    current_coords_cpu = current_coords.detach().cpu().to(torch.int32).contiguous()
    if prev_only_rows is None or prev_only_coords is None or prev_only_coords.numel() == 0:
        canonical_rows, canonical_coords = _canonicalize_sparse_row_domain(
            rows=current_rows_cpu,
            coords=current_coords_cpu,
        )
        return canonical_rows, canonical_coords, {
            "current_row_count": int(current_coords_cpu.shape[0]),
            "merged_row_count": int(current_coords_cpu.shape[0]),
            "appended_prev_only_row_count": 0,
        }

    current_lookup = {
        tuple(int(v) for v in row.tolist()): idx
        for idx, row in enumerate(current_coords_cpu)
    }
    append_coords: List[torch.Tensor] = []
    append_rows: List[torch.Tensor] = []
    appended = 0
    for prev_idx, row in enumerate(prev_only_coords):
        key = tuple(int(v) for v in row.tolist())
        if key in current_lookup:
            continue
        append_coords.append(row.to(dtype=torch.int32).unsqueeze(0))
        append_rows.append(prev_only_rows[:, prev_idx : prev_idx + 1, ...])
        appended += 1

    if append_coords:
        merged_coords = torch.cat([current_coords_cpu] + append_coords, dim=0)
        merged_rows = torch.cat([current_rows_cpu] + append_rows, dim=1)
    else:
        merged_coords = current_coords_cpu
        merged_rows = current_rows_cpu

    merged_rows, merged_coords = _canonicalize_sparse_row_domain(
        rows=merged_rows,
        coords=merged_coords,
    )

    return merged_rows, merged_coords, {
        "current_row_count": int(current_coords_cpu.shape[0]),
        "merged_row_count": int(merged_coords.shape[0]),
        "appended_prev_only_row_count": int(appended),
    }


def _align_prev_slat_state_to_current_coords(
    *,
    prev_feats: Optional[torch.Tensor],
    prev_coords: Optional[torch.Tensor],
    current_coords: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, int]]:
    if prev_feats is None or prev_coords is None:
        aligned = torch.zeros(
            (1, int(current_coords.shape[0]), 8),
            dtype=torch.float32,
        )
        return aligned, None, None, {
            "current_row_count": int(current_coords.shape[0]),
            "overlap_row_count": 0,
            "prev_only_row_count": 0,
        }
    return _align_sparse_row_domain_to_current_coords(
        prev_rows=prev_feats,
        prev_coords=prev_coords,
        current_coords=current_coords,
    )


def _merge_slat_state_union(
    *,
    current_feats: torch.Tensor,
    current_coords: torch.Tensor,
    prev_only_feats: Optional[torch.Tensor],
    prev_only_coords: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    return _merge_sparse_row_domain_union(
        current_rows=current_feats,
        current_coords=current_coords,
        prev_only_rows=prev_only_feats,
        prev_only_coords=prev_only_coords,
    )
